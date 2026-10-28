#!/usr/bin/env python3
"""
Optional persistent experience memory for the General Agents.

This module is intentionally NOT an agent and exposes no LLM tool to the main
agent. It is a host-side component shared by:

    - general_python_agent.py
    - general_web_agent.py

Its responsibilities are deliberately narrow:

    1. Load a small, human-readable persistent memory.
    2. Render the experiences applicable to the current agent/skill.
    3. Reflect conservatively on a completed interaction AFTER the main answer
       has already been finalized.
    4. Deduplicate, reinforce, update, or add generalized lessons.
    5. Persist validated changes atomically.

The main agent remains responsible for reasoning, tool choice, execution,
recovery, validation, and answering the user.

Design invariants
-----------------

- Experience is opt-in at the parent-agent level.
- When experience is disabled, the parent agent should not instantiate or call
  this manager at all.
- Experience memory is never exposed as a main-agent tool.
- The main agent must never edit experiences.json through execute_python.
- Most completed turns should produce NO memory change.
- Stored lessons are short, generalized, actionable, evidence-supported, and
  non-sensitive.
- Experience is soft guidance, never an absolute rule.
- Reflection failures must never prevent the already-produced main answer from
  being returned.
- /reset in a parent agent should clear conversation state, not this persistent
  memory.

Recommended project layout
--------------------------

    GeneralAgent/
    ├── general_python_agent.py
    ├── general_web_agent.py
    ├── experience_manager.py
    └── agent_memory/
        └── experiences.json

No vector database, embedding model, or retrieval system is used. The memory is
small enough that all applicable lessons can be supplied directly to the model.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from openai import OpenAI


# =============================================================================
# Defaults
# =============================================================================

MEMORY_VERSION = 1

DEFAULT_REFLECTION_MODEL = "gpt-5.6-luna"
DEFAULT_REFLECTION_REASONING_EFFORT = "low"

MAX_GLOBAL_EXPERIENCES = 10
MAX_SKILL_EXPERIENCES = 15
MAX_USER_PREFERENCES = 5
MAX_APPLICABLE_ITEMS = 30
MAX_LESSON_WORDS = 35
MAX_CANDIDATES_PER_TURN = 2

MAX_REFLECTION_EVIDENCE_CHARS = 30_000
MAX_REFLECTION_OUTPUT_TOKENS = 1_400
MAX_CONSOLIDATION_OUTPUT_TOKENS = 2_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY_PATH = SCRIPT_DIR / "agent_memory" / "experiences.json"


Scope = Literal["global", "skill", "user_preference"]
Action = Literal["add", "reinforce", "update", "discard"]


# =============================================================================
# Structured-output schemas for the reflection/consolidation calls
# =============================================================================

REFLECTION_REASON_CODES = (
    "explicit_correction",
    "explicit_preference",
    "observed_recovery",
    "validation_mismatch",
    "reinforcement",
    "duplicate",
    "skill_duplicate",
    "weak_signal",
    "transient",
    "unsupported",
    "sensitive",
    "no_active_skill",
)

POSITIVE_REASON_CODES = {
    "explicit_correction",
    "explicit_preference",
    "observed_recovery",
    "validation_mismatch",
    "reinforcement",
}

REFLECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "maxItems": MAX_CANDIDATES_PER_TURN,
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "reinforce", "update", "discard"],
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["global", "skill", "user_preference"],
                    },
                    "target_id": {
                        "type": ["string", "null"],
                    },
                    "lesson": {
                        "type": ["string", "null"],
                        "maxLength": 500,
                    },
                    "reason_code": {
                        "type": "string",
                        "enum": list(REFLECTION_REASON_CODES),
                    },
                },
                "required": [
                    "action",
                    "scope",
                    "target_id",
                    "lesson",
                    "reason_code",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["changes"],
    "additionalProperties": False,
}


REFLECTION_INSTRUCTIONS = """
You are the conservative Experience Manager for another AI agent.

The main agent has ALREADY completed the user's task and finalized its answer.
You cannot revise that answer. Your only job is to decide whether the supplied
observable interaction contains a small, durable lesson worth preserving for
future tasks.

IMPORTANT DEFAULT:
Most interactions should produce {"changes": []}. Successful task completion by
itself is NOT a reason to store experience.

Store or reinforce a lesson only when it is supported by a strong signal such
as:
- an explicit user correction;
- an explicit durable user preference;
- an objectively observed failure followed by a useful successful recovery;
- validation revealing a meaningful mismatch followed by a correction;
- independent evidence reinforcing an already-stored lesson.

Reject ordinary success, temporary website behavior, task-specific facts,
one-off filenames/paths/URLs/values, incidental library choices, unsupported
model speculation, and lessons that merely restate the active skill without an
observed operational refinement.

A stored lesson must be:
- generalizable beyond the exact current task;
- actionable;
- stable rather than transient;
- atomic: one main lesson;
- concise;
- supported by observable evidence;
- novel or a meaningful improvement/reinforcement of existing memory;
- correctly scoped;
- free of secrets and sensitive personal information.

SCOPES:
- global: broadly useful regardless of skill.
- skill: requires the active domain/skill to make sense.
- user_preference: a durable preference of this user, not a universal rule.

The active skill is classification context, not the automatic destination. A
lesson observed during a skill task can still be global. If no active skill is
provided, never propose scope="skill".

PREFERENCES:
Do not infer a user preference merely because the user accepted an answer or
because the agent chose a style once. Prefer explicit statements/corrections
that clearly express durable desired behavior.

DEDUPLICATION:
Compare candidates against the supplied existing memory.
- reinforce: substantially the same lesson already exists.
- update: an existing lesson should be materially improved/generalized.
- add: genuinely new lesson.
- discard: optional diagnostic outcome; prefer no change at all when there is
  no clear candidate.

For reinforce/update, target_id must identify an existing entry in the SAME
scope. For add, target_id must be null. For reinforce, lesson should be null.
For update/add, provide the final generalized lesson text.

Do not store raw conversation details. Do not store API keys, tokens,
passwords, cookies, private keys, credential-bearing URLs, personal secrets, or
sensitive inferred attributes.

Return only data conforming to the supplied schema. Do not provide prose.
""".strip()


CONSOLIDATION_INSTRUCTIONS = """
You are conservatively consolidating ONE small scope of an AI agent's persistent
experience memory because it is already at capacity and a new candidate lesson
may deserve admission.

Your task is NOT to maximize compression. Preserve all existing information
unless two or more lessons can be safely represented by one concise,
generalized, actionable lesson.

You receive:
- existing entries in one scope;
- one new candidate lesson;
- a maximum number of final entries.

You may either:
1. reject the candidate and preserve the existing information; or
2. accept the candidate only if the final set fits the limit without losing the
   substantive information represented by existing lessons.

Every existing source id must appear exactly once across the returned items.
If accept_candidate=true, include_candidate must be true in exactly one item.
If accept_candidate=false, include_candidate must be false in every item.

Merging is allowed only when the lessons are genuinely compatible. Do not merge
unrelated principles merely to create space. Do not invent new lessons. Do not
store secrets or task-specific/transient details.

Each final lesson must remain concise, atomic, actionable, and generalizable.
Return only structured data matching the supplied schema.
""".strip()


# =============================================================================
# Reports
# =============================================================================

@dataclass(frozen=True)
class ReflectionReport:
    """Non-fatal diagnostic result returned by reflect()."""

    attempted: bool
    proposed_changes: int = 0
    applied_changes: int = 0
    memory_changed: bool = False
    messages: tuple[str, ...] = ()


# =============================================================================
# Experience Manager
# =============================================================================

class ExperienceManager:
    """
    Small persistent experience-memory manager shared by the General Agents.

    The class is deliberately usable independently of either agent engine.

    Parameters
    ----------
    memory_path:
        JSON persistence path. Defaults to:
        GeneralAgent/agent_memory/experiences.json

    model:
        Model used only for post-turn reflection and rare capacity
        consolidation. The parent agent may pass its own model name.

    client:
        Optional OpenAI client. If omitted, creation is lazy and occurs only
        when reflection/consolidation actually needs an API call. This keeps
        load/render operations testable without API access.

    show_activity:
        Print concise experience diagnostics to stderr. Only relevant when the
        parent agent has experience mode enabled.
    """

    def __init__(
        self,
        memory_path: str | Path = DEFAULT_MEMORY_PATH,
        *,
        model: str = DEFAULT_REFLECTION_MODEL,
        reasoning_effort: str = DEFAULT_REFLECTION_REASONING_EFFORT,
        client: Any | None = None,
        show_activity: bool = True,
        max_global_experiences: int = MAX_GLOBAL_EXPERIENCES,
        max_skill_experiences: int = MAX_SKILL_EXPERIENCES,
        max_user_preferences: int = MAX_USER_PREFERENCES,
        max_applicable_items: int = MAX_APPLICABLE_ITEMS,
        max_lesson_words: int = MAX_LESSON_WORDS,
        max_candidates_per_turn: int = MAX_CANDIDATES_PER_TURN,
    ) -> None:
        self.memory_path = Path(memory_path).expanduser()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client = client
        self.show_activity = show_activity

        self.max_global_experiences = max_global_experiences
        self.max_skill_experiences = max_skill_experiences
        self.max_user_preferences = max_user_preferences
        self.max_applicable_items = max_applicable_items
        self.max_lesson_words = max_lesson_words
        self.max_candidates_per_turn = max_candidates_per_turn

        self._memory: dict[str, Any] | None = None
        self._load_ok: bool = False

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def canonical_skill_key(skill: str | Path | None) -> str | None:
        """
        Convert a skill path/name to a stable domain-neutral namespace.

        Examples
        --------
        cartography_geovisualization_skill.md
            -> cartography_geovisualization

        Skills/geospatial_data_discovery_acquisition_skill.md
            -> geospatial_data_discovery_acquisition
        """
        if skill is None:
            return None

        raw = str(skill).strip()
        if not raw:
            return None

        name = Path(raw).name

        if name.lower().endswith(".md"):
            name = name[:-3]

        if name.lower().endswith("_skill"):
            name = name[:-6]

        # Stable, readable, conservative namespace.
        key = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        key = re.sub(r"_+", "_", key)

        return key or None

    def load(self) -> bool:
        """
        Load and validate persistent memory.

        Missing memory is initialized as an empty v1 store. Malformed memory is
        NEVER overwritten automatically; the manager becomes unavailable for
        that turn so the parent agent can continue without experience.
        """
        try:
            parent = self.memory_path.parent
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._memory = None
            self._load_ok = False
            self._log(
                f"Could not create/access memory directory; experience disabled "
                f"for this turn: {type(exc).__name__}: {exc}"
            )
            return False

        if not self.memory_path.exists():
            empty = self._empty_memory()

            # Initialize the persistent file when possible. Failure to write does
            # not make the main agent fail; the in-memory empty store remains
            # usable for prompt rendering, though future persistence may fail.
            try:
                self._atomic_save(empty)
            except Exception as exc:
                self._log(
                    "Could not initialize experiences.json; continuing with an "
                    f"empty in-memory experience store: {type(exc).__name__}: {exc}"
                )

            self._memory = empty
            self._load_ok = True
            return True

        try:
            text = self.memory_path.read_text(encoding="utf-8")
            loaded = json.loads(text)
            validated = self._validate_memory(loaded)
        except Exception as exc:
            self._memory = None
            self._load_ok = False
            self._log(
                "Experience memory is malformed/unreadable. It was NOT "
                f"overwritten; continuing without experience for this turn: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

        self._memory = validated
        self._load_ok = True
        return True

    def applicable_lessons(
        self,
        active_skill_key: str | Path | None = None,
    ) -> dict[str, list[str]]:
        """
        Return applicable lesson TEXT only.

        IDs and evidence counts are intentionally hidden from the main agent.
        """
        if not self.load():
            return {
                "global": [],
                "skill": [],
                "user_preferences": [],
            }

        assert self._memory is not None
        skill_key = self.canonical_skill_key(active_skill_key)

        global_lessons = [
            item["lesson"] for item in self._memory["global"]
        ]

        skill_lessons: list[str] = []
        if skill_key:
            skill_lessons = [
                item["lesson"]
                for item in self._memory["skills"].get(skill_key, [])
            ]

        preferences = [
            item["lesson"] for item in self._memory["user_preferences"]
        ]

        total = len(global_lessons) + len(skill_lessons) + len(preferences)

        # Normally guaranteed by memory validation/cap configuration.
        if total > self.max_applicable_items:
            self._log(
                f"Applicable experience count ({total}) exceeds configured "
                f"limit ({self.max_applicable_items}); omitting experience "
                "rather than silently truncating it."
            )
            return {
                "global": [],
                "skill": [],
                "user_preferences": [],
            }

        return {
            "global": global_lessons,
            "skill": skill_lessons,
            "user_preferences": preferences,
        }

    def render_prompt_block(
        self,
        active_skill_key: str | Path | None = None,
    ) -> str:
        """
        Render the optional experience block for ONE user turn.

        If no applicable experience exists, returns an empty string. This is
        intentional: an experience-enabled agent with an empty store should not
        receive a pointless prompt modification.
        """
        lessons = self.applicable_lessons(active_skill_key)

        if not any(lessons.values()):
            return ""

        sections = [
            "\n\nEXPERIENCE MEMORY",
            "",
            (
                "These are short lessons learned from prior interactions. Apply "
                "them when relevant, but do not follow them mechanically. "
                "Current user requirements, current evidence, general agent "
                "instructions, and the active skill take precedence when they "
                "conflict."
            ),
        ]

        if lessons["global"]:
            sections.extend(
                [
                    "",
                    "Global experiences:",
                    *[f"- {lesson}" for lesson in lessons["global"]],
                ]
            )

        if lessons["skill"]:
            sections.extend(
                [
                    "",
                    "Active-skill experiences:",
                    *[f"- {lesson}" for lesson in lessons["skill"]],
                ]
            )

        if lessons["user_preferences"]:
            sections.extend(
                [
                    "",
                    "User preferences:",
                    *[
                        f"- {lesson}"
                        for lesson in lessons["user_preferences"]
                    ],
                ]
            )

        return "\n".join(sections)

    def reflect(
        self,
        interaction_evidence: Any,
        *,
        active_skill_key: str | Path | None = None,
        skill_markdown: str = "",
    ) -> ReflectionReport:
        """
        Reflect on a COMPLETED interaction and optionally update memory.

        This method is intended to be called only AFTER the parent agent has
        finalized its answer. Any failure is contained here and returned as a
        non-fatal report.

        The parent agent should supply a compact observable evidence window
        containing the previous and current completed real user turns when
        available. Hidden reasoning should not be supplied; this manager also
        strips obvious reasoning items defensively.
        """
        messages: list[str] = []

        try:
            if not self.load():
                return ReflectionReport(
                    attempted=False,
                    messages=("memory unavailable; reflection skipped",),
                )

            assert self._memory is not None

            prepared_evidence = self.prepare_interaction_evidence(
                interaction_evidence
            )

            if not prepared_evidence.strip():
                return ReflectionReport(
                    attempted=False,
                    messages=("no usable observable evidence; reflection skipped",),
                )

            skill_key = self.canonical_skill_key(active_skill_key)
            relevant_memory = self._relevant_memory_for_reflection(skill_key)

            reflection_input = self._build_reflection_input(
                evidence=prepared_evidence,
                relevant_memory=relevant_memory,
                active_skill_key=skill_key,
                skill_markdown=skill_markdown,
            )

            client = self._get_client()

            response = client.responses.create(
                model=self.model,
                instructions=REFLECTION_INSTRUCTIONS,
                input=reflection_input,
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=MAX_REFLECTION_OUTPUT_TOKENS,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "experience_reflection",
                        "strict": True,
                        "schema": self._reflection_schema(),
                    }
                },
            )

            output_text = (getattr(response, "output_text", "") or "").strip()
            if not output_text:
                raise RuntimeError(
                    "Experience reflection returned no structured output text."
                )

            payload = json.loads(output_text)
            proposed = payload.get("changes", [])

            if not isinstance(proposed, list):
                raise ValueError("Reflection payload 'changes' is not a list.")

            changed, applied, apply_messages = self._apply_reflection_payload(
                payload,
                active_skill_key=skill_key,
            )
            messages.extend(apply_messages)

            if self.show_activity:
                if changed:
                    self._log(
                        f"Reflection proposed {len(proposed)} change(s); "
                        f"applied {applied}."
                    )
                elif proposed:
                    self._log(
                        f"Reflection proposed {len(proposed)} change(s); none "
                        "passed host-side validation."
                    )

            return ReflectionReport(
                attempted=True,
                proposed_changes=len(proposed),
                applied_changes=applied,
                memory_changed=changed,
                messages=tuple(messages),
            )

        except Exception as exc:
            self._log(
                "Reflection failed; existing experience memory was preserved: "
                f"{type(exc).__name__}: {exc}"
            )
            return ReflectionReport(
                attempted=True,
                memory_changed=False,
                messages=(
                    f"reflection failed: {type(exc).__name__}: {exc}",
                ),
            )

    def prepare_interaction_evidence(self, evidence: Any) -> str:
        """
        Convert observable interaction evidence to a compact redacted string.

        Defensive behavior:
        - strips obvious reasoning items;
        - strips encrypted reasoning content;
        - redacts credential-like dictionary fields;
        - redacts credential-like text;
        - bounds evidence size.

        The parent agent should still provide a curated previous+current turn
        window rather than its entire session.
        """
        cleaned = self._sanitize_evidence_value(evidence)

        if isinstance(cleaned, str):
            rendered = cleaned
        else:
            rendered = json.dumps(
                cleaned,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        rendered = self._redact_text(rendered)

        if len(rendered) > MAX_REFLECTION_EVIDENCE_CHARS:
            half = MAX_REFLECTION_EVIDENCE_CHARS // 2
            omitted = len(rendered) - MAX_REFLECTION_EVIDENCE_CHARS
            rendered = (
                rendered[:half]
                + f"\n\n... <{omitted} characters omitted from reflection "
                  "evidence> ...\n\n"
                + rendered[-half:]
            )

        return rendered.strip()

    def memory_snapshot(self) -> dict[str, Any] | None:
        """
        Return a deep copy for inspection/testing.

        This is not intended to be injected into the main agent prompt.
        """
        if not self.load():
            return None

        assert self._memory is not None
        return copy.deepcopy(self._memory)

    # -------------------------------------------------------------------------
    # Reflection application
    # -------------------------------------------------------------------------

    def _apply_reflection_payload(
        self,
        payload: dict[str, Any],
        *,
        active_skill_key: str | None,
    ) -> tuple[bool, int, list[str]]:
        """
        Validate and transactionally apply model-proposed memory changes.

        No in-memory state is committed until the complete candidate memory has
        passed validation AND atomic disk persistence succeeds.
        """
        if not self._load_ok or self._memory is None:
            return False, 0, ["memory unavailable"]

        changes = payload.get("changes", [])

        if not isinstance(changes, list):
            return False, 0, ["invalid changes payload"]

        if len(changes) > self.max_candidates_per_turn:
            return False, 0, ["too many proposed changes"]

        working = copy.deepcopy(self._memory)
        messages: list[str] = []
        applied = 0
        touched_targets: set[str] = set()

        for index, raw_change in enumerate(changes):
            try:
                if not isinstance(raw_change, dict):
                    raise ValueError("change must be an object")

                action = raw_change.get("action")
                scope = raw_change.get("scope")
                target_id = raw_change.get("target_id")
                lesson = raw_change.get("lesson")
                reason_code = raw_change.get("reason_code")

                if action not in {"add", "reinforce", "update", "discard"}:
                    raise ValueError(f"unsupported action: {action!r}")

                if scope not in {"global", "skill", "user_preference"}:
                    raise ValueError(f"unsupported scope: {scope!r}")

                if reason_code not in REFLECTION_REASON_CODES:
                    raise ValueError(
                        f"unsupported reason_code: {reason_code!r}"
                    )

                if scope == "skill" and not active_skill_key:
                    raise ValueError(
                        "skill-scoped change rejected because no active skill "
                        "is available"
                    )

                if action == "discard":
                    # Diagnostic only. Never persists anything.
                    messages.append(
                        f"change {index}: discarded ({reason_code})"
                    )
                    continue

                if reason_code not in POSITIVE_REASON_CODES:
                    raise ValueError(
                        f"persistent action {action!r} rejected because "
                        f"reason_code {reason_code!r} is not a positive "
                        "evidence signal"
                    )

                # A user preference must stay a preference; one user's desired
                # style/workflow should never become supposed universal
                # expertise merely because the reflection model mis-scoped it.
                if (
                    reason_code == "explicit_preference"
                    and scope != "user_preference"
                ):
                    raise ValueError(
                        "explicit_preference may only persist in "
                        "user_preference scope"
                    )

                if (
                    scope == "user_preference"
                    and reason_code
                    not in {"explicit_preference", "explicit_correction"}
                ):
                    raise ValueError(
                        "user preferences require explicit preference/correction "
                        "evidence"
                    )

                if action == "add":
                    if target_id is not None:
                        raise ValueError("add requires target_id=null")

                    normalized_lesson = self._validated_lesson(lesson)

                    # Host-side exact duplicate safeguard. Semantic duplicate
                    # detection is handled by the reflection model.
                    exact = self._find_exact_lesson(
                        working,
                        normalized_lesson,
                        scope=scope,
                        active_skill_key=active_skill_key,
                    )

                    if exact is not None:
                        if exact["id"] in touched_targets:
                            raise ValueError(
                                f"experience {exact['id']} was already modified "
                                "by another change in this reflection"
                            )

                        touched_targets.add(exact["id"])
                        exact["evidence_count"] += 1
                        applied += 1
                        messages.append(
                            f"change {index}: exact duplicate converted to "
                            f"reinforcement of {exact['id']}"
                        )
                        continue

                    # Reject exact cross-scope duplicates rather than silently
                    # creating parallel copies in different scopes.
                    cross_scope = self._find_exact_lesson_anywhere(
                        working,
                        normalized_lesson,
                    )
                    if cross_scope is not None:
                        raise ValueError(
                            "add rejected because the same lesson already "
                            f"exists as {cross_scope['id']} in another scope"
                        )

                    entries = self._scope_entries(
                        working,
                        scope=scope,
                        active_skill_key=active_skill_key,
                        create=True,
                    )
                    assert entries is not None

                    limit = self._scope_limit(scope)

                    if len(entries) >= limit:
                        consolidated = self._consolidate_full_scope(
                            memory=working,
                            scope=scope,
                            active_skill_key=active_skill_key,
                            candidate_lesson=normalized_lesson,
                        )

                        if consolidated is None:
                            messages.append(
                                f"change {index}: candidate rejected because "
                                "scope is full and safe consolidation was not "
                                "available"
                            )
                            continue

                        self._replace_scope_entries(
                            working,
                            scope=scope,
                            active_skill_key=active_skill_key,
                            entries=consolidated,
                        )
                        touched_targets.update(
                            item["id"] for item in consolidated
                        )
                        applied += 1
                        messages.append(
                            f"change {index}: candidate admitted through safe "
                            "scope consolidation"
                        )
                        continue

                    new_id = self._next_id(
                        working,
                        self._scope_prefix(scope),
                    )
                    entries.append(
                        {
                            "id": new_id,
                            "lesson": normalized_lesson,
                            "evidence_count": 1,
                        }
                    )
                    touched_targets.add(new_id)
                    applied += 1
                    messages.append(
                        f"change {index}: added {new_id}"
                    )
                    continue

                # reinforce/update need an existing target in the stated scope.
                if not isinstance(target_id, str) or not target_id.strip():
                    raise ValueError(
                        f"{action} requires a non-empty target_id"
                    )

                if target_id in touched_targets:
                    raise ValueError(
                        f"target {target_id} was already modified by another "
                        "change in this reflection"
                    )

                target = self._find_target(
                    working,
                    target_id=target_id,
                    scope=scope,
                    active_skill_key=active_skill_key,
                )
                if target is None:
                    raise ValueError(
                        f"target {target_id!r} does not exist in scope {scope!r}"
                    )

                touched_targets.add(target_id)

                if action == "reinforce":
                    if lesson is not None:
                        raise ValueError("reinforce requires lesson=null")

                    target["evidence_count"] += 1
                    applied += 1
                    messages.append(
                        f"change {index}: reinforced {target_id}"
                    )
                    continue

                if action == "update":
                    normalized_lesson = self._validated_lesson(lesson)

                    other_exact = self._find_exact_lesson_anywhere(
                        working,
                        normalized_lesson,
                        exclude_id=target_id,
                    )
                    if other_exact is not None:
                        raise ValueError(
                            "update rejected because the proposed lesson would "
                            f"duplicate existing entry {other_exact['id']}"
                        )

                    target["lesson"] = normalized_lesson
                    target["evidence_count"] += 1
                    applied += 1
                    messages.append(
                        f"change {index}: updated {target_id}"
                    )
                    continue

            except Exception as exc:
                messages.append(
                    f"change {index}: rejected: {type(exc).__name__}: {exc}"
                )

        if applied == 0:
            return False, 0, messages

        # Full-host validation before disk mutation.
        validated_working = self._validate_memory(working)

        # Transaction boundary: save before replacing self._memory.
        try:
            self._atomic_save(validated_working)
        except Exception as exc:
            messages.append(                f"atomic save failed; all proposed changes rolled back: "
                f"{type(exc).__name__}: {exc}"
            )
            return False, 0, messages

        self._memory = validated_working
        self._load_ok = True
        return True, applied, messages

    # -------------------------------------------------------------------------
    # Rare capacity consolidation
    # -------------------------------------------------------------------------

    def _consolidate_full_scope(
        self,
        *,
        memory: dict[str, Any],
        scope: Scope,
        active_skill_key: str | None,
        candidate_lesson: str,
    ) -> list[dict[str, Any]] | None:
        """
        Try to admit one valid candidate when a scope is already full.

        Safety properties:
        - every existing entry must be represented exactly once in the model's
          returned source_ids;
        - existing lessons cannot simply disappear;
        - the candidate may be rejected;
        - unrelated lessons must not be forcibly merged;
        - host validation re-checks every returned lesson.
        """
        entries = self._scope_entries(
            memory,
            scope=scope,
            active_skill_key=active_skill_key,
            create=False,
        )

        if entries is None:
            return None

        limit = self._scope_limit(scope)

        if len(entries) < limit:
            raise RuntimeError(
                "Capacity consolidation called for a scope that is not full."
            )

        schema = self._consolidation_schema(limit)

        payload = {
            "scope": scope,
            "active_skill_key": (
                active_skill_key if scope == "skill" else None
            ),
            "maximum_final_entries": limit,
            "existing_entries": entries,
            "candidate_lesson": candidate_lesson,
        }

        try:
            client = self._get_client()
            response = client.responses.create(
                model=self.model,
                instructions=CONSOLIDATION_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False, indent=2),
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=MAX_CONSOLIDATION_OUTPUT_TOKENS,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "experience_scope_consolidation",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )

            output_text = (
                getattr(response, "output_text", "") or ""
            ).strip()
            if not output_text:
                raise RuntimeError("Consolidation returned no output.")

            result = json.loads(output_text)
            return self._validate_consolidation_result(
                result=result,
                memory=memory,
                existing_entries=entries,
                candidate_lesson=candidate_lesson,
                scope=scope,
                limit=limit,
            )

        except Exception as exc:
            self._log(
                "Capacity consolidation failed; candidate will not be admitted: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def _validate_consolidation_result(
        self,
        *,
        result: dict[str, Any],
        memory: dict[str, Any],
        existing_entries: list[dict[str, Any]],
        candidate_lesson: str,
        scope: Scope,
        limit: int,
    ) -> list[dict[str, Any]] | None:
        if not isinstance(result, dict):
            raise ValueError("Consolidation result must be an object.")

        accept = result.get("accept_candidate")
        items = result.get("items")

        if not isinstance(accept, bool):
            raise ValueError("accept_candidate must be boolean.")

        if not isinstance(items, list):
            raise ValueError("items must be a list.")

        if not (1 <= len(items) <= limit):
            raise ValueError(
                f"consolidated item count must be between 1 and {limit}"
            )

        existing_by_id = {
            item["id"]: item for item in existing_entries
        }
        expected_ids = set(existing_by_id)

        seen_source_ids: set[str] = set()
        candidate_mentions = 0
        final_entries: list[dict[str, Any]] = []

        # Preserve original scope order when choosing the stable ID for a merge.
        original_order = {
            item["id"]: index
            for index, item in enumerate(existing_entries)
        }

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Consolidation item must be an object.")

            source_ids = item.get("source_ids")
            include_candidate = item.get("include_candidate")
            lesson = item.get("lesson")

            if not isinstance(source_ids, list):
                raise ValueError("source_ids must be a list.")

            if len(source_ids) != len(set(source_ids)):
                raise ValueError(
                    "source_ids may not repeat within one consolidation item."
                )

            if not isinstance(include_candidate, bool):
                raise ValueError("include_candidate must be boolean.")

            if not source_ids and not include_candidate:
                raise ValueError(
                    "Every consolidation item must represent at least one "
                    "existing source or the candidate."
                )

            for source_id in source_ids:
                if source_id not in expected_ids:
                    raise ValueError(
                        f"Unknown consolidation source id: {source_id}"
                    )
                if source_id in seen_source_ids:
                    raise ValueError(
                        f"Source id represented more than once: {source_id}"
                    )
                seen_source_ids.add(source_id)

            if include_candidate:
                candidate_mentions += 1

            normalized_lesson = self._validated_lesson(lesson)

            evidence_count = sum(
                existing_by_id[source_id]["evidence_count"]
                for source_id in source_ids
            )
            if include_candidate:
                evidence_count += 1

            if source_ids:
                stable_id = min(
                    source_ids,
                    key=lambda value: original_order[value],
                )
            else:
                stable_id = self._next_id(
                    memory,
                    self._scope_prefix(scope),
                    reserved_ids={
                        entry["id"] for entry in final_entries
                    },
                )

            final_entries.append(
                {
                    "id": stable_id,
                    "lesson": normalized_lesson,
                    "evidence_count": max(1, evidence_count),
                }
            )

        if seen_source_ids != expected_ids:
            missing = sorted(expected_ids - seen_source_ids)
            raise ValueError(
                "Consolidation attempted to drop existing experience(s): "
                + ", ".join(missing)
            )

        if accept:
            if candidate_mentions != 1:
                raise ValueError(
                    "Accepted candidate must appear exactly once."
                )
        else:
            if candidate_mentions != 0:
                raise ValueError(
                    "Rejected candidate must not appear in final items."
                )

            # Candidate rejected: preserving existing memory is safer than
            # applying unrelated consolidation edits merely because a candidate
            # was considered.
            return None

        # Exact duplicates in the final set are not allowed.
        normalized_seen: set[str] = set()
        for entry in final_entries:
            key = self._lesson_key(entry["lesson"])
            if key in normalized_seen:
                raise ValueError(
                    "Consolidation produced duplicate final lessons."
                )
            normalized_seen.add(key)

        if len(final_entries) > limit:
            raise ValueError("Consolidated scope still exceeds capacity.")

        return final_entries

    # -------------------------------------------------------------------------
    # Reflection input / memory representation
    # -------------------------------------------------------------------------

    def _build_reflection_input(
        self,
        *,
        evidence: str,
        relevant_memory: dict[str, Any],
        active_skill_key: str | None,
        skill_markdown: str,
    ) -> str:
        safe_skill = self._redact_text(skill_markdown.strip())

        parts = [
            "<active_skill_key>",
            active_skill_key or "NONE",
            "</active_skill_key>",
            "",
            "<existing_applicable_memory>",
            json.dumps(
                relevant_memory,
                ensure_ascii=False,
                indent=2,
            ),
            "</existing_applicable_memory>",
            "",
        ]

        if safe_skill:
            parts.extend(
                [
                    "<active_skill_markdown>",
                    safe_skill,
                    "</active_skill_markdown>",
                    "",
                ]
            )
        else:
            parts.extend(
                [
                    "<active_skill_markdown>",
                    "NONE",
                    "</active_skill_markdown>",
                    "",
                ]
            )

        parts.extend(
            [
                "<observable_interaction_evidence>",
                evidence,
                "</observable_interaction_evidence>",
            ]
        )

        return "\n".join(parts)

    def _relevant_memory_for_reflection(
        self,
        active_skill_key: str | None,
    ) -> dict[str, Any]:
        assert self._memory is not None

        result = {
            "global": copy.deepcopy(self._memory["global"]),
            "active_skill_key": active_skill_key,
            "active_skill": [],
            "user_preferences": copy.deepcopy(
                self._memory["user_preferences"]
            ),
        }

        if active_skill_key:
            result["active_skill"] = copy.deepcopy(
                self._memory["skills"].get(active_skill_key, [])
            )

        return result

    # -------------------------------------------------------------------------
    # Memory validation
    # -------------------------------------------------------------------------

    def _empty_memory(self) -> dict[str, Any]:
        return {
            "version": MEMORY_VERSION,
            "global": [],
            "skills": {},
            "user_preferences": [],
        }

    def _validate_memory(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("experience memory root must be an object")

        allowed_root = {
            "version",
            "global",
            "skills",
            "user_preferences",
        }
        unexpected = set(value) - allowed_root
        if unexpected:
            raise ValueError(
                f"unexpected root field(s): {sorted(unexpected)}"
            )

        if value.get("version") != MEMORY_VERSION:
            raise ValueError(
                f"unsupported memory version: {value.get('version')!r}"
            )

        global_entries = self._validate_entry_list(
            value.get("global"),
            expected_prefix="g",
            max_items=self.max_global_experiences,
            label="global",
        )

        raw_skills = value.get("skills")
        if not isinstance(raw_skills, dict):
            raise ValueError("'skills' must be an object")

        skills: dict[str, list[dict[str, Any]]] = {}

        for raw_key, raw_entries in raw_skills.items():
            if not isinstance(raw_key, str):
                raise ValueError("skill keys must be strings")

            key = self.canonical_skill_key(raw_key)
            if not key or key != raw_key:
                raise ValueError(
                    f"skill key is not canonical: {raw_key!r}"
                )

            skills[key] = self._validate_entry_list(
                raw_entries,
                expected_prefix="s",
                max_items=self.max_skill_experiences,
                label=f"skill:{key}",
            )

        preferences = self._validate_entry_list(
            value.get("user_preferences"),
            expected_prefix="p",
            max_items=self.max_user_preferences,
            label="user_preferences",
        )

        validated = {
            "version": MEMORY_VERSION,
            "global": global_entries,
            "skills": skills,
            "user_preferences": preferences,
        }

        # IDs are globally unique so target lookup remains unambiguous.
        all_ids: list[str] = []
        all_ids.extend(item["id"] for item in global_entries)
        for entries in skills.values():
            all_ids.extend(item["id"] for item in entries)
        all_ids.extend(item["id"] for item in preferences)

        if len(all_ids) != len(set(all_ids)):
            raise ValueError("experience IDs must be globally unique")

        # Exact duplicate lessons anywhere in memory are rejected. Semantic
        # duplicates are handled by reflection/consolidation.
        lesson_keys: list[str] = []
        lesson_keys.extend(
            self._lesson_key(item["lesson"])
            for item in global_entries
        )
        for entries in skills.values():
            lesson_keys.extend(
                self._lesson_key(item["lesson"])
                for item in entries
            )
        lesson_keys.extend(
            self._lesson_key(item["lesson"])
            for item in preferences
        )

        if len(lesson_keys) != len(set(lesson_keys)):
            raise ValueError(
                "exact duplicate lessons are not allowed across memory scopes"
            )

        # Applicable-memory safety: each possible active skill must fit.
        for key, entries in skills.items():
            applicable_count = (
                len(global_entries)
                + len(entries)
                + len(preferences)
            )
            if applicable_count > self.max_applicable_items:
                raise ValueError(
                    f"applicable memory for skill {key!r} exceeds configured "
                    f"limit {self.max_applicable_items}"
                )

        no_skill_count = len(global_entries) + len(preferences)
        if no_skill_count > self.max_applicable_items:
            raise ValueError(
                "applicable global+preference memory exceeds configured limit"
            )

        return validated

    def _validate_entry_list(
        self,
        value: Any,
        *,
        expected_prefix: str,
        max_items: int,
        label: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a list")

        if len(value) > max_items:
            raise ValueError(
                f"{label} has {len(value)} items; maximum is {max_items}"
            )

        result: list[dict[str, Any]] = []

        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{label}[{index}] must be an object"
                )

            if set(item) != {"id", "lesson", "evidence_count"}:
                raise ValueError(
                    f"{label}[{index}] must contain exactly id, lesson, "
                    "evidence_count"
                )

            item_id = item["id"]
            if not isinstance(item_id, str):
                raise ValueError(f"{label}[{index}].id must be a string")

            match = re.fullmatch(
                rf"{re.escape(expected_prefix)}([0-9]+)",
                item_id,
            )
            if match is None or int(match.group(1)) < 1:
                raise ValueError(
                    f"{label}[{index}].id has invalid format: {item_id!r}"
                )

            lesson = self._validated_lesson(item["lesson"])

            evidence_count = item["evidence_count"]
            if (
                not isinstance(evidence_count, int)
                or isinstance(evidence_count, bool)
                or evidence_count < 1
            ):
                raise ValueError(
                    f"{label}[{index}].evidence_count must be integer >= 1"
                )

            result.append(
                {
                    "id": item_id,
                    "lesson": lesson,
                    "evidence_count": evidence_count,
                }
            )

        return result

    def _validated_lesson(self, lesson: Any) -> str:
        if not isinstance(lesson, str):
            raise ValueError("lesson must be a string")

        normalized = " ".join(lesson.split()).strip()

        if not normalized:
            raise ValueError("lesson may not be empty")

        if self._word_count(normalized) > self.max_lesson_words:
            raise ValueError(
                f"lesson exceeds {self.max_lesson_words}-word limit"
            )

        if len(normalized) > 500:
            raise ValueError("lesson exceeds 500-character safety limit")

        if self._contains_secret_like_text(normalized):
            raise ValueError(
                "lesson contains credential/secret-like content"
            )

        # Literal URLs are nearly always transient/source-specific rather than
        # durable procedural experience. Reject them at persistence time.
        if re.search(r"https?://", normalized, flags=re.IGNORECASE):
            raise ValueError(
                "lesson contains a literal URL; generalize the lesson instead"
            )

        return normalized

    # -------------------------------------------------------------------------
    # Scope/target helpers
    # -------------------------------------------------------------------------

    def _scope_entries(
        self,
        memory: dict[str, Any],
        *,
        scope: Scope,
        active_skill_key: str | None,
        create: bool,
    ) -> list[dict[str, Any]] | None:
        if scope == "global":
            return memory["global"]

        if scope == "user_preference":
            return memory["user_preferences"]

        if scope == "skill":
            if not active_skill_key:
                return None

            if create:
                return memory["skills"].setdefault(
                    active_skill_key,
                    [],
                )

            return memory["skills"].get(active_skill_key, [])

        raise ValueError(f"unsupported scope: {scope}")

    def _replace_scope_entries(
        self,
        memory: dict[str, Any],
        *,
        scope: Scope,
        active_skill_key: str | None,
        entries: list[dict[str, Any]],
    ) -> None:
        if scope == "global":
            memory["global"] = entries
            return

        if scope == "user_preference":
            memory["user_preferences"] = entries
            return

        if scope == "skill":
            if not active_skill_key:
                raise ValueError("active skill required")
            memory["skills"][active_skill_key] = entries
            return

        raise ValueError(f"unsupported scope: {scope}")

    def _scope_limit(self, scope: Scope) -> int:
        if scope == "global":
            return self.max_global_experiences
        if scope == "skill":
            return self.max_skill_experiences
        if scope == "user_preference":
            return self.max_user_preferences
        raise ValueError(f"unsupported scope: {scope}")

    @staticmethod
    def _scope_prefix(scope: Scope) -> str:
        if scope == "global":
            return "g"
        if scope == "skill":
            return "s"
        if scope == "user_preference":
            return "p"
        raise ValueError(f"unsupported scope: {scope}")

    def _find_target(
        self,
        memory: dict[str, Any],
        *,
        target_id: str,
        scope: Scope,
        active_skill_key: str | None,
    ) -> dict[str, Any] | None:
        entries = self._scope_entries(
            memory,
            scope=scope,
            active_skill_key=active_skill_key,
            create=False,
        )

        if entries is None:
            return None

        for item in entries:
            if item["id"] == target_id:
                return item

        return None

    def _find_exact_lesson(
        self,
        memory: dict[str, Any],
        lesson: str,
        *,
        scope: Scope,
        active_skill_key: str | None,
    ) -> dict[str, Any] | None:
        entries = self._scope_entries(
            memory,
            scope=scope,
            active_skill_key=active_skill_key,
            create=False,
        )

        if entries is None:
            return None

        key = self._lesson_key(lesson)

        for item in entries:
            if self._lesson_key(item["lesson"]) == key:
                return item

        return None

    def _find_exact_lesson_anywhere(
        self,
        memory: dict[str, Any],
        lesson: str,
        *,
        exclude_id: str | None = None,
    ) -> dict[str, Any] | None:
        key = self._lesson_key(lesson)

        for item in self._iter_all_entries(memory):
            if exclude_id is not None and item["id"] == exclude_id:
                continue
            if self._lesson_key(item["lesson"]) == key:
                return item

        return None

    @staticmethod
    def _lesson_key(lesson: str) -> str:
        return " ".join(lesson.casefold().split())

    @staticmethod
    def _iter_all_entries(
        memory: dict[str, Any],
    ) -> Iterable[dict[str, Any]]:
        yield from memory["global"]

        for entries in memory["skills"].values():
            yield from entries

        yield from memory["user_preferences"]

    def _next_id(
        self,
        memory: dict[str, Any],
        prefix: str,
        *,
        reserved_ids: set[str] | None = None,
    ) -> str:
        reserved = reserved_ids or set()

        max_number = 0

        for item in self._iter_all_entries(memory):
            item_id = item["id"]
            if item_id.startswith(prefix):
                suffix = item_id[len(prefix):]
                if suffix.isdigit():
                    max_number = max(max_number, int(suffix))

        for item_id in reserved:
            if item_id.startswith(prefix):
                suffix = item_id[len(prefix):]
                if suffix.isdigit():
                    max_number = max(max_number, int(suffix))

        candidate = max_number + 1
        return f"{prefix}{candidate:03d}"

    # -------------------------------------------------------------------------
    # Evidence sanitization / secret protection
    # -------------------------------------------------------------------------

    def _sanitize_evidence_value(self, value: Any) -> Any:
        # OpenAI SDK/Pydantic-like objects.
        if not isinstance(
            value,
            (dict, list, tuple, str, int, float, bool, type(None)),
        ):
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                try:
                    value = model_dump(mode="json", exclude_none=True)
                except TypeError:
                    value = model_dump(exclude_none=True)
                except Exception:
                    value = str(value)
            else:
                value = str(value)

        if isinstance(value, dict):
            item_type = str(value.get("type", "")).casefold()

            # Defensive removal of private/internal reasoning representations.
            if "reasoning" in item_type:
                return {
                    "type": value.get("type", "reasoning"),
                    "content": "<REASONING_REMOVED>",
                }

            clean: dict[str, Any] = {}

            for key, child in value.items():
                key_folded = str(key).casefold()

                if key_folded in {
                    "encrypted_content",
                    "reasoning",
                    "chain_of_thought",
                    "chain-of-thought",
                }:
                    continue

                if self._credential_field_name(key_folded):
                    clean[key] = "<REDACTED>"
                    continue

                clean[key] = self._sanitize_evidence_value(child)

            return clean

        if isinstance(value, (list, tuple)):
            return [
                self._sanitize_evidence_value(child)
                for child in value
            ]

        if isinstance(value, str):
            return self._redact_text(value)

        return value

    @staticmethod
    def _credential_field_name(key_folded: str) -> bool:
        compact = re.sub(r"[^a-z0-9]", "", key_folded)

        explicit = {
            "apikey",
            "accesstoken",
            "refreshtoken",
            "authtoken",
            "authorization",
            "password",
            "passwd",
            "secret",
            "clientsecret",
            "privatekey",
            "cookie",
            "sessioncookie",
        }

        return compact in explicit

    def _redact_text(self, text: str) -> str:
        redacted = str(text)

        patterns = [
            # Quoted assignments:
            # api_key = "..."
            (
                re.compile(
                    r"""(?ix)
                    \b(api[_-]?key|access[_-]?token|refresh[_-]?token|
                    auth[_-]?token|password|passwd|client[_-]?secret|
                    secret|private[_-]?key)
                    \s*[:=]\s*
                    (["'])
                    .*?
                    \2
                    """
                ),
                r"\1=<REDACTED>",
            ),
            # Bearer authorization.
            (
                re.compile(
                    r"(?i)\b(authorization\s*[:=]?\s*bearer\s+)"
                    r"[A-Za-z0-9._~+/=-]+"
                ),
                r"\1<REDACTED>",
            ),
            (
                re.compile(
                    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
                ),
                "Bearer <REDACTED>",
            ),
            # OpenAI-style secret key shapes and JWT-like strings.
            (
                re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
                "<REDACTED_KEY>",
            ),
            (
                re.compile(
                    r"\beyJ[A-Za-z0-9_-]{8,}\."
                    r"[A-Za-z0-9_-]{8,}\."
                    r"[A-Za-z0-9_-]{8,}\b"
                ),
                "<REDACTED_TOKEN>",
            ),
            # Credential-bearing URL query parameters.
            (
                re.compile(
                    r"(?i)([?&](?:api[_-]?key|key|token|access_token|"
                    r"signature|sig|secret)=)[^&#\s]+"
                ),
                r"\1<REDACTED>",
            ),
            # PEM private keys.
            (
                re.compile(
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
                    r"-----END [A-Z ]*PRIVATE KEY-----",
                    flags=re.DOTALL,
                ),
                "<REDACTED_PRIVATE_KEY>",
            ),
        ]

        for pattern, replacement in patterns:
            redacted = pattern.sub(replacement, redacted)

        return redacted

    def _contains_secret_like_text(self, text: str) -> bool:
        if "-----BEGIN" in text.upper() and "PRIVATE KEY" in text.upper():
            return True

        patterns = [
            r"\bsk-[A-Za-z0-9_-]{12,}\b",
            r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}",
            r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|"
            r"password|passwd|client[_-]?secret|private[_-]?key)"
            r"\s*[:=]\s*\S+",
            r"(?i)[?&](api[_-]?key|key|token|access_token|signature|sig|secret)=",
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _atomic_save(self, memory: dict[str, Any]) -> None:
        """
        Atomically replace experiences.json from a fully validated object.

        The temporary file is created in the same directory so os.replace()
        remains atomic on normal local filesystems.
        """
        validated = self._validate_memory(memory)

        parent = self.memory_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".experiences_",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as temp_file:
                json.dump(
                    validated,
                    temp_file,
                    ensure_ascii=False,
                    indent=2,
                )
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = Path(temp_file.name)

            os.replace(temp_path, self.memory_path)
            temp_path = None

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # API / schema helpers
    # -------------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _reflection_schema(self) -> dict[str, Any]:
        # Copy because tests/callers should not mutate the module constant.
        schema = copy.deepcopy(REFLECTION_OUTPUT_SCHEMA)
        schema["properties"]["changes"]["maxItems"] = (
            self.max_candidates_per_turn
        )
        return schema

    @staticmethod
    def _consolidation_schema(limit: int) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "accept_candidate": {
                    "type": "boolean",
                },
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": limit,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                            "include_candidate": {
                                "type": "boolean",