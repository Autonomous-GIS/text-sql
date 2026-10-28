from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI

from experience_manager import ExperienceManager


# =============================================================================
# Configuration
# =============================================================================

MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "medium"

MAX_AGENT_STEPS = 20
PYTHON_TIMEOUT_SECONDS = 60
MAX_TOOL_OUTPUT_CHARS = 20_000

SHOW_TOOL_ACTIVITY = True

SCRIPT_PATH = Path(__file__).resolve()
GENERAL_AGENT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = GENERAL_AGENT_DIR.parent
SKILLS_DIR = PROJECT_ROOT / "Skills"
WORKSPACE = PROJECT_ROOT / "Workspace"

# Use only the project-level Workspace directory.
WORKSPACE.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Skill loading
# =============================================================================

def _available_skills() -> list[Path]:
    """Return Markdown skill files available in the project Skills directory."""
    if not SKILLS_DIR.exists():
        return []

    return sorted(
        path
        for path in SKILLS_DIR.glob("*.md")
        if path.is_file()
    )


def resolve_skill_path(skill: str) -> Path:
    """
    Resolve a skill from either a path or a short skill name.

    Examples:
        --skill Skills/vector_gis_skill.md
        --skill vector_gis_skill
        --skill vector_gis_skill.md
    """
    raw = Path(skill).expanduser()

    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        # Explicit path relative to the shell.
        candidates.append(Path.cwd() / raw)

        # Explicit path relative to the project root.
        candidates.append(PROJECT_ROOT / raw)

        # Short name inside Skills/.
        filename = raw.name
        if not Path(filename).suffix:
            filename += ".md"
        candidates.append(SKILLS_DIR / filename)

    seen: set[Path] = set()

    for candidate in candidates:
        try:
            normalized = candidate.resolve()
        except OSError:
            normalized = candidate.absolute()

        if normalized in seen:
            continue
        seen.add(normalized)

        if normalized.is_file():
            return normalized

    available = ", ".join(path.stem for path in _available_skills()) or "none"

    raise FileNotFoundError(
        f"Could not find skill '{skill}'. "
        f"Available skills in {SKILLS_DIR}: {available}"
    )


def load_skill(skill: str | None) -> tuple[str, Path | None]:
    """
    Load zero or one Markdown skill.

    skill=None means the general Python-agent baseline.
    """
    if not skill:
        return "", None

    path = resolve_skill_path(skill)
    markdown = path.read_text(encoding="utf-8").strip()

    if not markdown:
        raise ValueError(f"Skill file is empty: {path}")

    return markdown, path


# =============================================================================
# Agent instructions
# =============================================================================

GENERAL_AGENT_INSTRUCTIONS = f"""
You are a general-purpose autonomous AI agent.

You have exactly ONE external tool:

    execute_python

This tool executes Python code in a working environment and returns evidence
from the execution, including stdout, stderr, exit status, and workspace file
changes.

Your role is to remain the reasoning brain of the agent. Python is your action
and observation environment, not a replacement for your reasoning.

GENERAL BEHAVIOR

1. Understand the user's actual goal.
2. Determine what you already know and what information is still needed.
3. If you already have enough reliable information, answer directly. Do not use
   Python merely to demonstrate tool use.
4. If information needed for the task can be inspected, calculated, derived,
   tested, or otherwise discovered through Python, use execute_python instead of
   asking the user for that information.
5. Use Python iteratively when useful:
       reason -> execute -> observe -> update understanding -> execute again
6. Treat execution errors as observations. Diagnose them and try a reasonable
   correction or alternative when possible.
7. Do not assume the structure or contents of files or datasets when they can
   be inspected. Gather only the evidence needed to make a good decision.
8. When several reasonable interpretations or approaches exist, use available
   evidence and your reasoning to choose the most appropriate one when an
   action is required.
9. If the user is asking for advice and the answer genuinely depends on
   different goals or conditions, explain the relevant alternatives rather
   than forcing an arbitrary choice.
10. Make reasonable evidence-supported assumptions when necessary. Mention an
    assumption when it materially affects the result.
11. After making an important modification or producing an artifact, validate
    the result when practical.
12. If a task cannot be completed because of an actual environmental limit
    (for example unavailable data, missing software, permissions, memory,
    timeout, or another execution constraint), try reasonable alternatives
    first. If the limitation remains, explain it clearly.
13. This version has no browser or web-search tool. Do not claim to have checked
    current internet information. Do not use Python as a substitute for web
    browsing in this version.
14. Never expose private chain-of-thought. Give concise conclusions,
    assumptions, useful evidence, and results instead.

PYTHON ENVIRONMENT

- Python starts with the project Workspace directory as its current working directory.
- Use relative paths whenever practical.
- Files in the workspace can be inspected, created, modified, and reused across
  Python executions.
- Each execute_python call runs as a fresh Python process. In-memory variables
  do not persist between calls, so persist anything important in files when
  needed.
- Print values that you need to observe. If code produces no stdout, you will
  still receive its exit status, stderr, and file-change information.
- Prefer focused inspections over dumping entire large files or datasets.
""".strip()


def build_system_prompt(skill_markdown: str = "") -> str:
    """
    Build instructions for one agent instance.

    Domain specialization comes only from the Markdown skill. The Python tool
    and general reasoning loop remain unchanged.
    """
    skill = skill_markdown.strip()

    if not skill:
        return (
            GENERAL_AGENT_INSTRUCTIONS
            + "\n\nDOMAIN SKILL\n\n"
            + "No domain-specific skill is loaded. Operate as the general "
              "Python agent."
        )

    return (
        GENERAL_AGENT_INSTRUCTIONS
        + "\n\nDOMAIN SKILL\n\n"
        + "The following Markdown specializes your domain reasoning. Apply it "
          "throughout interpretation, inspection, method selection, execution, "
          "validation, and response. It supplements the general behavior and "
          "does not add external tools.\n\n"
        + "<skill>\n"
        + skill
        + "\n</skill>"
    )



# =============================================================================
# The agent's ONE tool
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "name": "execute_python",
        "description": (
            "Execute Python code in the project Workspace. Use this to inspect "
            "available files/data/environment, perform calculations or analysis, "
            "test hypotheses, manipulate data, create artifacts, validate results, "
            "and gather evidence needed to answer the user. The execution result "
            "contains stdout, stderr, exit status, timeout information, and file "
            "changes. Print anything you need to observe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source code to execute.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


# =============================================================================
# Python execution environment
# =============================================================================

def _snapshot_workspace() -> dict[str, tuple[int, int]]:
    """Return {relative_path: (size_bytes, mtime_ns)} for Workspace files."""
    snapshot: dict[str, tuple[int, int]] = {}

    for path in WORKSPACE.rglob("*"):
        if not path.is_file():
            continue

        try:
            stat = path.stat()
            relative = str(path.relative_to(WORKSPACE))
            snapshot[relative] = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            # A file may disappear while being inspected. That is not fatal.
            continue

    return snapshot


def _describe_file_changes(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Describe files created, modified, or deleted by an execution."""
    changes: list[dict[str, Any]] = []

    all_paths = sorted(set(before) | set(after))

    for relative in all_paths:
        if relative not in before:
            changes.append(
                {
                    "path": relative,
                    "change": "created",
                    "size_bytes": after[relative][0],
                }
            )
        elif relative not in after:
            changes.append(
                {
                    "path": relative,
                    "change": "deleted",
                }
            )
        elif before[relative] != after[relative]:
            changes.append(
                {
                    "path": relative,
                    "change": "modified",
                    "size_bytes": after[relative][0],
                }
            )

    return changes


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Keep tool observations useful without flooding the model context."""
    if len(text) <= limit:
        return text

    half = limit // 2
    omitted = len(text) - limit
    return (
        text[:half]
        + f"\n\n... <{omitted} characters omitted> ...\n\n"
        + text[-half:]
    )


def _sanitized_environment() -> dict[str, str]:
    """
    Remove likely credentials before exposing the process environment to
    model-generated Python.

    This reduces accidental secret exposure but is NOT a complete security
    boundary. Secrets stored in readable files are not protected by this.
    """
    blocked_fragments = (
        "API_KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "PRIVATE_KEY",
        "AUTH",
    )

    clean: dict[str, str] = {}

    for key, value in os.environ.items():
        upper = key.upper()
        if any(fragment in upper for fragment in blocked_fragments):
            continue
        clean[key] = value

    # Make the workspace explicit to generated code if it wants the path.
    clean["AGENT_WORKSPACE"] = str(WORKSPACE)

    return clean


def execute_python(code: str, *, env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Execute model-generated Python in a child process.

    The working directory is the project Workspace/. Every call starts a fresh
    Python process, while files persist across calls.
    """
    before = _snapshot_workspace()

    temp_path: Path | None = None

    try:
        # Put the temporary script outside Workspace so it does not appear
        # as a user-visible workspace artifact.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="agent_exec_",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_file.write(code)
            temp_path = Path(temp_file.name)

        try:
            completed = subprocess.run(
                [sys.executable, str(temp_path)],
                cwd=WORKSPACE,
                env={**_sanitized_environment(), **(env_overrides or {})},
                capture_output=True,
                text=True,
                timeout=PYTHON_TIMEOUT_SECONDS,
            )

            after = _snapshot_workspace()

            return {
                "ok": completed.returncode == 0,
                "timed_out": False,
                "exit_code": completed.returncode,
                "stdout": _truncate(completed.stdout),
                "stderr": _truncate(completed.stderr),
                "file_changes": _describe_file_changes(before, after),
            }

        except subprocess.TimeoutExpired as exc:
            after = _snapshot_workspace()

            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            # TimeoutExpired may contain bytes depending on Python/platform.
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            return {
                "ok": False,
                "timed_out": True,
                "exit_code": None,
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "error": (
                    f"Execution exceeded the {PYTHON_TIMEOUT_SECONDS}-second "
                    "tool timeout."
                ),
                "file_changes": _describe_file_changes(before, after),
            }

        except Exception as exc:
            after = _snapshot_workspace()

            return {
                "ok": False,
                "timed_out": False,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "error": f"{type(exc).__name__}: {exc}",
                "file_changes": _describe_file_changes(before, after),
            }

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


# =============================================================================
# Agent loop
# =============================================================================

class GeneralPythonAgent:
    """
    Multi-turn agent that lets the LLM decide whether to answer or call Python.

    The complete response/tool history is preserved across user turns until
    reset() is called.
    """

    def __init__(
        self,
        *,
        skill_markdown: str = "",
        skill_name: str | None = None,
        model: str = MODEL,
        reasoning_effort: str = REASONING_EFFORT,
        max_steps: int = MAX_AGENT_STEPS,
        show_tool_activity: bool = SHOW_TOOL_ACTIVITY,
        experience_enabled: bool = False,
        experience_manager: ExperienceManager | None = None,
    ) -> None:
        self.client = OpenAI()

        self.skill_markdown = skill_markdown.strip()
        self.skill_name = skill_name or (
            "General (no skill)" if not self.skill_markdown else "Custom skill"
        )
        self.system_prompt = build_system_prompt(self.skill_markdown)

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_steps = max_steps
        self.show_tool_activity = show_tool_activity
        self.input_items: list[Any] = []

        # Real user-turn boundaries are tracked only so optional post-turn
        # experience reflection can inspect the previous/current completed
        # turns without confusing internally inserted recovery messages for
        # genuine user messages. This does not alter model input.
        self.user_turn_starts: list[int] = []

        # Experience is strictly opt-in. When disabled, the base system prompt
        # and existing model/tool loop are used unchanged.
        self.experience_enabled = experience_enabled
        self.experience_manager = experience_manager
        self.active_skill_key = (
            ExperienceManager.canonical_skill_key(self.skill_name)
            if self.skill_markdown
            else None
        )
        if self.experience_enabled and self.experience_manager is None:
            self.experience_manager = ExperienceManager(
                model=self.model,
                client=self.client,
            )

        self.turn_system_prompt = self.system_prompt

    @classmethod
    def from_skill_file(
        cls,
        skill: str | Path | None = None,
        **kwargs: Any,
    ) -> "GeneralPythonAgent":
        """
        Programmatic constructor for general or skill-specialized instances.

        Examples:
            GeneralPythonAgent.from_skill_file()
            GeneralPythonAgent.from_skill_file("vector_gis_skill")
            GeneralPythonAgent.from_skill_file("Skills/vector_gis_skill.md")
        """
        if skill is None:
            return cls(
                skill_markdown="",
                skill_name="General (no skill)",
                **kwargs,
            )

        markdown, path = load_skill(str(skill))
        assert path is not None

        return cls(
            skill_markdown=markdown,
            skill_name=path.stem,
            **kwargs,
        )

    def reset(self) -> None:
        """Clear conversation state. Workspace and experiences are kept."""
        self.input_items = []
        self.user_turn_starts = []
        self.turn_system_prompt = self.system_prompt

    def _prepare_turn_prompt(self) -> None:
        """Freeze optional applicable experiences for the complete user turn."""
        self.turn_system_prompt = self.system_prompt

        if not self.experience_enabled or self.experience_manager is None:
            return

        block = self.experience_manager.render_prompt_block(
            self.active_skill_key
        )
        if block:
            self.turn_system_prompt = self.system_prompt + block

    def _experience_evidence_window(self) -> dict[str, Any]:
        """Return up to the last two completed REAL user-turn blocks."""
        if not self.user_turn_starts:
            return {"turns": []}

        starts = self.user_turn_starts[-2:]
        turns: list[list[Any]] = []

        for offset, start in enumerate(starts):
            end = starts[offset + 1] if offset + 1 < len(starts) else len(self.input_items)
            turns.append(self.input_items[start:end])

        return {"turns": turns}

    def _reflect_experience(self) -> None:
        """Run non-fatal post-turn reflection after the answer is finalized."""
        if not self.experience_enabled or self.experience_manager is None:
            return

        self.experience_manager.reflect(
            self._experience_evidence_window(),
            active_skill_key=self.active_skill_key,
            skill_markdown=self.skill_markdown,
        )

    def _run_model(self):
        return self.client.responses.create(
            model=self.model,
            instructions=self.turn_system_prompt,
            input=self.input_items,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            reasoning={"effort": self.reasoning_effort},
        )

    def _show_tool_call(self, code: str, result: dict[str, Any]) -> None:
        if not self.show_tool_activity:
            return

        print("\n[execute_python]")
        print("-" * 72)
        print(code.rstrip())
        print("-" * 72)

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        changes = result.get("file_changes", [])

        if stdout:
            print("[stdout]")
            print(stdout.rstrip())

        if stderr:
            print("[stderr]")
            print(stderr.rstrip())

        if result.get("timed_out"):
            print("[status] timed out")
        else:
            print(f"[status] exit_code={result.get('exit_code')}")

        if changes:
            print("[workspace changes]")
            for item in changes:
                if "size_bytes" in item:
                    print(
                        f"  - {item['change']}: {item['path']} "
                        f"({item['size_bytes']} bytes)"
                    )
                else:
                    print(f"  - {item['change']}: {item['path']}")

    def ask(self, user_message: str) -> str:
        """
        Run the reason -> execute -> observe loop until the model answers.
        """
        self._prepare_turn_prompt()
        self.user_turn_starts.append(len(self.input_items))

        self.input_items.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        for _step in range(1, self.max_steps + 1):
            response = self._run_model()

            # Preserve ALL returned items, including reasoning items and any
            # tool calls/messages, as recommended for Responses API workflows.
            self.input_items.extend(response.output)

            tool_calls = [
                item
                for item in response.output
                if item.type == "function_call"
                and item.name == "execute_python"
            ]

            if tool_calls:
                # parallel_tool_calls=False means this should normally contain
                # a single call, which keeps the loop strictly observational:
                # execute -> observe -> let the LLM reason again.
                for call in tool_calls:
                    try:
                        arguments = json.loads(call.arguments)
                        code = arguments["code"]
                        result = execute_python(code)
                    except Exception as exc:
                        result = {
                            "ok": False,
                            "timed_out": False,
                            "exit_code": None,
                            "stdout": "",
                            "stderr": "",
                            "error": (
                                "The agent host could not process the tool call: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "file_changes": [],
                        }
                        code = call.arguments

                    self._show_tool_call(code, result)

                    self.input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                        }
                    )

                # The LLM now receives the observation and decides what to do.
                continue

            final_text = response.output_text.strip()

            if final_text:
                # The answer is already fixed. Experience reflection happens
                # afterward and cannot revise this response.
                self._reflect_experience()
                return final_text

            # No tool call and no visible answer is unusual. Let the loop make
            # one more pass with an explicit observation.
            self.input_items.append(
                {
                    "role": "user",
                    "content": (
                        "No final text or Python execution was produced. "
                        "Continue the task and either use execute_python if "
                        "evidence is needed or provide the final answer."
                    ),
                }
            )

        return (
            "I reached the configured agent-step limit before completing the "
            "task. The limit can be increased with MAX_AGENT_STEPS."
        )


# =============================================================================
# Simple terminal interface
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the general Python agent with zero or one Markdown skill."
        )
    )

    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        help=(
            "Skill name or Markdown path. Examples: "
            "'vector_gis_skill', 'vector_gis_skill.md', or "
            "'Skills/vector_gis_skill.md'. "
            "Omit for the general Python-agent baseline."
        ),
    )

    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List Markdown skills found in the project Skills directory and exit.",
    )

    parser.add_argument(
        "--experience",
        action="store_true",
        help=(
            "Enable persistent experience memory. Omit this flag to preserve "
            "the normal agent behavior without experience."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_skills:
        skills = _available_skills()

        if not skills:
            print(f"No Markdown skills found in: {SKILLS_DIR}")
            return

        print(f"Skills in {SKILLS_DIR}:")
        for path in skills:
            print(f"  - {path.stem}")
        return

    try:
        skill_markdown, skill_path = load_skill(args.skill)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skill error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    skill_name = (
        skill_path.stem
        if skill_path is not None
        else "General (no skill)"
    )

    print(f"General Python Agent | model: {MODEL}")
    print("Tools: execute_python")
    print(f"Skill: {skill_name}")

    if skill_path is not None:
        print(f"Skill file: {skill_path}")

    print(f"Workspace: {WORKSPACE}")

    if args.experience:
        print("Experience: enabled")

    print("Commands: /reset clears chat history, /exit quits.")

    agent = GeneralPythonAgent(
        skill_markdown=skill_markdown,
        skill_name=skill_name,
        experience_enabled=args.experience,
    )

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_message:
            continue

        command = user_message.lower()

        if command in {"/exit", "exit", "quit"}:
            print("Exiting.")
            break

        if command == "/reset":
            agent.reset()
            print("Conversation reset. Workspace files and experiences were kept.")
            continue

        try:
            answer = agent.ask(user_message)
            print(f"\nAgent: {answer}")
        except KeyboardInterrupt:
            print("\nRequest interrupted.")
        except Exception as exc:
            print(f"\nAgent error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()