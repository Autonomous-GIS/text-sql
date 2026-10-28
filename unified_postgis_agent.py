"""One general ReAct agent with a PostGIS skill and the legacy UI event contract.

No reasoning stage is forced by the UI: the compatibility events are observations,
not subagent calls. The original agent's one-tool loop remains the decision maker.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import sqlparse
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "GeneralAgent"))
from general_python_agent import (  # noqa: E402
    GeneralPythonAgent, execute_python,
)

STAGES = ("EEA", "MRA", "QLA", "SQLA", "RA")
KIND_TO_STAGE = {"intent": "EEA", "metadata": "MRA", "plan": "QLA", "sql": "SQLA", "review": "RA"}
PROGRESS_LINE = re.compile(r"^UI_PROGRESS:\s*(\{.*\})\s*$")


def allowed_databases() -> list[str]:
    return sorted(p.stem for p in (ROOT / "metadata").glob("*.meta"))


def database_url(database: str) -> str:
    """Select one known database without trusting a user-supplied DSN."""
    if database not in allowed_databases():
        raise ValueError("Unknown database; it must have registered metadata")
    template = os.getenv("POSTGIS_DATABASE_URL_TEMPLATE", "")
    if template:
        if template.count("{database}") != 1:
            raise ValueError("POSTGIS_DATABASE_URL_TEMPLATE needs one {database} placeholder")
        return template.replace("{database}", database)
    base = os.getenv("DATABASE_URL")
    if not base:
        raise RuntimeError("Set DATABASE_URL or POSTGIS_DATABASE_URL_TEMPLATE for the read-only database role")
    return make_url(base).set(database=database).render_as_string(hide_password=False)


def _read_only_select(sql: str) -> bool:
    """Host-side syntax screening; DB role permissions are the security boundary."""
    statements = sqlparse.parse(sql or "")
    if len(statements) != 1 or statements[0].get_type() != "SELECT":
        return False
    # Prevent SELECT INTO, which may create a table in PostgreSQL.
    normalized = sqlparse.format(sql, strip_comments=True).upper()
    if re.search(r"\bINTO\b", normalized) or re.search(r"\bFOR\s+UPDATE\b", normalized):
        return False
    return True


def validate_final_sql(sql: str, db: str) -> tuple[bool, str]:
    """EXPLAIN without execution, under a read-only transaction and timeout."""
    if not _read_only_select(sql):
        return False, "Only one read-only SELECT statement is permitted"
    engine = create_engine(database_url(db), pool_pre_ping=True, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL statement_timeout = '5000ms'"))
                conn.execute(text("EXPLAIN (FORMAT JSON) " + sql.strip().rstrip(";")))
        return True, "Read-only PostgreSQL EXPLAIN succeeded (not a correctness or runtime guarantee)"
    except Exception as exc:
        return False, f"Read-only EXPLAIN failed: {type(exc).__name__}: {str(exc)[:400]}"
    finally:
        engine.dispose()


def extract_json(payload: str) -> dict[str, Any]:
    """Model is instructed to return JSON; tolerate one Markdown code fence."""
    s = (payload or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    value = json.loads(s)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class StreamingPostGISAgent(GeneralPythonAgent):
    """Expose actual tool observations incrementally; preserve the original loop."""

    def iter_events(self, message: str, *, env_overrides: dict[str, str] | None = None) -> Iterator[dict[str, Any]]:
        self._prepare_turn_prompt()
        self.user_turn_starts.append(len(self.input_items))
        self.input_items.append({"role": "user", "content": message})
        for step in range(1, self.max_steps + 1):
            response = self._run_model()
            self.input_items.extend(response.output)
            calls = [o for o in response.output if o.type == "function_call" and o.name == "execute_python"]
            if calls:
                for call in calls:
                    try:
                        code = json.loads(call.arguments)["code"]
                        yield {"type": "tool_started", "step": step, "code": code}
                        observed = execute_python(code, env_overrides=env_overrides)
                    except Exception as exc:
                        code = getattr(call, "arguments", "")
                        observed = {"ok": False, "stdout": "", "stderr": "", "error": f"{type(exc).__name__}: {exc}", "file_changes": []}
                    # All outputs go back into the ReAct context, just like the general agent.
                    self.input_items.append({"type": "function_call_output", "call_id": call.call_id,
                                             "output": json.dumps(observed, ensure_ascii=False)})
                    yield {"type": "tool_completed", "step": step, "code": code, "observation": observed}
                continue
            final = (response.output_text or "").strip()
            if final:
                # Reflection is optional and never influences the answer already generated.
                yield {"type": "answer", "text": final}
                return
            self.input_items.append({"role": "user", "content": "No visible answer or Python call was produced. Continue or report the limitation."})
        yield {"type": "answer", "text": json.dumps({"sql_query": None, "summary": "Reached the agent step limit"})}


@dataclass
class Session:
    agent: StreamingPostGISAgent
    lock: Any = field(default_factory=threading.RLock)
    plan: dict[str, str] | None = None
    state: str = "NO_PLAN"
    history: list[dict[str, str]] = field(default_factory=list)


class ProgressAdapter:
    """Only presentation; never tells the ReAct agent which steps to execute."""
    def __init__(self, question: str):
        self.question = question
        self.buffer: dict[str, Any] = {}
        self.sent: set[str] = set()

    def record(self, stage: str, result: Any) -> list[dict[str, Any]]:
        if stage not in STAGES or stage in self.sent:
            return []
        self.buffer[stage] = result
        return self._drain()

    def _drain(self) -> list[dict[str, Any]]:
        events = []
        for stage in STAGES:
            if stage in self.sent:
                continue
            if stage not in self.buffer:
                break
            self.sent.add(stage)
            events.append({"node": stage, "result": self.buffer.pop(stage)})
        return events

    def complete(self, sql: str, valid: bool, validation: str) -> list[dict[str, Any]]:
        defaults = {
            "EEA": {"intent": self.question, "note": "Interpreted within the unified agent"},
            "MRA": "No separate metadata inspection was needed or reported",
            "QLA": "No separate plan was needed or reported",
            "SQLA": sql or "No SQL produced",
            "RA": {"status": "valid" if valid else "failed", "validation": validation, "sql": sql if valid else ""},
        }
        for stage, value in defaults.items():
            if stage not in self.sent and stage not in self.buffer:
                self.buffer[stage] = value
        return self._drain()


class UnifiedPostGISService:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self.model = os.getenv("POSTGIS_AGENT_MODEL", "gpt-5.6-luna")
        self.experience_enabled = os.getenv("POSTGIS_EXPERIENCE_ENABLED", "false").lower() == "true"
        self.skill = (ROOT / "Skills" / "postgis_sql_skill.md").read_text(encoding="utf-8")

    def session(self, user_id: str | None) -> Session:
        uid = str(user_id or "anonymous")
        with self._sessions_lock:
            if uid not in self._sessions:
                agent = StreamingPostGISAgent(skill_markdown=self.skill, skill_name="postgis_sql_skill",
                                               model=self.model, show_tool_activity=False,
                                               experience_enabled=self.experience_enabled)
                self._sessions[uid] = Session(agent=agent)
            return self._sessions[uid]

    def clear(self, user_id: str):
        with self._sessions_lock:
            self._sessions.pop(user_id, None)

    def history(self, user_id: str) -> list[dict[str, str]]:
        s = self.session(user_id)
        with s.lock:
            return list(s.history)

    def chat(self, user_id: str, message: str, knowledge_graph: dict[str, Any]) -> dict[str, Any]:
        session = self.session(user_id)
        with session.lock:
            payload = {"mode": "chat", "user_message": message, "conversation_state": session.state,
                       "existing_plan": session.plan, "available_databases": allowed_databases(),
                       "knowledge_graph": knowledge_graph}
            session.history.append({"role": "user", "content": message})
            try:
                answer = None
                for event in session.agent.iter_events(json.dumps(payload, ensure_ascii=False, default=str)):
                    if event["type"] == "answer":
                        answer = event["text"]
                parsed = extract_json(answer or "")
                status = parsed.get("status")
                if status not in {"plan_ready", "clarification_needed", "execute_plan", "answered_directly"}:
                    raise ValueError("Unrecognized chat status")
                plan = parsed.get("query_plan")
                if status == "execute_plan":
                    if not session.plan:
                        raise ValueError("Cannot execute a plan that has not been confirmed")
                    plan = dict(session.plan)
                    session.state, session.plan = "NO_PLAN", None
                elif status == "plan_ready":
                    if not isinstance(plan, dict):
                        raise ValueError("Query plan must be an object")
                    if plan.get("target_database") not in allowed_databases():
                        raise ValueError("Query plan selected an unknown database")
                    if plan.get("representation_type") not in {"map", "table"} or not plan.get("clear_question"):
                        raise ValueError("Query plan is missing required fields")
                    session.plan, session.state = plan, "PLAN_READY"
                elif status == "clarification_needed":
                    plan, session.state = None, "CLARIFICATION_NEEDED"
                else:
                    plan = None  # preserve existing state if this is a follow-up about the plan
                response = {"status": status, "response_to_user": str(parsed.get("response_to_user", "")), "query_plan": plan}
            except Exception as exc:
                response = {"status": "clarification_needed", "response_to_user":
                            "I couldn't complete the query plan. Could you restate your request?",
                            "query_plan": None}
                print(f"Unified chat error: {type(exc).__name__}: {exc}")
            session.history.append({"role": "assistant", "content": response["response_to_user"]})
            if session.agent.experience_enabled and response["status"] != "clarification_needed":
                session.agent._reflect_experience()
            return response

    def stream_sql(self, user_id: str | None, question: str, database: str) -> Iterator[dict[str, Any]]:
        session = self.session(user_id)
        with session.lock:
            progress = ProgressAdapter(question)
            try:
                url = database_url(database)
                payload = {"mode": "sql", "question": question, "database": database,
                           "output_requirement": "Produce one executable read-only SQL statement",
                           "metadata_file": f"metadata/{database}.meta",
                           "database_description_available": True}
                env = {"POSTGIS_DATABASE_URL": url, "POSTGIS_METADATA_PATH": str(ROOT / "metadata" / f"{database}.meta"),
                       "PGOPTIONS": "-c statement_timeout=12000 -c default_transaction_read_only=on"}
                answer = None
                for event in session.agent.iter_events(json.dumps(payload), env_overrides=env):
                    if event["type"] == "tool_started":
                        if "EEA" not in progress.sent:
                            yield from progress.record("EEA", {"intent": question, "note": "Unified agent has begun investigating"})
                    elif event["type"] == "tool_completed":
                        obs = event["observation"]
                        code = event.get("code", "")
                        # Public summaries, not private model reasoning or raw credentials.
                        for line in str(obs.get("stdout", "")).splitlines():
                            match = PROGRESS_LINE.match(line.strip())
                            if match:
                                try:
                                    marker = json.loads(match.group(1))
                                    stage = KIND_TO_STAGE.get(marker.get("kind"))
                                    summary = str(marker.get("summary", ""))[:1200]
                                    if stage and summary:
                                        yield from progress.record(stage, summary)
                                except (ValueError, TypeError):
                                    pass
                        if "MRA" not in progress.sent and ("create_engine" in code or "psycopg" in code or "metadata" in code):
                            yield from progress.record("MRA", "Inspected database or metadata with Python")
                        if "MRA" in progress.sent and "QLA" not in progress.sent:
                            yield from progress.record("QLA", "Investigating an approach using database observations")
                    elif event["type"] == "answer":
                        answer = event["text"]
                parsed = extract_json(answer or "")
                sql = parsed.get("sql_query")
                summary = str(parsed.get("summary", ""))[:1200]
                if not isinstance(sql, str) or not sql.strip():
                    raise ValueError(summary or "The agent did not produce an SQL query")
                sql = sql.strip()
                valid, validation = validate_final_sql(sql, database)
                yield from progress.complete(sql, valid, validation)
                yield {"node": "SQL", "result": sql if valid else "failed"}
                if not valid:
                    yield {"node": "ERROR", "result": validation}
                # Preserve the original final payload keys consumed by main.html.
                import html
                yield {"node": "FINAL", "sql_query": sql if valid else "failed",
                       "formatted_sql": f"<pre>{html.escape(sql)}</pre>" if valid else None,
                       "status": "valid" if valid else "failed"}
                session.plan, session.state = None, "NO_PLAN"
                if session.agent.experience_enabled:
                    session.agent._reflect_experience()
            except Exception as exc:
                yield from progress.complete("", False, f"{type(exc).__name__}: {exc}")
                yield {"node": "SQL", "result": "failed"}
                yield {"node": "ERROR", "result": f"{type(exc).__name__}: {str(exc)[:500]}"}
                yield {"node": "FINAL", "sql_query": "failed", "formatted_sql": None, "status": "failed"}
                session.plan, session.state = None, "NO_PLAN"