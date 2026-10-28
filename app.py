"""Spatial SQL Assistant API: one flexible ReAct agent, original UI contract.

All legacy multi-agent names are progress categories only. The unified agent is
implemented in unified_postgis_agent.py. templates/ are intentionally unchanged.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
from collections import OrderedDict

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from sqlalchemy import create_engine, text

from unified_postgis_agent import (
    UnifiedPostGISService, _read_only_select, database_url,
)

app = Flask(__name__)
CORS(app)
service = UnifiedPostGISService()
LongMemory: list[dict] = []
UserFeedbacks: dict[str, list[str]] = {}


def load_knowledge_graph() -> dict:
    """Preserve the existing serialized database directory shown by the UI."""
    from pathlib import Path
    import pickle
    path = Path(__file__).resolve().parent / "KNOWLEDGE_GRAPH"
    if not path.is_file():
        app.logger.warning("KNOWLEDGE_GRAPH missing; the chat has no catalog")
        return {}
    with path.open("rb") as f:
        return pickle.load(f)  # application-controlled trusted repository asset


KNOWLEDGE_GRAPH = load_knowledge_graph()


def colorize_sql(sql: str) -> str:
    """Local highlighting avoids submitting queries to an unrelated third party."""
    try:
        from pygments import highlight
        from pygments.lexers import SqlLexer
        from pygments.formatters import HtmlFormatter
        return highlight(sql, SqlLexer(), HtmlFormatter(noclasses=True))
    except ImportError:
        return f"<pre>{html.escape(sql)}</pre>"


def _get_engine(database: str):
    return create_engine(database_url(database), pool_pre_ping=True, connect_args={"connect_timeout": 5})


def _query(sql: str, database: str, max_rows: int = 5000) -> tuple[list[dict], bool]:
    if not _read_only_select(sql):
        raise ValueError("Only a single read-only SELECT statement is allowed")
    engine = _get_engine(database)
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL statement_timeout = '12000ms'"))
                result = connection.execute(text(sql))
                rows = result.mappings().fetchmany(max_rows + 1)
                return [dict(row) for row in rows[:max_rows]], len(rows) > max_rows
    finally:
        engine.dispose()


def _execute_map(sql: str, database: str) -> dict:
    if not _read_only_select(sql):
        raise ValueError("Only a single read-only SELECT statement is allowed")
    # The agent skill requires final map queries to return geometry AS geom.
    # Transform geometry only at the final output boundary for GeoJSON maps.
    max_features = int(os.getenv("MAX_MAP_FEATURES", "5000"))
    inner = sql.strip().rstrip(";")
    wrapped = f"""
        WITH map_rows AS (SELECT * FROM ({inner}) AS user_results LIMIT :feature_limit)
        SELECT jsonb_build_object(
          'type', 'FeatureCollection',
          'features', COALESCE(jsonb_agg(jsonb_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON(
                CASE WHEN ST_SRID(geom) = 4326 THEN geom
                     ELSE ST_Transform(geom, 4326) END
            )::jsonb,
            'properties', to_jsonb(map_rows) - 'geom'
          )), '[]'::jsonb)
        ) AS geojson FROM map_rows
    """
    engine = _get_engine(database)
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL statement_timeout = '12000ms'"))
                data = conn.execute(text(wrapped), {"feature_limit": max_features}).scalar_one()
        return {"geojson": data, "type": "map"}
    finally:
        engine.dispose()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    message = data.get("message")
    if not user_id or not message:
        return jsonify({"error": "Missing user_id or message"}), 400
    output = service.chat(str(user_id), str(message), KNOWLEDGE_GRAPH)
    return jsonify({"response": output["response_to_user"], "status": output["status"],
                    "query_plan": output["query_plan"]})


@app.route("/knowledge-graph", methods=["GET"])
def get_knowledge_graph():
    return jsonify({"knowledge_graph": KNOWLEDGE_GRAPH})


@app.route("/history/<user_id>", methods=["GET"])
def get_history(user_id: str):
    return jsonify({"history": service.history(user_id)})


@app.route("/clear/<user_id>", methods=["POST"])
def clear_history(user_id: str):
    service.clear(user_id)
    return jsonify({"message": "Conversation cleared"})


def generate_sql_query(question: str, database: str, user_id: str | None = None):
    """Compatibility generator: all nodes now come from ONE ReAct agent."""
    yield from service.stream_sql(user_id, question, database)


@app.route("/generate-sql-stream", methods=["POST"])
def generate_sql_stream():
    data = request.get_json(silent=True) or {}
    question, database, user_id = data.get("question"), data.get("database"), data.get("user_id")
    if not question or not database:
        return jsonify({"error": "Missing question or database"}), 400

    def stream():
        yield f"data: {json.dumps({'node': 'question', 'result': question})}\n\n"
        try:
            for event in generate_sql_query(str(question), str(database), str(user_id or "anonymous")):
                if event["node"] == "FINAL" and event["status"] == "valid":
                    event["formatted_sql"] = colorize_sql(event["sql_query"])
                    LongMemory.append({"user_id": user_id, "question": question,
                                       "Query": event["sql_query"], "feedback": None, "comment": ""})
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'node':'ERROR','result':str(exc)[:500]})}\n\n"
    return Response(stream_with_context(stream()), mimetype="text/event-stream")


@app.route("/generate-sql", methods=["POST"])
def generate_sql():
    """Legacy non-streaming endpoint, retaining the established response shape."""
    data = request.get_json(silent=True) or {}
    question, database, user_id = data.get("question"), data.get("database"), data.get("user_id")
    if not question or not database:
        return jsonify({"error": "Missing question or database"}), 400
    events = list(generate_sql_query(str(question), str(database), str(user_id or "anonymous")))
    final = next((e for e in reversed(events) if e.get("node") == "FINAL"), None)
    sql = final["sql_query"] if final else "failed"
    if final and final.get("status") == "valid":
        LongMemory.append({"user_id": user_id, "question": question, "Query": sql,
                           "feedback": None, "comment": ""})
    return jsonify({"sql_query": sql, "formatted_sql": colorize_sql(sql) if final and final.get("status") == "valid" else None,
                    "question": question, "database": database, "progress_updates": events})


@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.get_json(silent=True) or {}
    uid, sql = data.get("user_id"), data.get("query")
    feedback, comment = data.get("feedback"), str(data.get("comment", ""))
    for entry in reversed(LongMemory):
        if entry.get("user_id") == uid and entry.get("Query") == sql:
            if feedback in {"like", "dislike"}:
                entry["feedback"] = feedback
            if comment:
                entry["comment"] = comment
            break
    UserFeedbacks.setdefault(str(uid), []).append(f"Feedback: {feedback}; comment: {comment[:300]}")
    return jsonify({"message": "Feedback recorded successfully"})


@app.route("/long-memory", methods=["GET"])
def get_long_memory():
    return jsonify({"memory": LongMemory, "total_entries": len(LongMemory)})


@app.route("/execute-query", methods=["POST", "OPTIONS"])
def execute_sql():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.get_json(silent=True) or {}
    sql, db = data.get("sql"), data.get("database")
    output = str(data.get("outType") or "table").lower()
    if not sql or not db:
        return jsonify({"error": "Missing SQL query or database"}), 400
    if not _read_only_select(str(sql)):
        return jsonify({"error": "Only a single read-only SELECT is allowed"}), 403
    if output not in {"table", "map"}:
        return jsonify({"error": "Unrecognized output type"}), 400
    try:
        if output == "map":
            return jsonify(_execute_map(str(sql), str(db)))
        rows, truncated = _query(str(sql), str(db))
        return jsonify({"table": rows, "type": "table", "truncated": truncated})
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)[:500]}), 400
    except Exception as exc:
        app.logger.error("Query execution failed: %s", exc)
        return jsonify({"error": "Database execution failed"}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("main.html")


@app.route("/workflow.html", methods=["GET"])
def workflow():
    return render_template("workflow.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, threaded=True)