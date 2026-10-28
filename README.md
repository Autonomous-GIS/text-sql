# Spatial Text-to-SQL — Unified ReAct backend

This application keeps `templates/main.html` and `templates/workflow.html` unchanged. `app.py` preserves the original Flask API and the SSE JSON node names (`question`, `EEA`, `MRA`, `QLA`, `SQLA`, `RA`, `SQL`, `FINAL`, `ERROR`), but delegates SQL solving to one flexible general-purpose Python ReAct agent.

## Files

- `app.py`: UI-compatible HTTP API, read-only query execution, local SQL highlighting.
- `unified_postgis_agent.py`: single-agent application adapter, per-user sessions, existing progress-node mapper, final SQL validation.
- `GeneralAgent/general_python_agent.py`: the supplied general Python agent with one small backwards-compatible addition to `execute_python` for per-request environment overrides.
- `GeneralAgent/experience_manager.py`: supplied optional experience manager, unchanged.
- `Skills/postgis_sql_skill.md`: PostGIS exploration, semantic correctness, query-efficiency, and output guidance for the general agent.
- `metadata/`, `KNOWLEDGE_GRAPH`: original catalog and database metadata.

## Setup

Install dependencies with `pip install -r requirements.txt`. Set `OPENAI_API_KEY`, `DATABASE_URL` (or `POSTGIS_DATABASE_URL_TEMPLATE`), and optionally `POSTGIS_AGENT_MODEL`. For local development, `python app.py` starts Flask on port 3000. Existing HTML templates are unchanged.

The configured PostgreSQL role **must** be SELECT-only with access limited to the intended databases. The subprocess may run model-generated Python and therefore must be isolated at the OS/container/network layer in production: prompt instructions, `PGOPTIONS`, and SQL keyword checks are not a sandbox. Never give it superuser credentials or production secrets. Keep the metadata pickle trusted and under source control. SQL queries are screened as single SELECT statements and run in read-only transactions with a statement timeout. Exploratory Python tool calls have a 60-second process timeout; this is a tool-runtime bound, not an unrestricted sandbox.

## Behavior

The same per-user agent handles chat planning and query execution; response formatting is determined by the application request. `execute_python` can inspect the selected database, existing pickle metadata, PostgreSQL indexes, and query plans as needed. It can iterate and change course freely. Its final standalone SQL is validated by a separate read-only PostgreSQL `EXPLAIN` before a `valid` result is emitted; an executable plan does **not** prove semantic correctness or predict execution time.

The compatibility adapter maps observed progress and final validation into the old UI node names. The UI diagram still calls these milestones agents, but no specialized EEA/MRA/QLA/SQLA/RA LLM workers are executed. A skipped activity is marked explicitly rather than fabricated. Stream output and the `/execute-query` map/table result envelope preserve the existing interface. Map GeoJSON is projected to WGS84 and capped at `MAX_MAP_FEATURES` (default 5000); table output is capped at 5000 rows.

The optional `POSTGIS_EXPERIENCE_ENABLED=true` uses the supplied experience manager for conservative post-turn reflection. Disabled by default.

## Testing

The unit and integration contract tests in `tests/` mock network/LLM/DB connections; live PostgreSQL/PostGIS execution must still be verified in the deployed environment. In particular verify a simple query, a discovered schema query, a costly spatial join, SQL error recovery, map projection, and exact SSE events with both unchanged HTML files.