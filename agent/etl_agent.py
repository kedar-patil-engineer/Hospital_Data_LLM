# agent/etl_agent.py
#
# Phase 2 changes:
#   1. Error strings replaced with AgentError.format() using ErrorType codes
#   2. COLUMN_DICTIONARY loaded from config/field_dict.yaml (maintainable, no code changes needed)
#   3. explain_field() covers all ~60 columns via YAML
#
# Everything else (SQL guardrails, schema memory, run_agent, etc.) is unchanged.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import duckdb
from dotenv import load_dotenv
from agent.multi_agent import MultiAgentOrchestrator
from agent.errors import AgentError, ErrorType
from llm.factory import get_llm_provider
from agent.sql_validator import validate_sql_against_schema, format_validation_error

import os
import re

# ---------------------------------------------------------------------------
# Environment & DB wiring
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "hospital.duckdb"
CONFIG_DIR = BASE_DIR / "config"
FIELD_DICT_PATH = CONFIG_DIR / "field_dict.yaml"

load_dotenv()


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection to the hospital.duckdb file.
    Each call returns a NEW connection; callers should use context managers:
        with get_connection() as con: ...
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB file not found at {DB_PATH}. "
            f"Run etl/load_duckdb.py first."
        )
    return duckdb.connect(str(DB_PATH))


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return get_connection()


# ---------------------------------------------------------------------------
# SQL tools
# ---------------------------------------------------------------------------

def _duckdb_date(expr: str) -> str:
    return (
        f"COALESCE("
        f"CAST(try_strptime({expr}, '%Y-%m-%d') AS DATE), "
        f"CAST(try_strptime({expr}, '%d-%m-%Y') AS DATE), "
        f"CAST({expr} AS DATE)"
        f")"
    )


def normalize_sql_for_duckdb(sql: str) -> str:
    """Normalize SQL to handle common date issues for DuckDB."""
    sql = re.sub(
        r"(STR_TO_DATE|TO_DATE)\s*\(\s*([^,]+)\s*,\s*([^)]+)\)",
        r"try_strptime(\2, \3)",
        sql,
        flags=re.I,
    )

    date_cols = [
        "claim_billing_date", "visit_date", "encounter_date",
        "admission_date", "discharge_date", "denial_date",
        "appeal_resolution_date", "test_date", "prescribed_date",
        "registration_date", "procedure_date",
    ]

    for col in date_cols:
        sql = re.sub(
            rf"\b{col}\b",
            f"""
            CASE
              WHEN {col} LIKE '__-__-____'
                THEN try_strptime({col}, '%d-%m-%Y')
              ELSE try_strptime({col}, '%Y-%m-%d')
            END
            """,
            sql,
            flags=re.I,
        )

    return sql


def enforce_join_guardrails(sql: str) -> None:
    """Hard guardrails to prevent known-bad joins. Raises AgentError on violation."""
    s = re.sub(r"\s+", " ", sql.strip().lower())

    if re.search(r"join\s+denials\b.*\bon\b.*claim_id\s*=\s*.*encounter_id", s) or re.search(
        r"join\s+encounters\b.*\bon\b.*claim_id\s*=\s*.*encounter_id", s
    ):
        raise AgentError(
            ErrorType.INVALID_JOIN,
            "Do NOT join denials to encounters using claim_id = encounter_id.\n"
            "Correct path: encounters.encounter_id = claims_and_billing.encounter_id\n"
            "              claims_and_billing.claim_id = denials.claim_id",
        )


def run_sql(sql: str, max_rows: int = 500) -> str:
    """
    Executes SQL in DuckDB (READ-ONLY).
    - Blocks non-SELECT queries (returns AgentError-formatted string)
    - Shows up to max_rows in UI
    - Always includes row_count
    - Provides helpful schema hints on common errors
    """
    import pandas as pd

    def _df_to_md(df: pd.DataFrame) -> str:
        if df is None:
            return "No results."
        if df.empty:
            return "Row count: 0\n\n(no rows)"
        return f"Row count: {len(df)}\n\n" + df.head(max_rows).to_markdown(index=False)

    raw = (sql or "").strip()
    if not raw:
        return AgentError.format(ErrorType.SQL_EMPTY, "No SQL query was provided.")

    s = re.sub(r"\s+", " ", raw.lower()).strip()

    if ";" in s:
        return AgentError.format(ErrorType.SQL_BLOCKED, "Multiple statements are not allowed.")
    if not s.startswith("select") and not s.startswith("with"):
        return AgentError.format(ErrorType.SQL_BLOCKED, "Only read-only SELECT queries are allowed.")
    if re.search(r"\b(insert|update|delete|create|drop|alter|truncate|merge|copy)\b", s):
        return AgentError.format(ErrorType.SQL_BLOCKED, "Only read-only SELECT queries are allowed.")

    try:
        sql_norm = normalize_sql_for_duckdb(raw)
        enforce_join_guardrails(sql_norm)

        # NEW:
        with get_connection() as con:
            validation = validate_sql_against_schema(sql_norm, con)
            if not validation.valid:
                return format_validation_error(validation)
            df = con.execute(sql_norm).fetchdf()
            return _df_to_md(df)

    except AgentError as ae:
        # Join guardrail — already a typed error
        return str(ae)

    except Exception as e:
        msg = str(e)

        try:
            with get_connection() as con:
                m_col = re.search(r'Binder Error:.*?column "(.*?)".*?not found', msg, re.IGNORECASE)
                m_tbl = re.search(r'Catalog Error:.*?Table with name (.*?) does not exist', msg, re.IGNORECASE)

                if m_tbl:
                    tbl = m_tbl.group(1)
                    tables = con.execute("SHOW TABLES").fetchdf()
                    return (
                        AgentError.format(ErrorType.SCHEMA_MISMATCH, f"Table not found: {tbl}") +
                        f"\n\nAvailable tables:\n{tables.to_markdown(index=False)}"
                    )

                if m_col:
                    missing_col = m_col.group(1)
                    tbls = re.findall(r'from\s+([a-zA-Z_]\w*)|join\s+([a-zA-Z_]\w*)', raw, re.IGNORECASE)
                    tbls = [t[0] or t[1] for t in tbls if (t[0] or t[1])]
                    tbls = list(dict.fromkeys(tbls))[:5]

                    suggestions = []
                    for t in tbls:
                        try:
                            info = con.execute(f"PRAGMA table_info('{t}')").fetchdf()
                            cols = info["name"].tolist()
                            close = [c for c in cols if missing_col.lower() in c.lower()]
                            suggestions.append((t, cols[:50], close))
                        except Exception:
                            pass

                    out = [AgentError.format(ErrorType.SCHEMA_MISMATCH, f"Missing column '{missing_col}'.")]
                    if suggestions:
                        out.append("\nSchema hints (table → columns):")
                        for t, cols, close in suggestions:
                            out.append(f"\n- {t}: {', '.join(cols)}")
                            if close:
                                out.append(f"  close matches: {', '.join(close)}")
                    out.append("\nFix your SQL using correct column names.")
                    return "\n".join(out)

        except Exception:
            pass

        return AgentError.format(ErrorType.SQL_FAILED, msg)


def list_columns(table_name: str) -> str:
    table_name = (table_name or "").strip()
    if not table_name:
        return "[]"

    q = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = ?
    ORDER BY ordinal_position
    """

    with get_connection() as con:
        df = con.execute(q, [table_name]).fetchdf()

    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)


def profile_table(table_name: str) -> str:
    """Lightweight data-quality profile for a single table."""
    if not table_name:
        return "No table name provided."

    with get_connection() as con:
        try:
            tables_df = con.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                """
            ).fetchdf()
        except Exception as e:
            return AgentError.format(ErrorType.SQL_FAILED, f"Error checking tables: {e}")

        if table_name not in set(tables_df["table_name"]):
            return AgentError.format(ErrorType.SCHEMA_MISMATCH, f"Table '{table_name}' does not exist in DuckDB.")

        row_count = con.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}").fetchone()[0]

        cols_df = con.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position;
            """,
            [table_name],
        ).fetchdf()

        null_stats = []
        numeric_stats = []

        for _, row in cols_df.iterrows():
            col = row["column_name"]
            data_type = row["data_type"]

            null_count = con.execute(
                f"SELECT SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) FROM {table_name}"
            ).fetchone()[0]
            null_stats.append((col, null_count))

            if data_type.upper() in {"INTEGER", "BIGINT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"}:
                mn, mx, avg = con.execute(
                    f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}"
                ).fetchone()
                numeric_stats.append((col, mn, mx, avg))

    lines = [f"Table '{table_name}' profile:", f"- Row count: {row_count}", ""]
    lines.append("Null counts per column:")
    for col, n in null_stats:
        lines.append(f"  - {col}: {n} nulls")

    if numeric_stats:
        lines.append("")
        lines.append("Numeric column ranges:")
        for col, mn, mx, avg in numeric_stats:
            lines.append(f"  - {col}: min={mn}, max={mx}, avg={avg}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Column dictionary — loaded from config/field_dict.yaml
# ---------------------------------------------------------------------------

def _load_field_dict() -> Dict[str, str]:
    """
    Load the column dictionary from config/field_dict.yaml.
    Falls back to an empty dict if the file is missing,
    so the app still runs without the config directory.
    """
    if not FIELD_DICT_PATH.exists():
        return {}

    try:
        import yaml  # PyYAML — already in requirements via uvicorn[standard] deps
        with open(FIELD_DICT_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass

    return {}


# Load once at module import time — cheap, file is small
COLUMN_DICTIONARY: Dict[str, str] = _load_field_dict()


def explain_field(field_name: str) -> str:
    """
    Provide a natural-language explanation of a column or table.field name.
    Uses the full YAML-backed dictionary (all ~60 columns across 9 tables)
    with fuzzy substring matching as fallback.
    """
    if not field_name:
        return "Please provide a column or table.field name to explain."

    # Exact key match
    if field_name in COLUMN_DICTIONARY:
        return f"{field_name}: {COLUMN_DICTIONARY[field_name]}"

    # Fuzzy/substring match — check field against all keys
    field_lower = field_name.lower()
    best_match = None
    for key, desc in COLUMN_DICTIONARY.items():
        if field_lower in key.lower() or key.lower().endswith(f".{field_lower}"):
            best_match = (key, desc)
            break

    if best_match:
        return f"(Best match: {best_match[0]}) {best_match[1]}"

    return (
        f"No description found for '{field_name}'. "
        f"Use profile_table or list_columns to explore the schema. "
        f"You can also add a description to config/field_dict.yaml."
    )


# ---------------------------------------------------------------------------
# SQL Memory: schema introspection
# ---------------------------------------------------------------------------

_SCHEMA_MEMORY_CACHE: Optional[str] = None


def _build_schema_memory() -> str:
    lines: list[str] = []
    with get_connection() as con:
        tables_df = con.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
        ).fetchdf()

        tables = tables_df["table_name"].tolist()
        lines.append("DuckDB schema (tables → columns):")

        for table in tables:
            cols_df = con.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = ?
                ORDER BY ordinal_position;
                """,
                [table],
            ).fetchdf()

            col_parts = [
                f"{row['column_name']} ({row['data_type']})"
                for _, row in cols_df.iterrows()
            ]
            lines.append(f"- {table}: " + ", ".join(col_parts))

    lines.append("")
    lines.append("Common join keys / relationships:")
    lines.append("- patients.patient_id = encounters.patient_id")
    lines.append("- encounters.encounter_id = claims_and_billing.encounter_id")
    lines.append("- encounters.encounter_id = lab_tests.encounter_id")
    lines.append("- encounters.encounter_id = diagnoses.encounter_id")
    lines.append("- encounters.encounter_id = procedures.encounter_id")
    lines.append("- medications.encounter_id = encounters.encounter_id")
    lines.append("- claims_and_billing.claim_id = denials.claim_id")
    lines.append("- encounters.provider_id = providers.provider_id")

    return "\n".join(lines)


def get_schema_memory() -> str:
    """Compact schema memory injected into LLM prompts."""
    with get_connection() as con:
        tables_df = con.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='main' AND table_type='BASE TABLE'
            ORDER BY table_name
            """
        ).fetchdf()

        tables = tables_df["table_name"].tolist()
        lines = ["SCHEMA:"]
        for t in tables:
            cols_df = con.execute(f"PRAGMA table_info('{t}')").fetchdf()
            cols = cols_df["name"].tolist()
            lines.append(f"- {t}: {', '.join(cols)}")

    return "\n".join(lines)


def get_schema_summary() -> str:
    return _build_schema_memory()


# ---------------------------------------------------------------------------
# Tools specification for OpenAI function calling
# ---------------------------------------------------------------------------

def get_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": (
                    "Execute a read-only SQL SELECT query against the DuckDB "
                    "hospital dataset. Use this for any data exploration, "
                    "aggregations, joins, and analytics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The full SQL SELECT statement to execute."},
                        "max_rows": {"type": "integer", "description": "Maximum number of rows to preview (default 50)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_columns",
                "description": "List column names and types for a given table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Name of the table."}
                    },
                    "required": ["table_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "profile_table",
                "description": "Profile a DuckDB table: row count, null counts per column, numeric ranges.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Name of the table to profile."}
                    },
                    "required": ["table_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_field",
                "description": (
                    "Explain the meaning of a field or table.field in the dataset. "
                    "Covers all ~60 columns across 9 tables."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field_name": {
                            "type": "string",
                            "description": "The column or table.field to explain, e.g. 'claims_and_billing.billed_amount'.",
                        }
                    },
                    "required": ["field_name"],
                },
            },
        },
    ]


def dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute the corresponding Python function for a tool call."""
    try:
        if name == "run_sql":
            query = (arguments.get("query") or arguments.get("sql") or "").strip()
            max_rows = int(arguments.get("max_rows", 50) or 50)
            return run_sql(query, max_rows)

        elif name == "list_columns":
            table = (arguments.get("table_name") or arguments.get("table") or "").strip()
            return list_columns(table)

        elif name == "profile_table":
            table = (arguments.get("table_name") or arguments.get("table") or "").strip()
            return profile_table(table)

        elif name == "explain_field":
            field = (arguments.get("field_name") or arguments.get("field") or "").strip()
            return explain_field(field)

        return AgentError.format(ErrorType.TOOL_UNKNOWN, f"Unknown tool: {name}")

    except AgentError as ae:
        return str(ae)
    except Exception as e:
        return AgentError.format(ErrorType.TOOL_FAILED, f"Tool '{name}' failed: {e}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """
You are a Healthcare ETL & Data-Quality Agent working for a CA hospital.
Your job is to:

1. Help data engineers, analysts, and compliance teams explore the hospital
   DuckDB dataset (patients, encounters, claims, diagnoses, procedures,
   medications, labs, providers, denials).
2. Use ONLY read-only SQL (SELECT) via the run_sql tool when you need data.
   Never attempt INSERT, UPDATE, DELETE, CREATE, or DROP.
3. When writing SQL:
   - Prefer explicit JOINs with ON clauses.
   - Use the join key hints provided in the schema memory.
   - Use WHERE clauses to constrain by date, patient, or encounter when useful.
4. When the user asks questions that depend on the data, call tools instead
   of guessing. Run as many tool calls as needed to answer confidently.
5. When profiling data quality, use profile_table and also suggest additional
   checks the user could run (e.g., outliers, referential integrity).
6. If the user asks about a column meaning, use explain_field — it now covers
   all ~60 columns across all 9 tables.

Be concise and structured. For SQL examples, always show the full query in
a fenced code block with 'sql' as the language.
""".strip()

SQL_RULES = """
You are querying DuckDB.

CRITICAL JOIN GRAPH (do NOT invent joins):
- encounters.encounter_id = claims_and_billing.encounter_id
- claims_and_billing.claim_id = denials.claim_id
- encounters.provider_id = providers.provider_id
NEVER join denials directly to encounters using claim_id = encounter_id.

COLUMN RULE:
- Do NOT guess columns. If unsure, call list_columns or run:
  PRAGMA table_info('<table_name>');

DATE RULE (DuckDB):
- If date columns are VARCHAR (common in CSV loads), use:
  try_strptime(col, '%d-%m-%Y')  OR  try_strptime(col, '%Y-%m-%d')
- For month buckets:
  strftime(try_strptime(col, '%d-%m-%Y'), '%Y-%m')
"""


def build_system_prompt() -> str:
    schema_text = get_schema_memory()
    return (
        BASE_SYSTEM_PROMPT
        + "\n\n---\n\nSQL rules (DuckDB + join discipline):\n"
        + SQL_RULES
        + "\n\n---\n\nCurrent DuckDB schema and join hints:\n"
        + schema_text
        + "\n\nUse these rules + schema when writing SQL."
    )


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    chat_history: list | None = None,
    max_steps: int = 4,
    provider_name: str = "openai",
    openai_model: str | None = None,
    local_model: str | None = None,
) -> str:
    from agent.eval_tracker import EvalTracker

    tracker = EvalTracker(prompt_version="v1")
    tracker.provider = (provider_name or "").lower()

    history_text = ""
    if chat_history:
        trimmed = chat_history[-6:]
        chunks = []
        for item in trimmed:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                role = str(item[0])
                content = str(item[1])
            else:
                continue
            chunks.append(f"{role.upper()}: {content}")
        history_text = "\n".join(chunks)

    if history_text:
        user_message = (
            "Conversation so far (for context):\n"
            f"{history_text}\n\n"
            "Current user request:\n"
            f"{user_message}"
        )

    provider = get_llm_provider(
        provider_name,
        openai_model=openai_model,
        local_model=local_model,
    )

    model = (
        (openai_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        if tracker.provider in {"openai", "oa"}
        else (local_model or os.getenv("OLLAMA_MODEL") or "llama3.1").strip()
    )
    tracker.model = model

    orchestrator = MultiAgentOrchestrator(
        llm_provider=provider,
        model=model,
        tool_dispatcher=dispatch_tool,
        schema_provider=get_schema_memory,
        tracker=tracker,
    )

    answer = orchestrator.run(user_message)
    tracker.persist_jsonl("logs/raw/eval_log.jsonl")

    return answer
