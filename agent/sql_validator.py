# agent/sql_validator.py
"""
Schema-aware SQL pre-execution validator.

Phase 3 addition — catches LLM hallucinations BEFORE they hit DuckDB.

The problem:
    LLMs occasionally invent table names or column names that don't exist.
    The existing join guardrail catches one specific bad pattern, but
    doesn't catch arbitrary hallucinated schema references.

This module:
    1. Extracts all table references from a SQL query
    2. Extracts all column references (dotted: table.column)
    3. Validates each against the real DuckDB schema
    4. Returns a ValidationResult with pass/fail + specific violations

Used in etl_agent.py's run_sql() BEFORE execute — if validation fails,
the error is returned immediately without touching DuckDB.

Research value:
    This is a genuine contribution to LLM-augmented ETL safety.
    The validate_sql_against_schema() function can be cited in the
    dissertation as a governance mechanism that prevents data corruption
    from LLM hallucinations at the schema level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional

import duckdb


@dataclass
class ValidationResult:
    """
    Result of schema validation for a SQL query.

    Attributes:
        valid         : True if no violations found
        violations    : List of human-readable violation messages
        bad_tables    : Table names referenced in SQL but not in schema
        bad_columns   : Dotted column refs (table.col) not in schema
        warnings      : Non-blocking issues (e.g. unqualified column names)
    """
    valid: bool = True
    violations: List[str] = field(default_factory=list)
    bad_tables: List[str] = field(default_factory=list)
    bad_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for UI / log output."""
        if self.valid and not self.warnings:
            return "Schema validation passed."
        lines = []
        if not self.valid:
            lines.append("Schema validation FAILED:")
            for v in self.violations:
                lines.append(f"  - {v}")
        if self.warnings:
            lines.append("Warnings (non-blocking):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _get_schema(con: duckdb.DuckDBPyConnection) -> Dict[str, Set[str]]:
    """
    Fetch the real DuckDB schema as {table_name: {col1, col2, ...}}.
    Only BASE TABLE objects in the 'main' schema are included.
    """
    tables_df = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    ).fetchdf()

    schema: Dict[str, Set[str]] = {}
    for tbl in tables_df["table_name"].tolist():
        cols_df = con.execute(f"PRAGMA table_info('{tbl}')").fetchdf()
        schema[tbl.lower()] = {c.lower() for c in cols_df["name"].tolist()}

    return schema


def _extract_table_refs(sql: str) -> Set[str]:
    """
    Extract table names referenced in FROM and JOIN clauses.
    Returns lowercase set.

    Handles:
        FROM table_name
        FROM table_name alias
        JOIN table_name ON ...
        LEFT/RIGHT/INNER/OUTER JOIN table_name

    Does NOT handle subqueries perfectly — treats CTE names as tables
    and then filters them out during validation.
    """
    # Remove string literals to avoid false matches inside quotes
    sql_clean = re.sub(r"'[^']*'", "''", sql)
    sql_clean = re.sub(r'"[^"]*"', '""', sql_clean)

    patterns = [
        r'\bFROM\s+([a-zA-Z_]\w*)',
        r'\bJOIN\s+([a-zA-Z_]\w*)',
    ]
    refs: Set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, sql_clean, re.IGNORECASE):
            name = match.group(1).lower()
            # Skip SQL keywords that can follow FROM/JOIN
            if name not in {"select", "where", "on", "lateral", "unnest", "values"}:
                refs.add(name)
    return refs


def _extract_cte_names(sql: str) -> Set[str]:
    """
    Extract CTE names defined in WITH clauses so they can be
    excluded from table validation (they're not real tables).
    """
    pattern = r'\bWITH\b.*?([a-zA-Z_]\w*)\s+AS\s*\('
    names: Set[str] = set()
    for match in re.finditer(pattern, sql, re.IGNORECASE | re.DOTALL):
        names.add(match.group(1).lower())
    return names


def _extract_dotted_column_refs(sql: str) -> Set[str]:
    """
    Extract explicit table.column references from SQL.
    Returns set of "table.column" strings (both lowercased).

    Only catches dotted refs — unqualified column names are too
    ambiguous to validate without full SQL parsing.
    """
    # Remove string literals first
    sql_clean = re.sub(r"'[^']*'", "''", sql)
    pattern = r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b'
    refs: Set[str] = set()
    for match in re.finditer(pattern, sql_clean):
        table = match.group(1).lower()
        col = match.group(2).lower()
        # Filter out common non-table patterns (e.g. schema.table qualifiers,
        # function calls like string.length, numeric literals)
        if table not in {"information_schema", "pg_catalog", "duckdb_tables"}:
            refs.add(f"{table}.{col}")
    return refs


def validate_sql_against_schema(
    sql: str,
    con: duckdb.DuckDBPyConnection,
    strict_columns: bool = True,
) -> ValidationResult:
    """
    Validate a SQL query against the real DuckDB schema before execution.

    Args:
        sql            : The SQL string to validate (SELECT only)
        con            : Active DuckDB connection
        strict_columns : If True, validate dotted table.column references.
                         Set False to skip column checks (faster, less strict).

    Returns:
        ValidationResult with valid=True if no violations found.

    Example:
        result = validate_sql_against_schema(sql, con)
        if not result.valid:
            return result.summary()   # return error to UI
        # else proceed with execution
    """
    result = ValidationResult()

    try:
        schema = _get_schema(con)
    except Exception as e:
        # Can't fetch schema — skip validation rather than blocking execution
        result.warnings.append(f"Schema fetch failed, validation skipped: {e}")
        return result

    real_tables = set(schema.keys())

    # --- 1. Table reference validation ---
    referenced_tables = _extract_table_refs(sql)
    cte_names = _extract_cte_names(sql)

    # Table aliases: we can't reliably extract all aliases without a full parser,
    # so we only flag names that look like real table-name patterns (contain _)
    # or are longer than 3 chars (short aliases like 'e', 'cb', 'd' are likely aliases)
    for tbl in referenced_tables:
        if tbl in cte_names:
            continue  # CTE — not a real table, skip
        if tbl in real_tables:
            continue  # Valid table
        if len(tbl) <= 3:
            # Short name — likely an alias, not a table name. Warn but don't fail.
            result.warnings.append(
                f"'{tbl}' not found in schema — if this is a table alias, ignore this warning."
            )
        else:
            # Long name that doesn't match any real table — likely hallucination
            result.valid = False
            result.bad_tables.append(tbl)
            result.violations.append(
                f"Table '{tbl}' does not exist in DuckDB. "
                f"Available tables: {', '.join(sorted(real_tables))}."
            )

    # --- 2. Dotted column reference validation ---
    if strict_columns:
        dotted_refs = _extract_dotted_column_refs(sql)
        for ref in dotted_refs:
            parts = ref.split(".", 1)
            if len(parts) != 2:
                continue
            tbl, col = parts

            if tbl in cte_names:
                continue  # CTE column ref — can't validate
            if tbl not in real_tables:
                continue  # Already flagged as bad table above

            real_cols = schema.get(tbl, set())
            if col not in real_cols:
                result.valid = False
                result.bad_columns.append(ref)
                close = [c for c in real_cols if col in c or c in col]
                msg = f"Column '{tbl}.{col}' does not exist."
                if close:
                    msg += f" Did you mean: {', '.join(sorted(close)[:3])}?"
                else:
                    msg += f" Available columns: {', '.join(sorted(real_cols)[:10])}."
                result.violations.append(msg)

    return result


def format_validation_error(result: ValidationResult) -> str:
    """
    Format a failed ValidationResult as a string suitable for
    returning from run_sql() to the UI and agent.
    """
    from agent.errors import AgentError, ErrorType
    lines = [AgentError.format(ErrorType.SCHEMA_MISMATCH, "SQL references schema elements that do not exist.")]
    for v in result.violations:
        lines.append(f"  - {v}")
    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")
    lines.append("\nFix the SQL using the correct table/column names before retrying.")
    return "\n".join(lines)
