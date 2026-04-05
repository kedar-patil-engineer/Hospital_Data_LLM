# tests/test_guardrails.py
"""
Unit tests for SQL safety guardrails in agent/etl_agent.py.

Tests:
  - Read-only enforcement (blocks INSERT, UPDATE, DELETE, CREATE, DROP, etc.)
  - Multi-statement blocking
  - Join guardrail (denials ↔ encounters bad join pattern)
  - normalize_sql_for_duckdb (date column rewriting)
  - compute_confidence (multi_agent.py)

Run with:
    pytest tests/test_guardrails.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.errors import AgentError
from agent.etl_agent import run_sql, enforce_join_guardrails, normalize_sql_for_duckdb
from agent.multi_agent import compute_confidence


# ---------------------------------------------------------------------------
# 1. READ-ONLY ENFORCEMENT
# ---------------------------------------------------------------------------

class TestReadOnlyGuardrail:
    """run_sql must block any non-SELECT statement."""

    @pytest.mark.parametrize("bad_sql", [
        "INSERT INTO patients VALUES (1, 'test')",
        "UPDATE patients SET first_name = 'hack' WHERE patient_id = '1'",
        "DELETE FROM claims_and_billing",
        "DROP TABLE patients",
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE patients ADD COLUMN x INT",
        "TRUNCATE TABLE encounters",
        "MERGE INTO patients USING src ON patients.id = src.id WHEN MATCHED THEN UPDATE SET name = src.name",
        "COPY patients TO '/tmp/dump.csv'",
    ])
    def test_blocks_write_operations(self, bad_sql):
        result = run_sql(bad_sql)
        # Phase 2: error format is now "[SQL_BLOCKED] ..." via AgentError.format()
        assert "SQL_BLOCKED" in result, (
            f"Expected 'SQL_BLOCKED' in result for: {bad_sql!r}\nGot: {result!r}"
        )

    def test_blocks_multi_statement(self):
        result = run_sql("SELECT 1; DROP TABLE patients")
        assert "SQL_BLOCKED" in result

    def test_blocks_empty_query(self):
        result = run_sql("")
        # Empty query returns SQL_EMPTY or SQL_FAILED error code
        assert "SQL_EMPTY" in result or "SQL_FAILED" in result or "empty" in result.lower()

    def test_allows_select(self):
        """A simple SELECT 1 should not be blocked."""
        result = run_sql("SELECT 1 AS test_col")
        assert "SQL_BLOCKED" not in result, f"SELECT was unexpectedly blocked: {result}"

    def test_allows_with_cte(self):
        """WITH ... SELECT (CTE) should pass the guardrail."""
        result = run_sql("WITH cte AS (SELECT 1 AS x) SELECT x FROM cte")
        assert "SQL_BLOCKED" not in result, f"CTE was unexpectedly blocked: {result}"

    def test_blocks_insert_case_insensitive(self):
        result = run_sql("insert into patients values (1)")
        assert "SQL_BLOCKED" in result

    def test_blocks_drop_mixed_case(self):
        result = run_sql("DrOp TaBlE patients")
        assert "SQL_BLOCKED" in result


# ---------------------------------------------------------------------------
# 2. JOIN GUARDRAIL
# ---------------------------------------------------------------------------

class TestJoinGuardrail:
    """enforce_join_guardrails must raise AgentError on bad join patterns."""

    def test_blocks_denials_encounter_id_join(self):
        bad_sql = """
            SELECT * FROM encounters e
            JOIN denials d ON d.claim_id = e.encounter_id
        """
        with pytest.raises(AgentError):
            enforce_join_guardrails(bad_sql)

    def test_blocks_encounters_claim_id_join(self):
        bad_sql = """
            SELECT * FROM denials d
            JOIN encounters e ON d.claim_id = e.encounter_id
        """
        with pytest.raises(AgentError):
            enforce_join_guardrails(bad_sql)

    def test_allows_correct_join_path(self):
        """Correct join path: encounters → claims_and_billing → denials."""
        good_sql = """
            SELECT e.encounter_id, d.denial_reason
            FROM encounters e
            JOIN claims_and_billing cb ON cb.encounter_id = e.encounter_id
            JOIN denials d ON d.claim_id = cb.claim_id
        """
        enforce_join_guardrails(good_sql)

    def test_allows_simple_select_no_joins(self):
        enforce_join_guardrails("SELECT * FROM patients")

    def test_allows_patients_encounters_join(self):
        good_sql = "SELECT * FROM patients p JOIN encounters e ON p.patient_id = e.patient_id"
        enforce_join_guardrails(good_sql)


# ---------------------------------------------------------------------------
# 3. DATE NORMALIZATION
# ---------------------------------------------------------------------------

class TestNormalizeSqlForDuckdb:
    """normalize_sql_for_duckdb should rewrite known date columns to try_strptime."""

    def test_rewrites_visit_date(self):
        sql = "SELECT visit_date FROM encounters WHERE visit_date > '2024-01-01'"
        result = normalize_sql_for_duckdb(sql)
        assert "try_strptime" in result, "Expected try_strptime rewrite for visit_date"

    def test_rewrites_claim_billing_date(self):
        sql = "SELECT claim_billing_date FROM claims_and_billing"
        result = normalize_sql_for_duckdb(sql)
        assert "try_strptime" in result

    def test_rewrites_discharge_date(self):
        sql = "SELECT discharge_date FROM encounters"
        result = normalize_sql_for_duckdb(sql)
        assert "try_strptime" in result

    def test_leaves_non_date_columns_unchanged(self):
        sql = "SELECT patient_id, first_name FROM patients"
        result = normalize_sql_for_duckdb(sql)
        assert "try_strptime" not in result

    def test_rewrites_str_to_date_function(self):
        sql = "SELECT STR_TO_DATE(visit_date, '%Y-%m-%d') FROM encounters"
        result = normalize_sql_for_duckdb(sql)
        assert "try_strptime" in result


# ---------------------------------------------------------------------------
# 4. CONFIDENCE SCORING
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    """compute_confidence must return sensible scores for known inputs."""

    def test_zero_rows_gives_low_score(self):
        score = compute_confidence(
            sql_result="Row count: 0\n\n(no rows)",
            sql="SELECT * FROM patients",
            tool_calls_made=1,
            profiles_used=0,
        )
        assert score < 0.55, f"Expected low score for zero rows, got {score}"

    def test_healthy_result_gives_high_score(self):
        score = compute_confidence(
            sql_result="Row count: 42\n\n| col1 | col2 |\n|------|------|\n| a    | b    |",
            sql="SELECT patient_id, first_name FROM patients LIMIT 42",
            tool_calls_made=2,
            profiles_used=1,
        )
        assert score >= 0.80, f"Expected high score for healthy result, got {score}"

    def test_sql_error_gives_very_low_score(self):
        score = compute_confidence(
            sql_result="SQL failed: missing column 'foo'.",
            sql="SELECT foo FROM patients",
            tool_calls_made=1,
            profiles_used=0,
        )
        assert score < 0.50, f"Expected low score for SQL error, got {score}"

    def test_sql_blocked_gives_very_low_score(self):
        score = compute_confidence(
            sql_result="SQL blocked: only read-only SELECT queries are allowed.",
            sql="DROP TABLE patients",
            tool_calls_made=0,
            profiles_used=0,
        )
        assert score < 0.50

    def test_many_joins_reduces_score(self):
        simple_score = compute_confidence(
            sql_result="Row count: 10\n\n| a | b |",
            sql="SELECT a FROM t",
            tool_calls_made=1,
            profiles_used=0,
        )
        complex_score = compute_confidence(
            sql_result="Row count: 10\n\n| a | b |",
            sql="SELECT a FROM t JOIN t2 ON t.id=t2.id JOIN t3 ON t2.id=t3.id JOIN t4 ON t3.id=t4.id",
            tool_calls_made=1,
            profiles_used=0,
        )
        assert complex_score < simple_score, (
            f"Expected complex JOIN query to have lower confidence. "
            f"simple={simple_score}, complex={complex_score}"
        )

    def test_profiling_boosts_score(self):
        base = compute_confidence(
            sql_result="Row count: 10\n\n| a | b |",
            sql="SELECT a FROM t",
            tool_calls_made=1,
            profiles_used=0,
        )
        with_profile = compute_confidence(
            sql_result="Row count: 10\n\n| a | b |",
            sql="SELECT a FROM t",
            tool_calls_made=3,
            profiles_used=2,
        )
        assert with_profile >= base, (
            f"Expected profiling to boost or maintain score. base={base}, with_profile={with_profile}"
        )

    def test_score_always_in_valid_range(self):
        extreme_cases = [
            ("Row count: 0\n\n(no rows)", "SELECT * FROM t", 0, 0),
            ("SQL failed: table not found", "SELECT x FROM nonexistent", 0, 0),
            ("Row count: 500\n\n| a |", "SELECT a FROM t", 10, 5),
        ]
        for sql_result, sql, tool_calls, profiles in extreme_cases:
            score = compute_confidence(sql_result, sql, tool_calls, profiles)
            assert 0.10 <= score <= 0.98, (
                f"Score {score} out of bounds [0.10, 0.98] for input: {sql_result[:40]!r}"
            )

    def test_null_heavy_result_reduces_score(self):
        null_result = "Row count: 5\n\n| col |\n|-----|\n| None |\n| None |\n| None |\n| None |\n| None |"
        clean_result = "Row count: 5\n\n| col |\n|-----|\n| abc |\n| def |\n| ghi |\n| jkl |\n| mno |"
        null_score = compute_confidence(null_result, "SELECT col FROM t", 1, 0)
        clean_score = compute_confidence(clean_result, "SELECT col FROM t", 1, 0)
        assert null_score <= clean_score, (
            f"Expected null-heavy result to score <= clean result. null={null_score}, clean={clean_score}"
        )
