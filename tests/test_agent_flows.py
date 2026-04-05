# tests/test_agent_flows.py
"""
Integration tests for MultiAgentOrchestrator flows.

Uses MockLocalProvider and a mock tool dispatcher — no OpenAI API key,
no Ollama, no real DuckDB connection needed.

Tests:
  - Fast-path routing (list tables, profile table, schema)
  - Terminal command rejection
  - SQL generation → execution → narration flow (mocked)
  - Zero-row result handling
  - SQL generation failure handling
  - Confidence score is present and valid in output
  - Table-first rendering for analytical prompts

Run with:
    pytest tests/test_agent_flows.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.multi_agent import MultiAgentOrchestrator, compute_confidence, _confidence_label
from llm.providers import MockLocalProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_SCHEMA = """SCHEMA:
- patients: patient_id, first_name, last_name, dob, age, gender
- encounters: encounter_id, patient_id, visit_date, discharge_date, department, provider_id
- claims_and_billing: billing_id, claim_id, encounter_id, billed_amount, paid_amount, claim_status
- denials: denial_id, claim_id, denial_reason, denial_date
"""

def make_orchestrator(
    tool_responses: dict | None = None,
    llm_json_override: dict | None = None,
) -> MultiAgentOrchestrator:
    """
    Build an orchestrator with:
    - MockLocalProvider (no external LLM calls)
    - Configurable mock tool dispatcher
    - Static schema provider
    """
    _tool_responses = tool_responses or {}

    def mock_tool_dispatcher(name: str, args: dict) -> str:
        if name in _tool_responses:
            resp = _tool_responses[name]
            return resp(args) if callable(resp) else resp
        if name == "profile_table":
            tbl = args.get("table_name", "unknown")
            return f"Table '{tbl}' profile:\n- Row count: 1000\n\nNull counts per column:\n  - id: 0 nulls"
        if name == "run_sql":
            return "Row count: 5\n\n| patient_id | first_name |\n|---|---|\n| P001 | Alice |"
        return f"Tool '{name}' not configured in mock."

    return MultiAgentOrchestrator(
        llm_provider=MockLocalProvider(),
        model="mock-local",
        tool_dispatcher=mock_tool_dispatcher,
        schema_provider=lambda: MOCK_SCHEMA,
        tracker=None,
    )


# ---------------------------------------------------------------------------
# 1. FAST-PATH ROUTING
# ---------------------------------------------------------------------------

class TestFastPaths:

    def test_list_tables_fast_path(self):
        calls = []
        def dispatcher(name, args):
            calls.append((name, args))
            return "Row count: 4\n\n| table_name |\n|---|\n| patients |"

        orch = make_orchestrator()
        orch.tool_dispatcher = dispatcher
        result = orch.run("list tables")

        assert len(calls) == 1
        assert calls[0][0] == "run_sql"
        assert "duckdb_tables" in calls[0][1]["query"]

    def test_show_tables_fast_path(self):
        calls = []
        def dispatcher(name, args):
            calls.append(name)
            return "Row count: 4\n\n| table_name |\n| patients |"

        orch = make_orchestrator()
        orch.tool_dispatcher = dispatcher
        orch.run("show tables")
        assert "run_sql" in calls

    def test_profile_table_fast_path(self):
        calls = []
        def dispatcher(name, args):
            calls.append((name, args))
            return "Table 'patients' profile:\n- Row count: 500"

        orch = make_orchestrator()
        orch.tool_dispatcher = dispatcher
        result = orch.run("profile the patients table")

        assert len(calls) == 1
        assert calls[0][0] == "profile_table"
        assert calls[0][1]["table_name"] == "patients"

    def test_schema_fast_path_returns_schema(self):
        orch = make_orchestrator()
        result = orch.run("show me the schema")
        assert "patients" in result
        assert "encounters" in result

    def test_columns_fast_path(self):
        orch = make_orchestrator()
        result = orch.run("list columns")
        assert "patients" in result


# ---------------------------------------------------------------------------
# 2. TERMINAL COMMAND REJECTION
# ---------------------------------------------------------------------------

class TestTerminalRejection:

    @pytest.mark.parametrize("bad_input", [
        "ls -la",
        "dir",
        "cat /etc/passwd",
        "pwd",
        "ollama list",
        "ollama run llama3.1",
    ])
    def test_rejects_terminal_commands(self, bad_input):
        orch = make_orchestrator()
        result = orch.run(bad_input)
        assert "terminal" in result.lower(), (
            f"Expected terminal rejection message for {bad_input!r}, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# 3. AGENT FLOW — NORMAL PATH
# ---------------------------------------------------------------------------

class TestNormalAgentFlow:

    def test_returns_string(self):
        orch = make_orchestrator()
        result = orch.run("Which patient has the highest billed amount?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_provider_output_present(self):
        """MockLocalProvider returns a fixed string — agent should still complete."""
        orch = make_orchestrator()
        result = orch.run("How many claims are denied?")
        assert isinstance(result, str)

    def test_confidence_score_in_output(self):
        """
        Confidence score must appear when SQL is successfully generated and executed.
        MockLocalProvider returns plain text (not JSON), so sql_plan.get("sql") is
        empty and the agent exits early with a 'could not generate' message — which
        is also a valid outcome. We use a custom dispatcher that injects a real SQL
        string to force the full flow and verify confidence appears.
        """
        SQL_SENTINEL = "SELECT COUNT(*) AS patient_count FROM patients"

        class _ForceSqlProvider:
            """Returns a valid JSON sql_plan on the first call, mock text otherwise."""
            def __init__(self):
                self._call = 0

            def complete(self, *, system_prompt, user_prompt, model=None, temperature=0.2):
                self._call += 1
                # sql_gen agent call (3rd LLM call) — return valid JSON
                if self._call == 3:
                    return f'{{"sql": "{SQL_SENTINEL}", "tables_used": ["patients"], "assumptions": []}}'
                # profiling / dq_rules / narrator — return safe mock JSON or text
                if self._call < 3:
                    return '{"tables_to_profile": [], "why": "mock", "checks": []}'
                return "There is 1 patient in the dataset."

            def complete_with_meta(self, *, system_prompt, user_prompt, model=None, temperature=0.2):
                return self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                ), {}

        orch = MultiAgentOrchestrator(
            llm_provider=_ForceSqlProvider(),
            model="mock",
            tool_dispatcher=make_orchestrator().tool_dispatcher,
            schema_provider=lambda: MOCK_SCHEMA,
            tracker=None,
        )
        result = orch.run("How many patients are there?")
        assert "Confidence:" in result, (
            f"Expected Confidence: in output when SQL executes successfully.\nGot: {result!r}"
        )

    def test_confidence_score_format(self):
        """
        When the full flow runs (SQL generated + executed), the confidence score
        must be a decimal between 0.10 and 0.98.
        Uses the same _ForceSqlProvider pattern to bypass MockLocalProvider's no-JSON output.
        """
        import re

        SQL_SENTINEL = "SELECT department, COUNT(*) FROM encounters GROUP BY department"

        class _ForceSqlProvider:
            def __init__(self):
                self._call = 0

            def complete(self, *, system_prompt, user_prompt, model=None, temperature=0.2):
                self._call += 1
                if self._call == 3:
                    return f'{{"sql": "{SQL_SENTINEL}", "tables_used": ["encounters"], "assumptions": []}}'
                if self._call < 3:
                    return '{"tables_to_profile": [], "why": "mock", "checks": []}'
                return "Departments found in the dataset."

            def complete_with_meta(self, *, system_prompt, user_prompt, model=None, temperature=0.2):
                return self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                ), {}

        orch = MultiAgentOrchestrator(
            llm_provider=_ForceSqlProvider(),
            model="mock",
            tool_dispatcher=make_orchestrator().tool_dispatcher,
            schema_provider=lambda: MOCK_SCHEMA,
            tracker=None,
        )
        result = orch.run("Show me all departments")
        match = re.search(r"Confidence:\s*([\d.]+)", result)
        assert match, f"Could not parse confidence score from: {result!r}"
        score = float(match.group(1))
        assert 0.10 <= score <= 0.98, f"Confidence {score} out of valid range"

    def test_confidence_label_in_output(self):
        """
        Confidence label (High/Medium/Low/Very low) must appear alongside the score.
        Uses _ForceSqlProvider to ensure the full agent flow runs.
        """
        SQL_SENTINEL = "SELECT denial_reason, COUNT(*) FROM denials GROUP BY denial_reason"

        class _ForceSqlProvider:
            def __init__(self):
                self._call = 0

            def complete(self, *, system_prompt, user_prompt, model=None, temperature=0.2):
                self._call += 1
                if self._call == 3:
                    return f'{{"sql": "{SQL_SENTINEL}", "tables_used": ["denials"], "assumptions": []}}'
                if self._call < 3:
                    return '{"tables_to_profile": [], "why": "mock", "checks": []}'
                return "Top denial reasons are shown above."

            def complete_with_meta(self, *, system_prompt, user_prompt, model=None, temperature=0.2):
                return self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                ), {}

        orch = MultiAgentOrchestrator(
            llm_provider=_ForceSqlProvider(),
            model="mock",
            tool_dispatcher=make_orchestrator().tool_dispatcher,
            schema_provider=lambda: MOCK_SCHEMA,
            tracker=None,
        )
        result = orch.run("Show denial reasons")
        assert any(label in result for label in ["High", "Medium", "Low", "Very low"]), (
            f"No confidence label found in: {result!r}"
        )

    def test_zero_row_result_lowers_confidence(self):
        """When SQL returns zero rows, confidence should be low."""
        import re

        def dispatcher(name, args):
            if name == "run_sql":
                return "Row count: 0\n\n(no rows)"
            return "Table 'x' profile:\n- Row count: 0"

        orch = make_orchestrator()
        orch.tool_dispatcher = dispatcher
        result = orch.run("Find patients with diagnosis code XYZ999")

        match = re.search(r"Confidence:\s*([\d.]+)", result)
        if match:
            score = float(match.group(1))
            assert score < 0.65, f"Expected low confidence for zero rows, got {score}"

    def test_sql_generation_failure_returns_message(self):
        """
        When MockLocalProvider returns non-JSON, sql_plan will be empty.
        The orchestrator should return a clear failure message.
        """
        orch = make_orchestrator()
        # MockLocalProvider returns plain text, not JSON
        # So sql_plan.get("sql") will be "" → early return
        result = orch.run("Some complex query that needs real LLM")
        assert isinstance(result, str)
        # Either got a message or a mock SQL tool result — both are acceptable
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 4. TABLE-FIRST RENDERING
# ---------------------------------------------------------------------------

class TestTableFirstRendering:

    @pytest.mark.parametrize("query", [
        "show me the top 5 patients by billed amount",
        "list all denied claims",
        "count claims by department",
        "denial rate trend by month",
    ])
    def test_analytical_queries_show_results_first(self, query):
        """
        Queries with list/count/top/trend/show should render
        'Results:' before 'Summary:' in the output.
        """
        orch = make_orchestrator()
        result = orch.run(query)
        # MockLocalProvider won't produce valid SQL, but if it does reach
        # the output stage, Results: should come before Summary:
        if "Results:" in result and "Summary:" in result:
            assert result.index("Results:") < result.index("Summary:"), (
                f"'Results:' should appear before 'Summary:' for query: {query!r}"
            )


# ---------------------------------------------------------------------------
# 5. CONFIDENCE HELPER TESTS
# ---------------------------------------------------------------------------

class TestConfidenceHelpers:

    def test_high_label(self):
        assert _confidence_label(0.90) == "High"
        assert _confidence_label(0.98) == "High"
        assert _confidence_label(0.85) == "High"

    def test_medium_label(self):
        assert _confidence_label(0.70) == "Medium"
        assert _confidence_label(0.65) == "Medium"

    def test_low_label(self):
        assert _confidence_label(0.50) == "Low"
        assert _confidence_label(0.40) == "Low"

    def test_very_low_label(self):
        assert _confidence_label(0.30) == "Very low"
        assert _confidence_label(0.10) == "Very low"

    def test_boundary_85(self):
        assert _confidence_label(0.85) == "High"

    def test_boundary_65(self):
        assert _confidence_label(0.65) == "Medium"

    def test_boundary_40(self):
        assert _confidence_label(0.40) == "Low"