# agent/errors.py
"""
Structured error taxonomy for the CA Hospital Data-Quality Copilot.

Replaces plain error strings throughout the codebase with typed,
machine-readable error codes. Benefits:
  - UI can render different error states differently
  - Logs are machine-parseable (grep/filter by error type)
  - Dissertation: demonstrates governance-aware error handling

Usage:
    from agent.errors import AgentError, ErrorType

    raise AgentError(ErrorType.SQL_BLOCKED, "INSERT is not permitted.")

    # Or build a plain string response for the UI:
    return AgentError.format(ErrorType.LOW_CONFIDENCE, "Result has too many nulls.")
"""

from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    """
    Canonical error types for the ETL agent system.

    Each value is the machine-readable code written to logs and
    returned in API error responses.
    """

    # SQL layer errors
    SQL_BLOCKED       = "SQL_BLOCKED"       # write op / multi-statement attempted
    SQL_FAILED        = "SQL_FAILED"        # DuckDB execution error
    SQL_EMPTY         = "SQL_EMPTY"         # empty query submitted
    SCHEMA_MISMATCH   = "SCHEMA_MISMATCH"   # column / table not found in schema
    INVALID_JOIN      = "INVALID_JOIN"      # bad join pattern detected by guardrail

    # LLM / agent errors
    LLM_TIMEOUT       = "LLM_TIMEOUT"       # provider did not respond in time
    LLM_NO_SQL        = "LLM_NO_SQL"        # agent could not generate a SQL query
    LLM_PARSE_ERROR   = "LLM_PARSE_ERROR"   # LLM response was not valid JSON

    # Data quality signals
    LOW_CONFIDENCE    = "LOW_CONFIDENCE"    # confidence score below threshold
    ZERO_ROWS         = "ZERO_ROWS"         # query returned no results
    NULL_HEAVY        = "NULL_HEAVY"        # result contains excessive nulls

    # Tool errors
    TOOL_FAILED       = "TOOL_FAILED"       # a named tool raised an exception
    TOOL_UNKNOWN      = "TOOL_UNKNOWN"      # unknown tool name dispatched

    # Governance / compliance
    GOVERNANCE_BLOCK  = "GOVERNANCE_BLOCK"  # action blocked by governance policy
    UNSAFE_INPUT      = "UNSAFE_INPUT"      # user input failed safety check


# Human-readable descriptions for each error type
_ERROR_DESCRIPTIONS: dict[ErrorType, str] = {
    ErrorType.SQL_BLOCKED:      "SQL operation blocked — only SELECT queries are permitted.",
    ErrorType.SQL_FAILED:       "SQL execution failed in DuckDB.",
    ErrorType.SQL_EMPTY:        "No SQL query was provided.",
    ErrorType.SCHEMA_MISMATCH:  "Referenced table or column does not exist in the schema.",
    ErrorType.INVALID_JOIN:     "Invalid join pattern detected — check join key guidance.",
    ErrorType.LLM_TIMEOUT:      "LLM provider did not respond within the timeout window.",
    ErrorType.LLM_NO_SQL:       "The agent could not generate a SQL query for this request.",
    ErrorType.LLM_PARSE_ERROR:  "LLM response could not be parsed as valid JSON.",
    ErrorType.LOW_CONFIDENCE:   "Result confidence is low — verify with additional checks.",
    ErrorType.ZERO_ROWS:        "Query returned zero rows — result may be incomplete.",
    ErrorType.NULL_HEAVY:       "Result contains a high proportion of null values.",
    ErrorType.TOOL_FAILED:      "A tool call raised an exception during execution.",
    ErrorType.TOOL_UNKNOWN:     "Unknown tool name — check tool registration.",
    ErrorType.GOVERNANCE_BLOCK: "Action blocked by governance policy.",
    ErrorType.UNSAFE_INPUT:     "Input failed safety validation and was rejected.",
}


class AgentError(Exception):
    """
    Typed exception for agent-layer errors.

    Carries an ErrorType code so callers can handle specific
    error categories differently (UI rendering, logging, alerting).

    Example:
        try:
            run_sql(query)
        except AgentError as e:
            if e.error_type == ErrorType.SQL_BLOCKED:
                log_governance_event(e)
    """

    def __init__(self, error_type: ErrorType, detail: str = "") -> None:
        self.error_type = error_type
        self.detail = detail or _ERROR_DESCRIPTIONS.get(error_type, "")
        super().__init__(f"[{error_type.value}] {self.detail}")

    @property
    def code(self) -> str:
        """Machine-readable error code string, e.g. 'SQL_BLOCKED'."""
        return self.error_type.value

    def to_dict(self) -> dict:
        """Serialise to a dict suitable for API error responses and log entries."""
        return {
            "error": True,
            "error_type": self.error_type.value,
            "detail": self.detail,
        }

    @staticmethod
    def format(error_type: ErrorType, detail: str = "") -> str:
        """
        Build a plain-string error message prefixed with the error code.
        Use this when you need a string (e.g. for UI output) rather than
        raising an exception.

        Example:
            return AgentError.format(ErrorType.SQL_BLOCKED, "INSERT not allowed.")
            # → "[SQL_BLOCKED] INSERT not allowed."
        """
        msg = detail or _ERROR_DESCRIPTIONS.get(error_type, "")
        return f"[{error_type.value}] {msg}"

    @staticmethod
    def description(error_type: ErrorType) -> str:
        """Return the human-readable description for an error type."""
        return _ERROR_DESCRIPTIONS.get(error_type, error_type.value)
