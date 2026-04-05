# agent/multi_agent.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from llm.providers import LLMProvider


PROFILING_AGENT_PROMPT = """
You are the Data Profiling Agent.
Goal: Decide which tables should be profiled to answer the user question.

Return STRICT JSON:
{
  "tables_to_profile": ["table1", "table2"],
  "why": "short reason"
}

Rules:
- Return ONLY JSON (no markdown, no commentary).
- Use only tables that exist in the provided schema.
""".strip()


DQ_RULES_AGENT_PROMPT = """
You are the Data Quality Rules Agent.
Goal: Propose deterministic data-quality checks to run for the user's request.

Return STRICT JSON:
{
  "checks": [
    {"name": "missing_values", "table": "claims_and_billing", "columns": ["total_billed"]},
    {"name": "negative_amounts", "table": "claims_and_billing", "columns": ["total_paid"]}
  ],
  "why": "short reason"
}

Rules:
- Return ONLY JSON.
- Use only tables/columns that exist in the provided schema.
""".strip()


SQL_GENERATOR_AGENT_PROMPT = """
You are the SQL Generator Agent.
Goal: Write a SAFE, READ-ONLY DuckDB SQL query to answer the user's question.

Return STRICT JSON:
{
  "sql": "SELECT ...",
  "tables_used": ["table1", "table2"],
  "assumptions": ["..."]
}

Rules:
- Return ONLY JSON.
- SQL must be SELECT-only (no INSERT/UPDATE/DELETE/CREATE/DROP).
- Prefer explicit joins on known keys when needed.
""".strip()


NARRATOR_AGENT_PROMPT = """
You are the Narrator Agent.
Goal: Explain results in clear business language for healthcare analytics and data quality.

Guidelines:
- Be concise and specific.
- Summarize what the data shows.
- If query returned zero rows, explain likely reasons and next checks.
""".strip()


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    sql_result: str,
    sql: str,
    tool_calls_made: int,
    profiles_used: int,
) -> float:
    """
    Compute a real confidence score (0.0–1.0) from runtime signals.

    Signals (each contributes a penalty or boost):

    1. result_signal   — did we get rows back? How many?
       - Zero rows         : heavy penalty (-0.40)
       - 1–4 rows          : mild penalty  (-0.10)  may be intentional top-N
       - 5+ rows           : no penalty

    2. error_signal    — did the SQL execution fail or get blocked?
       - "SQL failed"      : heavy penalty (-0.50)
       - "SQL blocked"     : heavy penalty (-0.50)

    3. null_signal     — is the result full of NULL / None values?
       - null fraction > 30% of result tokens: penalty (-0.15)

    4. complexity_signal — how many JOINs / subqueries are in the SQL?
       - Each join adds a small uncertainty penalty (-0.05, max -0.20)
       - Reasoning: more joins = more chance of bad join key

    5. tool_signal     — did the agent use profiling to ground its answer?
       - profiles_used > 0 : small boost (+0.05)
       - tool_calls_made > 2: small boost (+0.05) agent did thorough work

    Base score: 0.90
    Final score clamped to [0.10, 0.98]
    """
    score = 0.90

    result = (sql_result or "").strip()
    sql_lower = (sql or "").lower()

    # 1. Result row signal
    if "Row count: 0" in result or "(no rows)" in result:
        score -= 0.40
    else:
        row_match = re.search(r"Row count:\s*(\d+)", result)
        if row_match:
            row_count = int(row_match.group(1))
            if row_count < 5:
                score -= 0.10
        # No rows info found at all — mild uncertainty
        else:
            score -= 0.05

    # 2. Error signal
    if result.startswith("SQL failed") or result.startswith("SQL blocked"):
        score -= 0.50

    # 3. Null density signal
    # Count how many times "None" or "null" appear in the result string
    total_tokens = max(len(result.split()), 1)
    null_tokens = result.lower().count("none") + result.lower().count("null")
    null_fraction = null_tokens / total_tokens
    if null_fraction > 0.30:
        score -= 0.15

    # 4. SQL complexity signal (join penalty)
    join_count = len(re.findall(r"\bjoin\b", sql_lower))
    subquery_count = len(re.findall(r"\bselect\b", sql_lower)) - 1  # outer SELECT
    complexity_penalty = min((join_count + max(subquery_count, 0)) * 0.05, 0.20)
    score -= complexity_penalty

    # 5. Tool grounding boost
    if profiles_used > 0:
        score += 0.05
    if tool_calls_made > 2:
        score += 0.05

    return round(max(0.10, min(0.98, score)), 2)


def _confidence_label(score: float) -> str:
    """Human-readable label for the confidence score."""
    if score >= 0.85:
        return "High"
    if score >= 0.65:
        return "Medium"
    if score >= 0.40:
        return "Low"
    return "Very low"


@dataclass
class MultiAgentOrchestrator:
    llm_provider: LLMProvider
    model: str
    tool_dispatcher: Callable[[str, Dict[str, Any]], str]
    schema_provider: Callable[[], str]
    tracker: Any | None = None

    def _safe_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}

    def _call_llm_json(self, system_prompt: str, user_prompt: str, agent_name: str) -> Dict[str, Any]:
        import time
        from agent.eval_tracker import _sha1

        t0 = time.time()
        prompt_hash = _sha1(system_prompt + "\n" + user_prompt)

        try:
            if hasattr(self.llm_provider, "complete_with_meta"):
                text, meta = self.llm_provider.complete_with_meta(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model,
                    temperature=0.2,
                )
            else:
                text = self.llm_provider.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model,
                    temperature=0.2,
                )
                meta = {}

            out = self._safe_json(text)

            if self.tracker:
                self.tracker.record_llm(
                    agent=agent_name,
                    latency_s=time.time() - t0,
                    prompt_hash=prompt_hash,
                    tokens_in=meta.get("tokens_in"),
                    tokens_out=meta.get("tokens_out"),
                    cost_usd_est=None,
                )
            return out

        except Exception as e:
            if self.tracker:
                self.tracker.record_llm(
                    agent=agent_name,
                    latency_s=time.time() - t0,
                    prompt_hash=prompt_hash,
                    status="error",
                    error=str(e),
                )
            return {}

    def _call_llm_text(self, system_prompt: str, user_prompt: str, agent_name: str) -> str:
        import time
        from agent.eval_tracker import _sha1

        t0 = time.time()
        prompt_hash = _sha1(system_prompt + "\n" + user_prompt)

        try:
            if hasattr(self.llm_provider, "complete_with_meta"):
                text, meta = self.llm_provider.complete_with_meta(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model,
                    temperature=0.2,
                )
            else:
                text = self.llm_provider.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model,
                    temperature=0.2,
                )
                meta = {}

            if self.tracker:
                self.tracker.record_llm(
                    agent=agent_name,
                    latency_s=time.time() - t0,
                    prompt_hash=prompt_hash,
                    tokens_in=meta.get("tokens_in"),
                    tokens_out=meta.get("tokens_out"),
                    cost_usd_est=None,
                )
            return (text or "").strip()

        except Exception as e:
            if self.tracker:
                self.tracker.record_llm(
                    agent=agent_name,
                    latency_s=time.time() - t0,
                    prompt_hash=prompt_hash,
                    status="error",
                    error=str(e),
                )
            return ""

    def run(self, user_message: str) -> str:
        import time

        # Track tool calls made in this run for confidence scoring
        tool_calls_made = 0

        def _tool(name: str, args: Dict[str, Any]) -> str:
            nonlocal tool_calls_made
            t0 = time.time()
            try:
                out = self.tool_dispatcher(name, args)
                tool_calls_made += 1
                if self.tracker:
                    self.tracker.record_tool(tool=name, latency_s=time.time() - t0)
                return out
            except Exception as e:
                if self.tracker:
                    self.tracker.record_tool(
                        tool=name,
                        latency_s=time.time() - t0,
                        status="error",
                        error=str(e),
                    )
                raise

        def _should_show_table_first(msg: str) -> bool:
            """Force evidence-first output for analytical prompts."""
            patterns = [
                r"\bby month\b",
                r"\btrend\b",
                r"\btrends\b",
                r"\bcount\b",
                r"\bcounts\b",
                r"\blist\b",
                r"\btop\s+\d+\b",
                r"\btop\b",
                r"\btable format\b",
                r"\bshow\b",
            ]
            return any(re.search(p, msg) for p in patterns)

        msg_raw = (user_message or "").strip()
        msg = msg_raw.lower()

        # ---------- NON-APP COMMANDS ----------
        if re.match(r"^(type|cat|dir|ls|pwd)\b", msg) or msg.startswith("ollama "):
            return "Run that in your terminal, not inside the app chat."

        # ---------- FAST PATHS ----------
        if re.search(r"\b(list|show)\s+tables\b", msg) or msg in {"tables", "list tables", "show tables"}:
            return _tool(
                "run_sql",
                {"query": "SELECT * FROM duckdb_tables() ORDER BY table_name", "max_rows": 200},
            )

        m_profile = re.search(r"\bprofile\s+(?:the\s+)?([a-zA-Z_]\w*)\s+table\b", msg)
        if m_profile:
            table = m_profile.group(1)
            return _tool("profile_table", {"table_name": table})

        if re.search(r"\b(schema|columns|list columns)\b", msg):
            return self.schema_provider()

        # ---------- NORMAL MULTI-AGENT FLOW ----------
        schema = self.schema_provider()

        profiling = self._call_llm_json(
            system_prompt=PROFILING_AGENT_PROMPT,
            user_prompt=f"SCHEMA:\n{schema}\n\nUSER REQUEST:\n{msg_raw}",
            agent_name="profiling",
        )

        tables_to_profile = profiling.get("tables_to_profile", []) or []
        if not isinstance(tables_to_profile, list):
            tables_to_profile = []

        profiles: List[str] = []
        for t in tables_to_profile[:6]:
            try:
                profiles.append(_tool("profile_table", {"table_name": str(t)}))
            except Exception as e:
                profiles.append(f"Profile failed for {t}: {e}")

        dq_rules = self._call_llm_json(
            system_prompt=DQ_RULES_AGENT_PROMPT,
            user_prompt=(
                f"SCHEMA:\n{schema}\n\n"
                f"TABLE PROFILES (may be partial):\n{chr(10).join(profiles)}\n\n"
                f"USER REQUEST:\n{msg_raw}"
            ),
            agent_name="dq_rules",
        )

        sql_plan = self._call_llm_json(
            system_prompt=SQL_GENERATOR_AGENT_PROMPT,
            user_prompt=(
                f"SCHEMA:\n{schema}\n\n"
                f"TABLE PROFILES (may be partial):\n{chr(10).join(profiles)}\n\n"
                f"DQ CHECK IDEAS (optional):\n{json.dumps(dq_rules, indent=2)}\n\n"
                f"USER REQUEST:\n{msg_raw}"
            ),
            agent_name="sql_gen",
        )

        sql = (sql_plan.get("sql") or "").strip()
        if not sql:
            return "I could not generate a SQL query for that request."

        sql_result = _tool("run_sql", {"query": sql, "max_rows": 50})

        # --- REAL confidence score (replaces hardcoded 0.85 / 0.35) ---
        confidence = compute_confidence(
            sql_result=sql_result,
            sql=sql,
            tool_calls_made=tool_calls_made,
            profiles_used=len(profiles),
        )
        label = _confidence_label(confidence)

        narration = self._call_llm_text(
            system_prompt=NARRATOR_AGENT_PROMPT,
            user_prompt=(
                f"USER REQUEST:\n{msg_raw}\n\n"
                f"SQL USED:\n{sql}\n\n"
                f"SQL RESULT (preview):\n{sql_result}\n\n"
                "Explain the result briefly in 2-4 bullets or short paragraphs. "
                "Do not restate the whole table. "
                "If insufficient data, say so and suggest next checks."
            ),
            agent_name="narrator",
        )

        if not narration:
            narration = "Insufficient data. Try refining the question."

        confidence_line = f"Confidence: {confidence:.2f} ({label})"

        if _should_show_table_first(msg):
            return (
                f"Results:\n\n{sql_result}\n\n"
                f"Summary:\n{narration.strip()}\n\n"
                f"{confidence_line}"
            )

        return narration.strip() + f"\n\n{confidence_line}"
