# pages/eval_dashboard.py
"""
Evaluation Dashboard — Agent Run Analytics

Reads from logs/raw/eval_log.jsonl and visualises:
  - Run summary (total runs, success rate, avg latency, total tokens)
  - Latency trend over time (per run)
  - Per-agent latency breakdown (profiling, dq_rules, sql_gen, narrator)
  - Token usage over time (tokens_in vs tokens_out)
  - Error rate by agent
  - Provider comparison (OpenAI vs Ollama vs errors)
  - Tool call frequency

Accessible from Streamlit multipage:
  - Place this file in pages/ folder
  - Streamlit will auto-discover it as "Eval Dashboard" page

Research value:
  This page provides direct empirical evidence for the dissertation
  evaluation chapter — latency, token usage, error rates, and
  provider comparisons are all observable system behaviours that
  support qualitative findings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Evaluation Dashboard — Agent Run Analytics",
    layout="wide",
)

st.title("Evaluation Dashboard")
st.caption("Agent run analytics from logs/raw/eval_log.jsonl — research evidence for dissertation.")

# ---------------------------------------------------------------------------
# Load JSONL log
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "raw" / "eval_log.jsonl"

# Allow override via sidebar for demo purposes
with st.sidebar:
    st.header("Settings")
    custom_path = st.text_input(
        "Log file path (optional override)",
        value="",
        placeholder="logs/raw/eval_log.jsonl",
    )
    if custom_path.strip():
        LOG_PATH = Path(custom_path.strip())

    st.markdown("---")
    st.caption(f"Reading from:\n`{LOG_PATH}`")


@st.cache_data(show_spinner=False)
def load_eval_log(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse eval_log.jsonl into two DataFrames:
      - runs_df  : one row per run (summary-level)
      - events_df: one row per event (LLM call or tool call)
    """
    runs = []
    events = []

    if not Path(path).exists():
        return pd.DataFrame(), pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue

            run_id = run.get("run_id", "")
            ts_min = None

            for ev in run.get("events", []):
                ts = ev.get("ts")
                if ts and ts_min is None:
                    ts_min = ts
                events.append({
                    "run_id":        run_id,
                    "provider":      run.get("provider", "unknown"),
                    "model":         run.get("model", "unknown"),
                    "event_type":    ev.get("type", ""),
                    "agent":         ev.get("agent", ev.get("tool", "")),
                    "latency_s":     ev.get("latency_s", 0.0),
                    "tokens_in":     ev.get("tokens_in"),
                    "tokens_out":    ev.get("tokens_out"),
                    "status":        ev.get("status", "ok"),
                    "error":         ev.get("error"),
                    "ts":            ts,
                })

            runs.append({
                "run_id":          run_id,
                "prompt_version":  run.get("prompt_version", "v1"),
                "provider":        run.get("provider", "unknown"),
                "model":           run.get("model", "unknown"),
                "total_latency_s": run.get("total_latency_s", 0.0),
                "llm_calls":       run.get("llm_calls", 0),
                "tool_calls":      run.get("tool_calls", 0),
                "tokens_in":       run.get("tokens_in"),
                "tokens_out":      run.get("tokens_out"),
                "cost_usd_est":    run.get("cost_usd_est"),
                "ts":              ts_min,
                "has_error":       any(
                    e.get("status") == "error"
                    for e in run.get("events", [])
                ),
            })

    runs_df = pd.DataFrame(runs)
    events_df = pd.DataFrame(events)

    # Convert timestamps to datetime
    for df in [runs_df, events_df]:
        if "ts" in df.columns and not df.empty:
            df["dt"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")

    return runs_df, events_df


# Load data
if not LOG_PATH.exists():
    st.warning(
        f"Log file not found at `{LOG_PATH}`. "
        "Run the app and ask the agent some questions first to generate log data."
    )
    st.stop()

with st.spinner("Loading evaluation logs..."):
    runs_df, events_df = load_eval_log(str(LOG_PATH))

if runs_df.empty:
    st.warning("Log file exists but contains no parseable runs.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.subheader("Filters")

    providers = ["All"] + sorted(runs_df["provider"].dropna().unique().tolist())
    selected_provider = st.selectbox("Provider", providers)

    models = ["All"] + sorted(runs_df["model"].dropna().unique().tolist())
    selected_model = st.selectbox("Model", models)

    show_errors_only = st.checkbox("Show error runs only", value=False)

# Apply filters
filtered = runs_df.copy()
if selected_provider != "All":
    filtered = filtered[filtered["provider"] == selected_provider]
if selected_model != "All":
    filtered = filtered[filtered["model"] == selected_model]
if show_errors_only:
    filtered = filtered[filtered["has_error"] == True]

filtered_events = events_df[events_df["run_id"].isin(filtered["run_id"])]

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.markdown("## Summary")

total_runs    = len(filtered)
error_runs    = int(filtered["has_error"].sum())
success_runs  = total_runs - error_runs
success_rate  = (success_runs / total_runs * 100) if total_runs > 0 else 0
avg_latency   = filtered["total_latency_s"].mean() if not filtered.empty else 0
total_tok_in  = filtered["tokens_in"].dropna().sum()
total_tok_out = filtered["tokens_out"].dropna().sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total runs",    total_runs)
col2.metric("Success rate",  f"{success_rate:.1f}%")
col3.metric("Avg latency",   f"{avg_latency:.2f}s")
col4.metric("Total tokens in",  f"{int(total_tok_in):,}" if total_tok_in else "N/A")
col5.metric("Total tokens out", f"{int(total_tok_out):,}" if total_tok_out else "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Latency trend
# ---------------------------------------------------------------------------

st.markdown("## Latency per Run")
st.caption("Total end-to-end latency for each agent run. Ollama runs are much slower than OpenAI.")

latency_df = filtered[["dt", "total_latency_s", "provider", "model"]].dropna(subset=["total_latency_s"])
if not latency_df.empty:
    latency_sorted = latency_df.sort_values("dt").reset_index(drop=True)
    latency_sorted.index = latency_sorted.index + 1
    st.line_chart(
        latency_sorted[["total_latency_s"]].rename(columns={"total_latency_s": "latency (s)"}),
        use_container_width=True,
    )
else:
    st.info("No latency data available for the selected filters.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-agent latency breakdown
# ---------------------------------------------------------------------------

st.markdown("## Per-Agent Latency Breakdown")
st.caption("Average latency per agent across all selected runs. Narrator is often slowest.")

llm_events = filtered_events[filtered_events["event_type"] == "llm_call"]
if not llm_events.empty:
    agent_latency = (
        llm_events.groupby("agent")["latency_s"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_latency_s", "count": "calls"})
        .sort_values("avg_latency_s", ascending=False)
    )
    agent_latency["avg_latency_s"] = agent_latency["avg_latency_s"].round(3)
    agent_latency["min"] = agent_latency["min"].round(3)
    agent_latency["max"] = agent_latency["max"].round(3)

    st.bar_chart(
        agent_latency.set_index("agent")[["avg_latency_s"]],
        use_container_width=True,
    )
    st.dataframe(agent_latency, use_container_width=True, hide_index=True)
else:
    st.info("No LLM call events available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

st.markdown("## Token Usage Over Time")
st.caption("tokens_in = prompt tokens sent to LLM. tokens_out = completion tokens returned.")

token_df = filtered[["dt", "tokens_in", "tokens_out"]].dropna(subset=["tokens_in", "tokens_out"])
if not token_df.empty:
    token_sorted = token_df.sort_values("dt").reset_index(drop=True)
    token_sorted.index = token_sorted.index + 1
    st.line_chart(
        token_sorted[["tokens_in", "tokens_out"]],
        use_container_width=True,
    )
    avg_in  = token_sorted["tokens_in"].mean()
    avg_out = token_sorted["tokens_out"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg tokens in per run",  f"{avg_in:.0f}")
    c2.metric("Avg tokens out per run", f"{avg_out:.0f}")
    c3.metric("Avg in/out ratio",       f"{(avg_in/avg_out):.1f}x" if avg_out > 0 else "N/A")
else:
    st.info("No token data available. Token tracking requires OpenAI provider.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Error rate by agent
# ---------------------------------------------------------------------------

st.markdown("## Error Rate by Agent")
st.caption("Percentage of LLM calls that returned an error, grouped by agent.")

if not llm_events.empty:
    error_by_agent = (
        llm_events.groupby("agent")
        .apply(lambda g: pd.Series({
            "total_calls":  len(g),
            "error_calls":  (g["status"] == "error").sum(),
            "error_rate_%": round((g["status"] == "error").mean() * 100, 1),
        }))
        .reset_index()
        .sort_values("error_rate_%", ascending=False)
    )
    st.dataframe(error_by_agent, use_container_width=True, hide_index=True)

    if error_by_agent["error_calls"].sum() > 0:
        st.bar_chart(
            error_by_agent.set_index("agent")[["error_rate_%"]],
            use_container_width=True,
        )
    else:
        st.success("No errors recorded across all agent calls in the selected runs.")
else:
    st.info("No LLM call events available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Provider comparison
# ---------------------------------------------------------------------------

st.markdown("## Provider Comparison")
st.caption("Latency and success rate by provider. Useful for OpenAI vs Ollama comparisons.")

if not runs_df.empty:
    provider_stats = (
        runs_df.groupby("provider")
        .agg(
            runs=("run_id", "count"),
            avg_latency_s=("total_latency_s", "mean"),
            min_latency_s=("total_latency_s", "min"),
            max_latency_s=("total_latency_s", "max"),
            error_runs=("has_error", "sum"),
        )
        .reset_index()
    )
    provider_stats["success_rate_%"] = (
        (provider_stats["runs"] - provider_stats["error_runs"])
        / provider_stats["runs"] * 100
    ).round(1)
    provider_stats["avg_latency_s"] = provider_stats["avg_latency_s"].round(2)

    st.dataframe(provider_stats, use_container_width=True, hide_index=True)

    st.bar_chart(
        provider_stats.set_index("provider")[["avg_latency_s"]],
        use_container_width=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Tool call frequency
# ---------------------------------------------------------------------------

st.markdown("## Tool Call Frequency")
st.caption("Which tools the agent calls most often.")

tool_events = filtered_events[filtered_events["event_type"] == "tool_call"]
if not tool_events.empty:
    tool_freq = (
        tool_events.groupby("agent")
        .agg(
            total_calls=("agent", "count"),
            avg_latency_s=("latency_s", "mean"),
            error_calls=("status", lambda x: (x == "error").sum()),
        )
        .reset_index()
        .rename(columns={"agent": "tool"})
        .sort_values("total_calls", ascending=False)
    )
    tool_freq["avg_latency_s"] = tool_freq["avg_latency_s"].round(4)

    st.bar_chart(
        tool_freq.set_index("tool")[["total_calls"]],
        use_container_width=True,
    )
    st.dataframe(tool_freq, use_container_width=True, hide_index=True)
else:
    st.info("No tool call events in the selected runs.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Raw log table
# ---------------------------------------------------------------------------

st.markdown("## Raw Run Log")
with st.expander("View raw run data (all columns)", expanded=False):
    display_cols = [
        "run_id", "provider", "model", "total_latency_s",
        "llm_calls", "tool_calls", "tokens_in", "tokens_out",
        "has_error", "prompt_version", "dt",
    ]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available], use_container_width=True, hide_index=True)

    csv = filtered[available].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered runs as CSV",
        data=csv,
        file_name="eval_runs_filtered.csv",
        mime="text/csv",
    )
