# app.py  (Phase 1 refactor — pure Streamlit wiring)
#
# All DQ logic  → dq_helpers.py
# All analytics → analytics.py
#
# This file only handles:
#   - Page config & session state init
#   - Sidebar controls
#   - Main layout & routing
#   - LLM agent chat interface
#   - Upload & KPI comparison UI
#   - Docs section

import streamlit as st
import pandas as pd
from io import BytesIO

from pydantic import BaseModel, Field, ValidationError
from graphviz import Digraph

from agent.etl_agent import run_agent, get_connection

# Extracted modules (Phase 1 refactor)
from dq_helpers import (
    run_duckdb_data_quality_scan,
    run_duckdb_anomaly_checks,
    compute_dqi_scores,
    run_fhir_validations,
    generate_dq_insights,
    build_pdf_report,
)
from analytics import build_advanced_charts, build_claims_charts


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class ClaimsKpiSnapshot(BaseModel):
    period_label: str
    total_billed: float = Field(..., ge=0)
    total_paid: float = Field(..., ge=0)
    denial_rate: float = Field(..., ge=0, le=1)


# ---------------------------------------------------------------------------
# Cached KPI helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _get_sample_claims_kpi_dict() -> dict:
    """
    Returns plain dict — @st.cache_data cannot pickle Pydantic models.
    Call get_sample_claims_kpi() which wraps this into ClaimsKpiSnapshot.
    """
    with get_connection() as con:
        row = con.execute(
            """
            SELECT
                SUM(billed_amount)  AS total_billed,
                SUM(paid_amount)    AS total_paid,
                AVG(CASE WHEN claim_status = 'Denied' THEN 1.0 ELSE 0.0 END) AS denial_rate
            FROM claims_and_billing;
            """
        ).fetchone()
    return {
        "period_label": "Sample CA Q1-2025",
        "total_billed": float(row[0] or 0),
        "total_paid": float(row[1] or 0),
        "denial_rate": float(row[2] or 0),
    }


def get_sample_claims_kpi() -> ClaimsKpiSnapshot:
    """Wraps the cached dict into a ClaimsKpiSnapshot Pydantic model."""
    d = _get_sample_claims_kpi_dict()
    return ClaimsKpiSnapshot(**d)


def compute_kpi_from_df(df: pd.DataFrame, label: str = "Uploaded Dataset") -> ClaimsKpiSnapshot:
    if df is None or df.empty:
        return ClaimsKpiSnapshot(period_label=label, total_billed=0.0, total_paid=0.0, denial_rate=0.0)

    cols_lower = {c.lower(): c for c in df.columns}
    required = ["billed_amount", "paid_amount", "claim_status"]
    missing = [c for c in required if c not in cols_lower]
    if missing:
        raise ValueError("Uploaded file must contain columns (case-insensitive): " + ", ".join(required))

    billed_col = cols_lower["billed_amount"]
    paid_col = cols_lower["paid_amount"]
    status_col = cols_lower["claim_status"]

    tmp = df[[billed_col, paid_col, status_col]].copy()
    tmp[billed_col] = pd.to_numeric(tmp[billed_col], errors="coerce")
    tmp[paid_col] = pd.to_numeric(tmp[paid_col], errors="coerce")

    total_billed = float(tmp[billed_col].fillna(0).sum())
    total_paid = float(tmp[paid_col].fillna(0).sum())
    total_claims = len(tmp)

    if total_claims > 0:
        denied = tmp[status_col].astype(str).str.strip().str.lower().eq("denied").sum()
        denial_rate = denied / total_claims
    else:
        denial_rate = 0.0

    return ClaimsKpiSnapshot(
        period_label=label,
        total_billed=total_billed,
        total_paid=total_paid,
        denial_rate=denial_rate,
    )


# ---------------------------------------------------------------------------
# Data lineage graph
# ---------------------------------------------------------------------------

def build_lineage_graph() -> Digraph:
    g = Digraph(name="hospital_lineage")
    g.attr(rankdir="LR")
    g.attr("node", shape="box")
    for node_id, label in [
        ("csv", "CSV Files\n(Data/*.csv)"),
        ("etl", "ETL Loader\n(etl/load_duckdb.py)"),
        ("duckdb", "DuckDB Warehouse\n(db/hospital.duckdb)"),
        ("agent", "ETL Agent\n(agent/etl_agent.py)"),
        ("tools", "SQL Tools\n(run_sql, profile_table,\nexplain_field)"),
        ("sql", "SQL Queries\n(SELECT-only)"),
        ("agg", "Aggregations & DQ Checks\n(KPIs, anomalies, profiles)"),
        ("visuals", "Streamlit UI\n(charts, tables, KPIs)"),
    ]:
        g.node(node_id, label)
    for src, dst in [
        ("csv", "etl"), ("etl", "duckdb"), ("duckdb", "sql"),
        ("agent", "tools"), ("tools", "sql"), ("agent", "sql"),
        ("sql", "agg"), ("agg", "visuals"),
    ]:
        g.edge(src, dst)
    g.node("user", "User\n(Analyst / Engineer)", shape="ellipse")
    g.edge("user", "agent")
    g.edge("visuals", "user")
    try:
        with get_connection() as con:
            tables_df = con.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            ).fetch_df()
        for _, row in tables_df.iterrows():
            tbl = row["table_name"]
            g.node(f"tbl_{tbl}", tbl, shape="box")
            g.edge("duckdb", f"tbl_{tbl}")
    except Exception:
        pass
    return g


# ---------------------------------------------------------------------------
# Docs section
# ---------------------------------------------------------------------------

def render_docs_section() -> None:
    st.markdown("---")
    st.markdown("## Documentation & Tutorials")
    tabs = st.tabs([
        "How the ETL Agent Works",
        "System Architecture Diagram",
        "What is DuckDB?",
        "How SQL Tools Are Triggered",
        "Agentic AI Workflow",
    ])

    tabs[0].markdown("""
### How the ETL Agent Works
1. **User enters a natural-language question**
2. **The app sends the question, schema, and history to the ETL Agent**
3. **The agent decides whether it needs to use a tool** (SQL, profiling, field explanation)
4. **If SQL is required**, the agent generates a SELECT query → DuckDB executes it
5. **The agent reads the returned data** and produces a human-readable answer
""")

    tabs[1].markdown("""
### System Architecture Overview
**1. Data Layer** — CSV files loaded into DuckDB via `etl/load_duckdb.py`

**2. Agent Layer** — `agent/etl_agent.py` defines tools and orchestrates the multi-agent flow

**3. Application Layer** — Streamlit UI (`app.py`), DQ helpers (`dq_helpers.py`), analytics (`analytics.py`)

**Flow:** CSV → ETL Loader → DuckDB → ETL Agent (Tools) → SQL → Results → UI
""")

    tabs[2].markdown("""
### What is DuckDB?
DuckDB is an **embedded OLAP database** — runs in-process, no server needed.
- Extremely fast for columnar analytics
- Can query CSV and Parquet directly
- Supports full SQL: GROUP BY, JOINs, window functions
- All agent SQL queries are SELECT-only via DuckDB
""")

    tabs[3].markdown("""
### How SQL Tools Are Triggered
The ETL Agent has a restricted toolset:
- `run_sql(query)` — execute a safe SELECT on DuckDB
- `profile_table(name)` — row count, column count, null stats
- `explain_field(table, column)` — schema + description

Forbidden: DELETE, INSERT, UPDATE, CREATE, DROP
""")

    tabs[4].markdown("""
### Agentic AI Workflow
1. **Intent Detection** — understand whether to run SQL, profile data, explore anomalies
2. **Planning** — choose which tools to call and in what order
3. **Tool Execution Loop** — call tool → get result → decide next step
4. **Reasoning** — analyze DataFrames, produce insights and recommendations
5. **Conversation Memory** — resolve follow-up references like "same patient as before"
""")


# ---------------------------------------------------------------------------
# Page config & session state
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CA Hospital Data-Quality Copilot (LLM + Agentic ETL Assistant)",
    layout="wide",
)

for key in [
    "history", "chat_messages", "dq_results", "anomaly_results",
    "dqi_results", "dqi_overall", "dq_insights", "fhir_results",
    "kpi_compare", "user_claims_df", "user_claims_kpi",
    "user_query", "clear_query",
]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ("history", "chat_messages") else (False if key == "clear_query" else (None if key != "user_query" else ""))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Tools")

    st.subheader("LLM Provider")
    provider_choice = st.selectbox("Provider", ["openai", "ollama", "mock"], index=0)
    st.session_state.llm_provider = provider_choice

    if provider_choice == "openai":
        st.session_state.openai_model = st.text_input("OpenAI model", value=st.session_state.get("openai_model", "gpt-4o-mini"))
        st.session_state.local_model = st.session_state.get("local_model", "llama3.1")
    else:
        st.session_state.local_model = st.text_input("Local model (Ollama)", value=st.session_state.get("local_model", "llama3.1"))
        st.session_state.openai_model = st.session_state.get("openai_model", "gpt-4o-mini")

    st.markdown("---")
    dq_btn       = st.button("Run Data-Quality Scan",          use_container_width=True)
    anomaly_btn  = st.button("Run Built-in Anomaly Checks",    use_container_width=True)
    dqi_btn      = st.button("Compute DQI Score",              use_container_width=True)
    insights_btn = st.button("Generate AI DQ Insights",        use_container_width=True)
    fhir_btn     = st.button("Run HL7 / FHIR Validators",      use_container_width=True)
    analytics_btn= st.button("Run Advanced Analytics Dashboard",use_container_width=True)

    st.markdown("---")
    if st.button("Run Claims/Billing Charts", use_container_width=True):
        build_claims_charts()

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.chat_messages = []
        st.session_state.user_query = ""
        st.success("Conversation cleared.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("CA Hospital Data-Quality Copilot (LLM + Agentic ETL Assistant)")
st.caption("LLM + DuckDB + Streamlit — Natural language agent + deterministic data-quality checks.")

st.markdown("""
### LLM Agent – Ask about your Hospital Data
- Ask about tables (e.g., *Profile the claims_and_billing table*)
- Request ETL / DQ SQL (e.g., *Give me SQL to find overpaid claims*)
- Ask business questions (e.g., *Which insurance paid the most?*)
""")

if st.session_state.clear_query:
    st.session_state.user_query = ""
    st.session_state.clear_query = False

with st.expander("View Data Lineage (CSV → DuckDB → Agent → SQL → Visuals)", expanded=False):
    st.graphviz_chart(build_lineage_graph(), use_container_width=True)

user_input = st.text_area(
    "Your question / request:",
    key="user_query",
    placeholder="e.g., Profile the claims_and_billing table and suggest checks for missing or inconsistent values.",
)

if st.button("Run Agent", type="primary"):
    question = st.session_state.user_query.strip()
    if question:
        chat_history = st.session_state.chat_messages[-4:]
        with st.spinner("Consulting the ETL Agent..."):
            answer = run_agent(
                question,
                chat_history=chat_history,
                provider_name=st.session_state.get("llm_provider", "openai"),
                openai_model=st.session_state.get("openai_model"),
                local_model=st.session_state.get("local_model"),
            )
        st.session_state.history.append({"question": question, "answer": answer})
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.clear_query = True

if st.session_state.history:
    st.markdown("## Conversation")
    for idx, turn in enumerate(st.session_state.history, start=1):
        preview = turn["question"][:80] + ("..." if len(turn["question"]) > 80 else "")
        with st.expander(f"Turn {idx}: {preview}", expanded=(idx == len(st.session_state.history))):
            st.markdown(f"**You:** {turn['question']}")
            st.markdown(f"**Agent:**\n\n{turn['answer']}")


# ---------------------------------------------------------------------------
# Sidebar action handlers
# ---------------------------------------------------------------------------

if dq_btn:
    st.session_state.dq_results = run_duckdb_data_quality_scan()

if anomaly_btn:
    st.session_state.anomaly_results = run_duckdb_anomaly_checks()

if analytics_btn:
    with st.spinner("Building advanced analytics charts from DuckDB..."):
        build_advanced_charts()

if dqi_btn:
    with st.spinner("Computing DQI scores from DuckDB..."):
        dqi_df = compute_dqi_scores(max_rows_for_anomalies=500)
        st.session_state.dqi_results = dqi_df
        if not dqi_df.empty:
            total_rows = dqi_df["row_count"].sum()
            overall = (
                (dqi_df["row_count"] * dqi_df["dqi_score"]).sum() / total_rows
                if total_rows > 0
                else float(dqi_df["dqi_score"].mean())
            )
            st.session_state.dqi_overall = round(overall, 1)
        else:
            st.session_state.dqi_overall = None

if insights_btn:
    with st.spinner("Asking the AI agent to analyze DQI and anomalies..."):
        st.session_state.dq_insights = generate_dq_insights()

if fhir_btn:
    with st.spinner("Running HL7 / FHIR-style healthcare validations..."):
        st.session_state.fhir_results = run_fhir_validations(max_rows=500)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if st.session_state.dq_results is not None:
    st.markdown("## Data-Quality Scan (Row & Column Counts)")
    st.dataframe(st.session_state.dq_results, use_container_width=True)

if st.session_state.dqi_results is not None:
    st.markdown("## Automated Data-Quality Score (DQI)")
    overall = st.session_state.dqi_overall
    if overall is not None:
        if overall >= 90:   status = "Excellent"
        elif overall >= 75: status = "Good"
        elif overall >= 60: status = "Fair"
        else:               status = "Needs Attention"
        st.markdown(f"**Overall DQI: {overall:.1f} / 100** — {status}")

    dqi_display = st.session_state.dqi_results[[
        "table", "row_count", "column_count", "null_cells",
        "completeness_score", "anomaly_rows", "anomaly_score", "dqi_score", "dqi_bar",
    ]]
    st.markdown(dqi_display.to_html(escape=False, index=False), unsafe_allow_html=True)

if st.session_state.dq_insights:
    st.markdown("## AI Data-Quality Insights")
    st.markdown(st.session_state.dq_insights)
    pdf_bytes = build_pdf_report()
    st.download_button(
        label="Download Full PDF Data-Quality Report",
        data=pdf_bytes,
        file_name="hospital_data_quality_report.pdf",
        mime="application/pdf",
    )

if st.session_state.fhir_results is not None:
    st.markdown("## HL7 / FHIR Validation Checks")
    any_fhir_rows = False
    for name, df in st.session_state.fhir_results.items():
        if df is not None and not df.empty:
            any_fhir_rows = True
            with st.expander(name.replace("_", " ").title(), expanded=False):
                st.dataframe(df.head(500), use_container_width=True)
                st.download_button(
                    label=f"Download all {len(df)} rows as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name}.csv",
                    mime="text/csv",
                )
    if not any_fhir_rows:
        st.success("All HL7 / FHIR-style validators passed on the scanned sample.")

if st.session_state.anomaly_results is not None:
    st.markdown("## Built-in Anomaly Checks (DuckDB/SQL)")
    any_rows = False
    for name, df in st.session_state.anomaly_results.items():
        if df is not None and not df.empty:
            any_rows = True
            with st.expander(name.replace("_", " ").title(), expanded=False):
                st.dataframe(df.head(500), use_container_width=True)
    if not any_rows:
        st.success("No anomalies found by the built-in checks (within the scanned sample).")


# ---------------------------------------------------------------------------
# Upload & KPI comparison
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Upload Your Own Claims / Billing Data")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file with 'billed_amount', 'paid_amount', and 'claim_status' columns.",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
        if user_df.empty:
            st.error("Uploaded file is empty.")
        else:
            cols_lower = {c.lower(): c for c in user_df.columns}
            required = ["billed_amount", "paid_amount", "claim_status"]
            missing = [c for c in required if c not in cols_lower]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.session_state.user_claims_df = user_df
                user_kpi = compute_kpi_from_df(user_df, label="Uploaded Dataset")
                st.session_state.user_claims_kpi = user_kpi
                st.success(f"Uploaded `{uploaded_file.name}` — {len(user_df):,} rows.")
                st.markdown("### Preview")
                st.dataframe(user_df.head(200), use_container_width=True)
                st.markdown("### KPIs from Uploaded Dataset")
                st.write(pd.DataFrame([{
                    "period": user_kpi.period_label,
                    "total_billed": user_kpi.total_billed,
                    "total_paid": user_kpi.total_paid,
                    "denial_rate (%)": round(user_kpi.denial_rate * 100, 2),
                }]))
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")


st.markdown("---")
st.markdown("## Compare KPIs with Your Own System")

with st.expander("Compare sample dataset KPIs with your own numbers"):
    sample_kpi = get_sample_claims_kpi()
    st.markdown("**Sample dataset (from local DuckDB):**")
    st.write(pd.DataFrame([{
        "period": sample_kpi.period_label,
        "total_billed": sample_kpi.total_billed,
        "total_paid": sample_kpi.total_paid,
        "denial_rate (%)": round(sample_kpi.denial_rate * 100, 2),
    }]))

    col1, col2, col3 = st.columns(3)
    with col1:
        user_billed = st.number_input("Your total billed", min_value=0.0, value=0.0, step=1000.0)
    with col2:
        user_paid = st.number_input("Your total paid", min_value=0.0, value=0.0, step=1000.0)
    with col3:
        user_denial_pct = st.number_input("Your denial rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    if st.button("Compare KPIs", key="compare_manual_kpi"):
        try:
            your_kpi = ClaimsKpiSnapshot(
                period_label="Your System Snapshot",
                total_billed=user_billed,
                total_paid=user_paid,
                denial_rate=user_denial_pct / 100.0,
            )
            st.session_state.kpi_compare = pd.DataFrame([
                {"period": sample_kpi.period_label, "total_billed": sample_kpi.total_billed,
                 "total_paid": sample_kpi.total_paid, "denial_rate (%)": round(sample_kpi.denial_rate * 100, 2)},
                {"period": your_kpi.period_label, "total_billed": your_kpi.total_billed,
                 "total_paid": your_kpi.total_paid, "denial_rate (%)": round(your_kpi.denial_rate * 100, 2)},
            ])
        except ValidationError as e:
            st.error(f"Validation error: {e}")

    if st.session_state.kpi_compare is not None:
        st.markdown("### KPI Comparison")
        st.dataframe(st.session_state.kpi_compare, use_container_width=True)


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

render_docs_section()
