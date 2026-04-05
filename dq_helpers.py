# dq_helpers.py
"""
Data-quality helpers for the CA Hospital Data-Quality Copilot.

Extracted from app.py to keep it maintainable.
Contains all DuckDB-backed DQ functions:
  - run_duckdb_data_quality_scan()
  - run_duckdb_anomaly_checks()
  - compute_dqi_scores()
  - run_fhir_validations()
  - generate_dq_insights()
  - build_pdf_report()

Import in app.py:
    from dq_helpers import (
        run_duckdb_data_quality_scan,
        run_duckdb_anomaly_checks,
        compute_dqi_scores,
        run_fhir_validations,
        generate_dq_insights,
        build_pdf_report,
    )
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from io import BytesIO
from typing import Dict

from agent.etl_agent import run_agent, get_connection

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ---------------------------------------------------------------------------
# DQ Scan
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_duckdb_data_quality_scan() -> pd.DataFrame:
    """Basic row/column counts for all main tables (fast + cached)."""
    tables = ["patients", "encounters", "claims_and_billing", "denials"]
    rows = []
    with get_connection() as con:
        for tbl in tables:
            cnt = con.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0]
            cols = con.execute(
                f"""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = '{tbl}';
                """
            ).fetchone()[0]
            rows.append({"table": tbl, "row_count": cnt, "column_count": cols})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anomaly checks
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_duckdb_anomaly_checks(max_rows: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Deterministic anomaly checks (no AI agent).
    Returns a dict of check_name → DataFrame of flagged rows.
    """
    results: Dict[str, pd.DataFrame] = {}

    with get_connection() as con:
        results["encounter_missing_patient"] = (
            con.execute(
                """
                SELECT encounter_id, patient_id, visit_date, department
                FROM encounters
                WHERE patient_id IS NULL
                   OR patient_id NOT IN (SELECT patient_id FROM patients);
                """
            )
            .fetch_df()
            .head(max_rows)
        )

        results["overpaid_claims"] = (
            con.execute(
                """
                SELECT billing_id, claim_id, billed_amount, paid_amount, claim_status
                FROM claims_and_billing
                WHERE paid_amount > billed_amount;
                """
            )
            .fetch_df()
            .head(max_rows)
        )

        results["encounter_date_issues"] = (
            con.execute(
                """
                SELECT encounter_id, visit_date, discharge_date, department
                FROM encounters
                WHERE discharge_date IS NOT NULL
                  AND discharge_date < visit_date;
                """
            )
            .fetch_df()
            .head(max_rows)
        )

    return results


# ---------------------------------------------------------------------------
# DQI scoring
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_dqi_scores(max_rows_for_anomalies: int = 500) -> pd.DataFrame:
    """
    Compute an automated Data-Quality Index (DQI) score (0–100) per table.

    Components:
      - completeness_score : fraction of non-null cells × 100
      - anomaly_score      : (1 - anomaly_ratio) × 100
      - dqi_score          : 70% completeness + 30% anomaly_score
    """
    dq_df = run_duckdb_data_quality_scan()
    anomaly_dict = run_duckdb_anomaly_checks(max_rows=max_rows_for_anomalies)

    table_issues = {
        "encounters": (
            len(anomaly_dict.get("encounter_missing_patient", pd.DataFrame()))
            + len(anomaly_dict.get("encounter_date_issues", pd.DataFrame()))
        ),
        "claims_and_billing": len(anomaly_dict.get("overpaid_claims", pd.DataFrame())),
    }

    rows = []
    with get_connection() as con:
        for _, row in dq_df.iterrows():
            table = row["table"]
            row_count = int(row["row_count"])
            col_count = int(row["column_count"])

            total_nulls = 0
            if row_count > 0 and col_count > 0:
                cols_df = con.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                    ORDER BY ordinal_position;
                    """,
                    [table],
                ).fetch_df()

                for col_name in cols_df["column_name"]:
                    null_count = con.execute(
                        f'SELECT SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) FROM {table};'
                    ).fetchone()[0]
                    total_nulls += int(null_count or 0)

            total_cells = row_count * col_count
            completeness_score = (
                max(0.0, 100.0 * (1.0 - (total_nulls / total_cells)))
                if total_cells > 0
                else 0.0
            )

            issues = table_issues.get(table, 0)
            if row_count > 0:
                anomaly_ratio = min(issues / row_count, 1.0)
                anomaly_score = max(0.0, 100.0 * (1.0 - anomaly_ratio))
            else:
                anomaly_score = 100.0

            dqi_score = 0.7 * completeness_score + 0.3 * anomaly_score

            if dqi_score >= 90:
                bar_color = "green"
            elif dqi_score >= 75:
                bar_color = "gold"
            elif dqi_score >= 60:
                bar_color = "orange"
            else:
                bar_color = "red"

            safe_score = max(0.0, min(dqi_score, 100.0))
            color_bar = (
                f"<div style='background:{bar_color}; "
                f"width:{safe_score}%; height:12px; border-radius:4px;'></div>"
            )

            rows.append({
                "table": table,
                "row_count": row_count,
                "column_count": col_count,
                "null_cells": total_nulls,
                "completeness_score": round(completeness_score, 1),
                "anomaly_rows": issues,
                "anomaly_score": round(anomaly_score, 1),
                "dqi_score": round(dqi_score, 1),
                "dqi_bar": color_bar,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FHIR / HL7 validators
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_fhir_validations(max_rows: int = 500) -> Dict[str, pd.DataFrame]:
    """
    HL7 / FHIR-style healthcare validators using DuckDB.

    Checks:
      1. Patient.gender uses FHIR enum values
      2. Patient age vs DOB mismatch > 2 years
      3. ICD-10-like diagnosis_code pattern
      4. CPT-like procedure_code pattern (4–5 digits)
      5. Insurance provider sanity (missing / placeholder)
    """
    results: Dict[str, pd.DataFrame] = {}

    with get_connection() as con:

        try:
            results["patient_gender_not_fhir"] = (
                con.execute(
                    """
                    SELECT patient_id, first_name, last_name, gender
                    FROM patients
                    WHERE gender IS NULL
                       OR lower(gender) NOT IN ('male','female','other','unknown');
                    """
                )
                .fetch_df()
                .head(max_rows)
            )
        except Exception:
            results["patient_gender_not_fhir"] = pd.DataFrame()

        try:
            results["patient_age_dob_mismatch"] = (
                con.execute(
                    """
                    SELECT patient_id, first_name, last_name, dob, age, calc_age
                    FROM (
                        SELECT patient_id, first_name, last_name, dob, age,
                               DATEDIFF('year', STRPTIME(dob, '%d-%m-%Y'), CURRENT_DATE) AS calc_age
                        FROM patients
                        WHERE dob IS NOT NULL AND age IS NOT NULL
                    ) t
                    WHERE ABS(calc_age - age) > 2;
                    """
                )
                .fetch_df()
                .head(max_rows)
            )
        except Exception:
            results["patient_age_dob_mismatch"] = pd.DataFrame()

        try:
            results["diagnosis_invalid_icd10_pattern"] = (
                con.execute(
                    r"""
                    SELECT diagnosis_id, encounter_id, diagnosis_code
                    FROM diagnoses
                    WHERE diagnosis_code IS NULL
                       OR NOT (diagnosis_code ~ '^[A-TV-Z][0-9][0-9AB](\\.[0-9A-TV-Z]{1,4})?$');
                    """
                )
                .fetch_df()
            )
        except Exception:
            results["diagnosis_invalid_icd10_pattern"] = pd.DataFrame()

        try:
            results["procedure_invalid_cpt_pattern"] = (
                con.execute(
                    """
                    SELECT procedure_id, encounter_id, procedure_code
                    FROM procedures
                    WHERE procedure_code IS NULL
                       OR NOT (procedure_code ~ '^[0-9]{4,5}$');
                    """
                )
                .fetch_df()
                .head(max_rows)
            )
        except Exception:
            results["procedure_invalid_cpt_pattern"] = pd.DataFrame()

        try:
            results["claim_invalid_insurance_provider"] = (
                con.execute(
                    """
                    SELECT billing_id, patient_id, encounter_id, insurance_provider
                    FROM claims_and_billing
                    WHERE insurance_provider IS NULL
                       OR TRIM(insurance_provider) = ''
                       OR lower(insurance_provider) IN ('unknown','n/a','na');
                    """
                )
                .fetch_df()
                .head(max_rows)
            )
        except Exception:
            results["claim_invalid_insurance_provider"] = pd.DataFrame()

    return results


# ---------------------------------------------------------------------------
# AI-generated DQ insights
# ---------------------------------------------------------------------------

def generate_dq_insights() -> str:
    """
    Use the ETL Agent LLM to generate human-readable insights
    from DQI scores, row/column profile, and anomaly checks.
    Returns a markdown string.
    """
    dq_df = st.session_state.get("dq_results")
    if dq_df is None:
        dq_df = run_duckdb_data_quality_scan()
        st.session_state.dq_results = dq_df

    dqi_df = st.session_state.get("dqi_results")
    if dqi_df is None:
        dqi_df = compute_dqi_scores(max_rows_for_anomalies=500)
        st.session_state.dqi_results = dqi_df

    anomalies = st.session_state.get("anomaly_results")
    if anomalies is None:
        anomalies = run_duckdb_anomaly_checks(max_rows=500)
        st.session_state.anomaly_results = anomalies

    overall = st.session_state.get("dqi_overall")

    summary_lines = []
    if overall is not None:
        summary_lines.append(f"Overall DQI score: {overall:.1f} / 100.")

    summary_lines.append("\nPer-table DQI (rows, completeness, anomalies, final score):")
    for _, row in dqi_df.iterrows():
        summary_lines.append(
            f"- {row['table']}: rows={row['row_count']}, "
            f"completeness={row['completeness_score']}, "
            f"anomaly_rows={row['anomaly_rows']}, "
            f"dqi={row['dqi_score']}."
        )

    summary_lines.append("\nAnomaly checks summary:")
    for name, df in anomalies.items():
        count = 0 if df is None else len(df)
        summary_lines.append(f"- {name}: {count} rows flagged.")

    dq_text = "\n".join(summary_lines)

    prompt = f"""
You are a senior healthcare data-quality engineer.
Do NOT run any tools or SQL. Only read the summary below and generate insights.

Data-quality summary (from DuckDB metrics and anomaly checks):
{dq_text}

Write a concise markdown report with these sections:

1. **Overall Assessment** – 2–4 bullet points summarizing overall data quality.
2. **Key Table-Level Issues** – bullet points per table that has notable problems
   (low completeness, many anomalies, or lower DQI).
3. **Recommended Fixes & Validation Rules** – 4–7 concrete actions that an ETL
   / data engineering team should implement (e.g., constraints, checks, alerts).

Keep it under 300 words. Be specific but not verbose.
"""

    return run_agent(
        prompt,
        chat_history=[],
        provider_name=st.session_state.get("llm_provider", "openai"),
        openai_model=st.session_state.get("openai_model"),
        local_model=st.session_state.get("local_model"),
    )


# ---------------------------------------------------------------------------
# PDF report builder
# ---------------------------------------------------------------------------

def build_pdf_report() -> bytes:
    """
    Build a PDF data-quality report from current session state.
    Includes: Overall DQI, row/column profile, DQI per table, AI insights.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=50,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("CA Hospital Data-Quality Report", styles["Title"]))
    story.append(Spacer(1, 12))

    overall = st.session_state.get("dqi_overall")
    if overall is not None:
        if overall >= 90:
            status = "Excellent"
        elif overall >= 75:
            status = "Good"
        elif overall >= 60:
            status = "Fair"
        else:
            status = "Needs Attention"
        text = f"Overall DQI: {overall:.1f} / 100 ({status})"
    else:
        text = "Overall DQI: Not yet computed."
    story.append(Paragraph(text, styles["Heading3"]))
    story.append(Spacer(1, 12))

    dq_df = st.session_state.get("dq_results")
    if dq_df is not None and not dq_df.empty:
        story.append(Paragraph("Row & Column Profile", styles["Heading3"]))
        story.append(Spacer(1, 6))
        dq_subset = dq_df[["table", "row_count", "column_count"]]
        dq_data = [list(dq_subset.columns)] + dq_subset.values.tolist()
        dq_table = Table(dq_data, hAlign="LEFT")
        dq_table.setStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
        story.append(dq_table)
        story.append(Spacer(1, 12))

    dqi_df = st.session_state.get("dqi_results")
    if dqi_df is not None and not dqi_df.empty:
        story.append(Paragraph("DQI per Table", styles["Heading3"]))
        story.append(Spacer(1, 6))
        cols = ["table", "row_count", "column_count", "null_cells",
                "completeness_score", "anomaly_rows", "anomaly_score", "dqi_score"]
        dqi_subset = dqi_df[cols]
        dqi_data = [cols] + dqi_subset.values.tolist()
        dqi_table = Table(dqi_data, hAlign="LEFT")
        dqi_table.setStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
        story.append(dqi_table)
        story.append(Spacer(1, 12))

    insights = st.session_state.get("dq_insights")
    if insights:
        story.append(Paragraph("AI Data-Quality Insights", styles["Heading3"]))
        story.append(Spacer(1, 6))
        clean = (
            insights.replace("**", "")
            .replace("# ", "").replace("## ", "").replace("### ", "")
        )
        for line in clean.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), styles["BodyText"]))
                story.append(Spacer(1, 3))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
