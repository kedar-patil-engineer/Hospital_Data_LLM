# analytics.py
"""
Advanced analytics helpers for the CA Hospital Data-Quality Copilot.

Extracted from app.py to keep it maintainable.
Contains all DuckDB-backed analytics and chart data functions:
  - get_denial_trend()
  - get_billing_distribution()
  - get_provider_efficiency()
  - get_los_distribution()
  - get_department_costs()
  - get_cost_per_encounter()
  - get_claim_status_counts()
  - get_billed_amount_sample()
  - build_advanced_charts()
  - build_claims_charts()

Import in app.py:
    from analytics import (
        build_advanced_charts,
        build_claims_charts,
    )
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from agent.etl_agent import get_connection


# ---------------------------------------------------------------------------
# Chart data helpers (all cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_denial_trend() -> pd.DataFrame:
    """Monthly denial rate trend based on claim_billing_date."""
    with get_connection() as con:
        return con.execute(
            """
            SELECT
                STRFTIME(STRPTIME(claim_billing_date, '%d-%m-%Y %H:%M'), '%Y-%m') AS month,
                COUNT(*) AS total_claims,
                SUM(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) AS denied_claims,
                CASE
                    WHEN COUNT(*) = 0 THEN 0
                    ELSE SUM(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) * 1.0
                         / COUNT(*)
                END AS denial_rate
            FROM claims_and_billing
            WHERE claim_billing_date IS NOT NULL
            GROUP BY month
            ORDER BY month;
            """
        ).fetch_df()


@st.cache_data(show_spinner=False)
def get_billing_distribution(sample_size: int = 3000) -> pd.DataFrame:
    """Bucketed billed-amount distribution for histogram-like bar chart."""
    with get_connection() as con:
        df = con.execute(
            "SELECT billed_amount FROM claims_and_billing WHERE billed_amount IS NOT NULL;"
        ).fetch_df()

    if df.empty:
        return df

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    df = df[df["billed_amount"].notna()].copy()
    df["bucket"] = pd.cut(df["billed_amount"], bins=10)
    bucket_counts = (
        df.groupby("bucket").size().reset_index(name="cnt").sort_values("bucket")
    )
    bucket_counts["bucket_label"] = bucket_counts["bucket"].astype(str)
    return bucket_counts[["bucket_label", "cnt"]]


@st.cache_data(show_spinner=False)
def get_provider_efficiency(top_n: int = 10) -> pd.DataFrame:
    """Provider-level efficiency: encounters, claims, billed, paid, denial rate."""
    with get_connection() as con:
        df = con.execute(
            """
            SELECT
                e.provider_id,
                COUNT(DISTINCT e.encounter_id) AS encounters,
                COUNT(DISTINCT cb.billing_id) AS claims,
                SUM(cb.billed_amount) AS total_billed,
                SUM(cb.paid_amount) AS total_paid,
                SUM(CASE WHEN cb.claim_status = 'Denied' THEN 1 ELSE 0 END) AS denied_claims
            FROM encounters e
            LEFT JOIN claims_and_billing cb ON cb.encounter_id = e.encounter_id
            GROUP BY e.provider_id;
            """
        ).fetch_df()

    if df.empty:
        return df

    df["denial_rate"] = df.apply(
        lambda r: (r["denied_claims"] / r["claims"]) if r["claims"] else 0.0, axis=1
    )
    return df.sort_values("total_billed", ascending=False).head(top_n).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_los_distribution() -> pd.DataFrame:
    """Length-of-stay histogram data."""
    with get_connection() as con:
        df = con.execute(
            "SELECT length_of_stay FROM encounters WHERE length_of_stay IS NOT NULL;"
        ).fetch_df()

    if df.empty:
        return df

    df = df[df["length_of_stay"].notna()].copy()
    df["bucket"] = pd.cut(df["length_of_stay"], bins=10)
    bucket_counts = (
        df.groupby("bucket").size().reset_index(name="cnt").sort_values("bucket")
    )
    bucket_counts["bucket_label"] = bucket_counts["bucket"].astype(str)
    return bucket_counts[["bucket_label", "cnt"]]


@st.cache_data(show_spinner=False)
def get_department_costs() -> pd.DataFrame:
    """Average cost per encounter by department."""
    with get_connection() as con:
        return con.execute(
            """
            WITH cost_per_encounter AS (
                SELECT encounter_id, SUM(billed_amount) AS total_billed
                FROM claims_and_billing
                GROUP BY encounter_id
            )
            SELECT
                e.department,
                COUNT(*) AS encounters,
                AVG(c.total_billed) AS avg_billed_per_encounter
            FROM encounters e
            JOIN cost_per_encounter c ON e.encounter_id = c.encounter_id
            WHERE e.department IS NOT NULL
            GROUP BY e.department
            ORDER BY avg_billed_per_encounter DESC;
            """
        ).fetch_df()


@st.cache_data(show_spinner=False)
def get_cost_per_encounter() -> pd.DataFrame:
    """Billed/paid aggregated by encounter."""
    with get_connection() as con:
        return con.execute(
            """
            SELECT encounter_id,
                   SUM(billed_amount) AS total_billed,
                   SUM(paid_amount) AS total_paid
            FROM claims_and_billing
            GROUP BY encounter_id;
            """
        ).fetch_df()


@st.cache_data(show_spinner=False)
def get_claim_status_counts() -> pd.DataFrame:
    """Counts by claim_status for bar chart."""
    with get_connection() as con:
        return con.execute(
            """
            SELECT claim_status, COUNT(*) AS cnt
            FROM claims_and_billing
            GROUP BY claim_status
            ORDER BY cnt DESC;
            """
        ).fetch_df()


@st.cache_data(show_spinner=False)
def get_billed_amount_sample(sample_size: int = 3000) -> pd.DataFrame:
    """Sample billed_amount for distribution chart."""
    with get_connection() as con:
        df = con.execute(
            "SELECT billed_amount FROM claims_and_billing WHERE billed_amount IS NOT NULL;"
        ).fetch_df()

    if df.empty:
        return df

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    return df.sort_values("billed_amount").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart renderers (Streamlit UI — call from app.py only)
# ---------------------------------------------------------------------------

def build_advanced_charts() -> None:
    """Render advanced analytics charts in the Streamlit main area."""
    st.markdown("## Advanced Analytics Dashboard")

    denial_df = get_denial_trend()
    if not denial_df.empty:
        st.markdown("### Denial Rate Trend (Monthly)")
        st.line_chart(denial_df.set_index("month")[["denial_rate"]], use_container_width=True)
        st.caption("Denial rate = denied claims / total claims per month.")
    else:
        st.info("No data available for denial trend.")

    st.markdown("---")

    bill_dist = get_billing_distribution()
    if not bill_dist.empty:
        st.markdown("### Billed Amount Distribution (Sampled)")
        st.bar_chart(bill_dist.set_index("bucket_label")["cnt"], use_container_width=True)
        st.caption("Sampled billed amounts binned into ranges for distribution view.")
    else:
        st.info("No billed amount data available for distribution.")

    st.markdown("---")

    prov_df = get_provider_efficiency()
    if not prov_df.empty:
        st.markdown("### Provider Efficiency (Top by Total Billed)")
        st.bar_chart(prov_df.set_index("provider_id")[["total_paid"]], use_container_width=True)
        st.caption("Total paid amount per provider (top by total billed).")
        st.dataframe(
            prov_df[["provider_id", "encounters", "claims", "total_billed", "total_paid", "denial_rate"]],
            use_container_width=True,
        )
    else:
        st.info("No provider-level data available.")

    st.markdown("---")

    los_df = get_los_distribution()
    if not los_df.empty:
        st.markdown("### Length of Stay Distribution")
        st.bar_chart(los_df.set_index("bucket_label")["cnt"], use_container_width=True)
        st.caption("Distribution of encounter length_of_stay values (bucketed).")
    else:
        st.info("No length_of_stay data available.")

    st.markdown("---")

    dept_df = get_department_costs()
    if not dept_df.empty:
        st.markdown("### Average Cost per Encounter by Department")
        st.bar_chart(dept_df.set_index("department")[["avg_billed_per_encounter"]], use_container_width=True)
        st.caption("Higher bars = departments with higher average billed cost per encounter.")
    else:
        st.info("No department cost data available.")


def build_claims_charts() -> None:
    """Render claims/billing charts in the Streamlit sidebar or main area."""
    st.subheader("Claims/Billing Visuals")

    status_df = get_claim_status_counts()
    billed_df = get_billed_amount_sample()

    if not status_df.empty:
        st.caption("Claims by Status")
        st.bar_chart(data=status_df.set_index("claim_status"), use_container_width=True)
    else:
        st.info("No claims data found for status breakdown.")

    if not billed_df.empty:
        st.caption("Sampled Billed Amount Distribution")
        st.line_chart(billed_df, use_container_width=True)
    else:
        st.info("No billed amount distribution available.")
