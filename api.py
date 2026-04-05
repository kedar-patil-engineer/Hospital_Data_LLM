# api.py
#
# Phase 2 change: API key authentication added via FastAPI dependency injection.
#
# Setup:
#   1. Add API_KEY=your-secret-key to your .env file
#   2. All endpoints now require:  X-API-Key: your-secret-key  header
#   3. Missing or wrong key returns HTTP 401
#
# Example curl:
#   curl -H "X-API-Key: your-secret-key" http://localhost:8000/dq/scan
#
# The /health endpoint is intentionally public (no auth required).

from typing import List, Dict, Any
import os

import pandas as pd
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent.etl_agent import run_agent, get_connection
from agent.errors import AgentError, ErrorType

load_dotenv()

# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)


def _get_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency — validates the X-API-Key header against
    the API_KEY environment variable.

    Returns the key if valid, raises HTTP 401 if missing or wrong.
    Set API_KEY in your .env file. If API_KEY is not set, auth is
    disabled (development mode) and a warning is logged.
    """
    expected = os.getenv("API_KEY", "").strip()

    if not expected:
        # Dev mode — no key configured, allow all requests
        # Log a warning so it's visible in server output
        import warnings
        warnings.warn(
            "API_KEY is not set in .env — authentication is disabled. "
            "Set API_KEY=your-secret-key for production use.",
            stacklevel=2,
        )
        return "dev-mode"

    if api_key == expected:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": True,
            "error_type": ErrorType.GOVERNANCE_BLOCK.value,
            "detail": "Invalid or missing API key. Provide a valid X-API-Key header.",
        },
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CA Hospital Data-Quality Copilot API",
    description=(
        "FastAPI layer on top of the CA Hospital Data-Quality Copilot.\n\n"
        "**Authentication:** All endpoints (except /health) require an "
        "`X-API-Key` header. Set `API_KEY` in your `.env` file.\n\n"
        "Exposes: LLM agent, deterministic DuckDB data-quality checks, "
        "and KPI comparison as REST endpoints."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AgentQuery(BaseModel):
    query: str
    provider: str = "openai"
    openai_model: str | None = None
    local_model: str | None = None


class AgentAnswer(BaseModel):
    answer: str


class ClaimsKpiSnapshot(BaseModel):
    period_label: str
    total_billed: float = Field(..., ge=0)
    total_paid: float = Field(..., ge=0)
    denial_rate: float = Field(
        ..., ge=0, le=1,
        description="Denial rate as a 0–1 ratio (e.g., 0.12 for 12%)",
    )


class KPICompareInput(BaseModel):
    period_label: str = "Your System Snapshot"
    total_billed: float = Field(..., ge=0)
    total_paid: float = Field(..., ge=0)
    denial_rate: float = Field(..., ge=0, le=1)


class KPICompareResult(BaseModel):
    sample: ClaimsKpiSnapshot
    user: ClaimsKpiSnapshot
    delta_billed: float
    delta_paid: float
    delta_denial_rate: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sample_claims_kpi() -> ClaimsKpiSnapshot:
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
    return ClaimsKpiSnapshot(
        period_label="Sample CA Q1-2025",
        total_billed=float(row[0] or 0),
        total_paid=float(row[1] or 0),
        denial_rate=float(row[2] or 0),
    )


def dq_scan_basic() -> pd.DataFrame:
    tables = ["patients", "encounters", "claims_and_billing", "denials"]
    rows: List[Dict[str, Any]] = []
    with get_connection() as con:
        for tbl in tables:
            row_count = con.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0]
            col_count = con.execute(
                f"SELECT COUNT(*) FROM information_schema.columns "
                f"WHERE table_schema = 'main' AND table_name = '{tbl}';"
            ).fetchone()[0]
            rows.append({"table": tbl, "row_count": int(row_count), "column_count": int(col_count)})
    return pd.DataFrame(rows)


def anomaly_checks(max_rows: int = 500) -> Dict[str, pd.DataFrame]:
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
            ).fetch_df().head(max_rows)
        )
        results["overpaid_claims"] = (
            con.execute(
                """
                SELECT billing_id, claim_id, billed_amount, paid_amount, claim_status
                FROM claims_and_billing
                WHERE paid_amount > billed_amount;
                """
            ).fetch_df().head(max_rows)
        )
        results["encounter_date_issues"] = (
            con.execute(
                """
                SELECT encounter_id, visit_date, discharge_date, department
                FROM encounters
                WHERE discharge_date IS NOT NULL
                  AND discharge_date < visit_date;
                """
            ).fetch_df().head(max_rows)
        )
    return results


def compute_dqi_scores(max_rows_for_anomalies: int = 500) -> pd.DataFrame:
    dq_df = dq_scan_basic()
    anomaly_dict = anomaly_checks(max_rows=max_rows_for_anomalies)

    table_to_anomalies: Dict[str, int] = {
        "encounters": (
            len(anomaly_dict.get("encounter_missing_patient", pd.DataFrame()))
            + len(anomaly_dict.get("encounter_date_issues", pd.DataFrame()))
        ),
        "claims_and_billing": len(anomaly_dict.get("overpaid_claims", pd.DataFrame())),
        "patients": 0,
        "denials": 0,
    }

    completeness_scores: List[float] = []
    anomaly_scores: List[float] = []

    with get_connection() as con:
        for _, row in dq_df.iterrows():
            tbl = row["table"]
            row_count = int(row["row_count"])
            col_count = int(row["column_count"])

            columns = con.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema = 'main' AND table_name = '{tbl}';"
            ).fetchall()
            col_names = [r[0] for r in columns]

            if row_count == 0 or not col_names:
                completeness_scores.append(100.0)
            else:
                expr = " + ".join(
                    [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in col_names]
                )
                null_cells = con.execute(f"SELECT {expr} AS null_cells FROM {tbl};").fetchone()[0]
                total_cells = row_count * len(col_names)
                completeness = (
                    100.0 * (1.0 - (float(null_cells or 0) / float(total_cells)))
                    if total_cells > 0 else 100.0
                )
                completeness_scores.append(max(0.0, min(100.0, completeness)))

            issue_rows = table_to_anomalies.get(tbl, 0)
            if row_count == 0:
                anomaly_scores.append(100.0)
            else:
                ratio = min(1.0, issue_rows / float(row_count))
                anomaly_scores.append(100.0 * (1.0 - ratio))

    dq_df["completeness_score"] = completeness_scores
    dq_df["anomaly_score"] = anomaly_scores
    dq_df["dqi_score"] = 0.7 * dq_df["completeness_score"] + 0.3 * dq_df["anomaly_score"]
    return dq_df


# ---------------------------------------------------------------------------
# Public endpoint (no auth)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Public health check — no authentication required."""
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------------------
# Protected endpoints (require X-API-Key header)
# ---------------------------------------------------------------------------

@app.post("/agent/query", response_model=AgentAnswer, tags=["Agent"])
def agent_query(
    payload: AgentQuery,
    _key: str = Security(_get_api_key),
) -> AgentAnswer:
    """Natural-language query endpoint for the ETL agent."""
    answer = run_agent(
        payload.query,
        provider_name=payload.provider,
        openai_model=payload.openai_model,
        local_model=payload.local_model,
    )
    return AgentAnswer(answer=answer)


@app.get("/dq/scan", tags=["Data Quality"])
def dq_scan_endpoint(
    _key: str = Security(_get_api_key),
) -> List[Dict[str, Any]]:
    """Basic data-quality scan: row + column counts per core table."""
    return dq_scan_basic().to_dict(orient="records")


@app.get("/dq/checks", tags=["Data Quality"])
def dq_checks_endpoint(
    max_rows: int = 500,
    _key: str = Security(_get_api_key),
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic anomaly checks on DuckDB tables."""
    results = anomaly_checks(max_rows=max_rows)
    return {name: df.to_dict(orient="records") for name, df in results.items()}


@app.get("/dq/dqi", tags=["Data Quality"])
def dq_dqi_endpoint(
    max_rows_for_anomalies: int = 500,
    _key: str = Security(_get_api_key),
) -> List[Dict[str, Any]]:
    """Compute a simple DQI score per table (0–100)."""
    return compute_dqi_scores(max_rows_for_anomalies).to_dict(orient="records")


@app.get("/insights", tags=["Agent"])
def dq_insights_endpoint(
    max_rows_for_anomalies: int = 500,
    provider: str = "openai",
    openai_model: str | None = None,
    local_model: str | None = None,
    _key: str = Security(_get_api_key),
) -> dict:
    """Generate AI-powered DQ insights from DQI scores."""
    dqi_df = compute_dqi_scores(max_rows_for_anomalies)
    dqi_json = dqi_df.to_dict(orient="records")
    prompt = (
        "You are a healthcare data-quality expert.\n"
        f"Here is table-level DQI data:\n{dqi_json}\n\n"
        "Summarize risks, problem tables, and next actions."
    )
    answer = run_agent(prompt, provider_name=provider, openai_model=openai_model, local_model=local_model)
    return {"insights": answer}


@app.post("/kpi/compare", response_model=KPICompareResult, tags=["KPI"])
def kpi_compare_endpoint(
    payload: KPICompareInput,
    _key: str = Security(_get_api_key),
) -> KPICompareResult:
    """Compare user-provided KPI snapshot with the sample DuckDB dataset."""
    sample = get_sample_claims_kpi()
    user = ClaimsKpiSnapshot(
        period_label=payload.period_label,
        total_billed=payload.total_billed,
        total_paid=payload.total_paid,
        denial_rate=payload.denial_rate,
    )
    return KPICompareResult(
        sample=sample,
        user=user,
        delta_billed=user.total_billed - sample.total_billed,
        delta_paid=user.total_paid - sample.total_paid,
        delta_denial_rate=user.denial_rate - sample.denial_rate,
    )
