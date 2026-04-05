# agent/live_compare.py

import os
from typing import Optional, List, Tuple
from pathlib import Path  # NEW

import duckdb
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# Use the same base-dir logic as etl_agent.py
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "hospital.duckdb"

LIVE_DB_URL = os.getenv("LIVE_DB_URL")



class ClaimsRow(BaseModel):
    billing_id: str
    patient_id: str
    encounter_id: str
    insurance_provider: str
    claim_status: str
    billed_amount: float
    paid_amount: float
    payment_method: Optional[str] = None
    claim_id: Optional[str] = None
    denial_reason: Optional[str] = None

    @field_validator("billed_amount", "paid_amount")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount cannot be negative")
        return v


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)



def get_live_engine() -> Optional[Engine]:
    if not LIVE_DB_URL:
        return None
    return create_engine(LIVE_DB_URL)


def validate_sample(df: pd.DataFrame) -> Tuple[int, int, List[str]]:
    """Validate a dataframe sample with Pydantic. Returns (ok_count, error_count, error_examples)."""
    ok = 0
    errors = 0
    error_examples: List[str] = []

    for _, row in df.iterrows():
        try:
            ClaimsRow(**row.to_dict())
            ok += 1
        except ValidationError as e:
            errors += 1
            if len(error_examples) < 5:
                error_examples.append(str(e))

    return ok, errors, error_examples


def compare_claims_with_live(sample_size: int = 500) -> str:
    """
    Compare claims_and_billing in DuckDB vs the live DB:
    - basic stats differences
    - Pydantic validation errors for each side
    Returns a plain-text summary for the UI.
    """
    engine = get_live_engine()
    if engine is None:
        return (
            "Live DB comparison is not configured.\n\n"
            "Set LIVE_DB_URL in your .env (SQLAlchemy URL to your test database) "
            "to enable this feature."
        )

    with get_duckdb_connection() as con:
        duck_df = con.execute(
            f"""
            SELECT billing_id, patient_id, encounter_id,
                   insurance_provider, claim_status,
                   billed_amount, paid_amount,
                   payment_method, claim_id, denial_reason
            FROM claims_and_billing
            LIMIT {sample_size}
            """
        ).fetchdf()

    # NOTE: for SQL Server we use TOP; change to LIMIT if using Postgres/MySQL
    live_df = pd.read_sql(
        f"""
        SELECT TOP {sample_size}
               billing_id, patient_id, encounter_id,
               insurance_provider, claim_status,
               billed_amount, paid_amount,
               payment_method, claim_id, denial_reason
        FROM claims_and_billing
        """,
        engine,
    )

    # Pydantic validation
    duck_ok, duck_err, duck_err_examples = validate_sample(duck_df)
    live_ok, live_err, live_err_examples = validate_sample(live_df)

    # Quick numeric stats
    def num_stats(df: pd.DataFrame, col: str) -> Tuple[float, float, float]:
        s = df[col].astype(float)
        return float(s.min()), float(s.max()), float(s.mean())

    duck_min, duck_max, duck_mean = num_stats(duck_df, "billed_amount")
    live_min, live_max, live_mean = num_stats(live_df, "billed_amount")

    summary_lines = [
        "🔄 DuckDB vs Live DB — claims_and_billing (sample comparison)",
        "",
        f"Sample size (per source): {len(duck_df)} DuckDB rows, {len(live_df)} Live DB rows",
        "",
        "💰 Billed Amount (DuckDB vs Live)",
        f"- min:   {duck_min:,.2f}  vs  {live_min:,.2f}",
        f"- max:   {duck_max:,.2f}  vs  {live_max:,.2f}",
        f"- mean:  {duck_mean:,.2f}  vs  {live_mean:,.2f}",
        "",
        "✅ Pydantic validation",
        f"- DuckDB: {duck_ok} ok, {duck_err} with errors",
        f"- Live DB: {live_ok} ok, {live_err} with errors",
    ]

    if duck_err_examples or live_err_examples:
        summary_lines.append("")
        summary_lines.append("Example validation errors:")
        for e in duck_err_examples:
            summary_lines.append(f"- DuckDB row error: {e}")
        for e in live_err_examples:
            summary_lines.append(f"- Live DB row error: {e}")

    return "\n".join(summary_lines)
