# etl/load_duckdb.py

import duckdb
import pandas as pd
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"

DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.getenv("DUCKDB_PATH", DB_DIR / "hospital.duckdb"))

TABLE_FILES = {
    "claims_and_billing": "claims_and_billing.csv",
    "denials": "denials.csv",
    "diagnoses": "diagnoses.csv",
    "encounters": "encounters.csv",
    "lab_tests": "lab_tests.csv",
    "medications": "medications.csv",
    "patients": "patients.csv",
    "procedures": "procedures.csv",
    "providers": "providers.csv",
}

# Known date-like columns across datasets (only parsed if present in each CSV)
DATE_COL_CANDIDATES = [
    "claim_billing_date",
    "denial_date",
    "appeal_resolution_date",
    "visit_date",
    "admission_date",
    "discharge_date",
    "test_date",
    "prescribed_date",
    "registration_date",
    "procedure_date",
]


def _read_csv_safely(file_path: Path) -> pd.DataFrame:
    """
    Read CSV with:
    - low_memory=False for stable dtype inference
    - parse_dates for known date columns (only if they exist)
    - adds deterministic ETL load timestamp (UTC)
    """
    # Read header only once to detect date columns
    header_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    parse_dates = [c for c in DATE_COL_CANDIDATES if c in header_cols]

    df = pd.read_csv(
        file_path,
        low_memory=False,
        parse_dates=parse_dates if parse_dates else None,
    )

    # Research/traceability metadata
    df["_etl_loaded_at"] = pd.Timestamp.utcnow()

    return df


def load_tables() -> None:
    print(f"[INFO] Using DB: {DB_PATH}")
    print(f"[INFO] Using DATA_DIR: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    con = duckdb.connect(DB_PATH.as_posix())

    try:
        for table_name, file_name in TABLE_FILES.items():
            file_path = DATA_DIR / file_name
            if not file_path.exists():
                print(f"[WARN] File not found: {file_path}")
                continue

            print(f"[INFO] Loading {file_name} into table {table_name} ...")
            df = _read_csv_safely(file_path)

            # Overwrite table each run for deterministic rebuild
            con.execute(f"DROP TABLE IF EXISTS {table_name};")
            con.register("df_temp", df)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_temp;")
            con.unregister("df_temp")

            # Post-load sanity row count (useful for reproducibility)
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"[DQ] {table_name}: {row_count} rows loaded")

        print("\n[INFO] Tables in database:")
        print(con.execute("SHOW TABLES;").fetchdf())

    finally:
        con.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    load_tables()
