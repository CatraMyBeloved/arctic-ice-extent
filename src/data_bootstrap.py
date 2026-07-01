"""Make training runs self-bootstrapping and reproducible.

Training notebooks/scripts should call :func:`ensure_extent_data` before loading
data. It guarantees the pan-Arctic daily extent table exists in Postgres,
downloading the NSIDC source file and ingesting it if necessary. Every step is
idempotent, so it is safe to call at the top of every run.

The ingestion here mirrors notebook ``01b`` for the pan-Arctic daily table so
that a fresh machine can reproduce the training data from nothing but the DB
connection and internet access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import sqlalchemy

from .data_utils import DATABASE_URL, DATA_DIR, PROJECT_DIR

RAW_TABLES_DIR = DATA_DIR / "raw" / "tables"
DAILY_CSV = RAW_TABLES_DIR / "N_seaice_extent_daily_v4.0.csv"
PAN_ARCTIC_TABLE = "ice_extent_pan_arctic_daily"

# NSIDC G02135 v4.0 pan-Arctic daily extent file.
NSIDC_BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135"
DAILY_REL_URL = "north/daily/data/N_seaice_extent_daily_v4.0.csv"


def _table_populated(engine: sqlalchemy.Engine, table: str) -> bool:
    """Return True if ``table`` exists and has at least one row."""
    if not sqlalchemy.inspect(engine).has_table(table):
        return False
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", engine)["n"].iloc[0]
    return int(n) > 0


def _download_daily_csv() -> None:
    """Download the NSIDC daily extent CSV, reusing the retry-capable session."""
    # utilities/ is not a package; add it to the path to reuse the downloader.
    sys.path.insert(0, str(PROJECT_DIR / "utilities"))
    from download_geotiffs import make_session, download_file  # type: ignore

    RAW_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{NSIDC_BASE_URL}/{DAILY_REL_URL}"
    print(f"Downloading {DAILY_CSV.name} from NSIDC...")
    download_file(make_session(), url, str(DAILY_CSV))


def _ingest_pan_arctic_daily(engine: sqlalchemy.Engine) -> int:
    """Load the daily CSV into the pan-Arctic table (mirrors notebook 01b)."""
    df = pd.read_csv(
        DAILY_CSV,
        skiprows=2,
        names=["year", "month", "day", "extent", "missing", "source"],
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["region"] = "pan_arctic"
    df = df[["date", "region", "extent"]].rename(columns={"extent": "extent_mkm2"})
    df.to_sql(PAN_ARCTIC_TABLE, engine, if_exists="replace", index=False)
    return len(df)


def ensure_extent_data(download: bool = True, verbose: bool = True) -> None:
    """Ensure the pan-Arctic daily extent table is present and populated.

    Args:
        download: If True, download the NSIDC source when it is missing. If
            False and the data is absent, raise with instructions instead.
        verbose: Print what was done.

    Raises:
        RuntimeError: If the DB is unreachable, or data is missing and either
            ``download`` is False or ingestion fails.
    """
    try:
        engine = sqlalchemy.create_engine(DATABASE_URL)
        with engine.connect():
            pass
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable message
        raise RuntimeError(
            f"Cannot reach the database at {DATABASE_URL}. "
            "Start it with `podman compose up -d` (see docs/database.md)."
        ) from e

    if _table_populated(engine, PAN_ARCTIC_TABLE):
        if verbose:
            print(f"✓ '{PAN_ARCTIC_TABLE}' already populated.")
        return

    if not DAILY_CSV.exists():
        if not download:
            raise RuntimeError(
                f"Missing {DAILY_CSV} and download=False. "
                "Run `uv run python utilities/download_nsidc.py` or pass download=True."
            )
        _download_daily_csv()

    n = _ingest_pan_arctic_daily(engine)
    if verbose:
        print(f"✓ Ingested {n} rows into '{PAN_ARCTIC_TABLE}'.")
