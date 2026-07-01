"""Download the NSIDC G02135 sea ice index tables used by 01b ingestion.

Fetches the three source files that `notebooks/01b_data_ingestion_nsidc.ipynb` loads into
PostgreSQL. Reuses the retry/session helpers from `download_geotiffs`. Existing files are
skipped, so this is safe to re-run.

Usage:
    uv run python utilities/download_nsidc.py
"""

import os

from download_geotiffs import make_session, download_file

BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135"

# (relative URL under BASE_URL, local filename) — matches the paths 01b reads.
FILES = [
    ("north/daily/data/N_seaice_extent_daily_v4.0.csv",
     "N_seaice_extent_daily_v4.0.csv"),
    ("north/daily/data/N_seaice_extent_climatology_1981-2010_v4.0.csv",
     "N_seaice_extent_climatology_1981-2010_v4.0.csv"),
    ("seaice_analysis/N_Sea_Ice_Index_Regional_Daily_Data_G02135_v4.0.xlsx",
     "N_Sea_Ice_Index_Regional_Daily_Data_G02135_v4.0.xlsx"),
]


def download_nsidc_tables(output_dir: str = "../data/raw/tables") -> None:
    """Download the pan-Arctic daily, climatology, and regional tables to output_dir."""
    session = make_session()
    os.makedirs(output_dir, exist_ok=True)
    for rel_url, filename in FILES:
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest):
            print(f"Skipping existing {filename}")
            continue
        url = f"{BASE_URL}/{rel_url}"
        print(f"Downloading {filename}")
        download_file(session, url, dest)
    print(f"Done. Files in {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    download_nsidc_tables()
