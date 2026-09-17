"""Build the unified match dataset.

Pulls data/raw/matches_source.csv from xgabora/Club-Football-Match-Data on
GitHub (see schema.py for why -- football-data.co.uk itself blocks
automated downloads), filters to the leagues in schema.LEAGUES, normalizes
column names, derives a season label, and writes
data/processed/matches.parquet.

Usage:
    python -m betting_model.ingest              # download if missing, then build
    python -m betting_model.ingest --refresh     # force re-download first
    python -m betting_model.ingest --no-download # only use what's already in data/raw/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

from betting_model.schema import COLUMN_MAP, LEAGUES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "matches_source.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "matches.parquet"

SOURCE_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/main/data/Matches.csv"


def download(force: bool = False, timeout: float = 120.0) -> None:
    if RAW_PATH.exists() and not force:
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SOURCE_URL, timeout=timeout)
    resp.raise_for_status()
    RAW_PATH.write_bytes(resp.content)


def _season_label(match_date: pd.Timestamp) -> str:
    # football-data.co.uk seasons run roughly July-June; a match in Jan 2024
    # belongs to the 2023-24 season, one in July 2024 to 2024-25.
    start_year = match_date.year if match_date.month >= 7 else match_date.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["Division"].isin(LEAGUES)].copy()

    unified = pd.DataFrame(index=raw.index)
    for field, source_col in COLUMN_MAP.items():
        unified[field] = raw[source_col] if source_col in raw.columns else pd.NA

    unified["date"] = pd.to_datetime(unified["date"], errors="coerce")
    for goal_col in ("fthg", "ftag", "hthg", "htag"):
        unified[goal_col] = pd.to_numeric(unified[goal_col], errors="coerce")

    unified["league"] = raw["Division"].map(LEAGUES)
    unified["season"] = unified["date"].apply(_season_label)

    unified = unified.dropna(subset=["date", "home_team", "away_team", "ftr"])
    unified = unified.sort_values(["league", "date"]).reset_index(drop=True)
    return unified


def build_dataset(attempt_download: bool = True, force_download: bool = False) -> pd.DataFrame:
    if attempt_download:
        download(force=force_download)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found and --no-download was set. "
            f"Either drop a copy there, or omit --no-download to fetch it from {SOURCE_URL}"
        )

    raw = pd.read_csv(RAW_PATH, low_memory=False)
    return normalize(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="force re-download of the source CSV")
    parser.add_argument(
        "--no-download", action="store_true",
        help="don't fetch the source CSV; only use what's already in data/raw/",
    )
    args = parser.parse_args()

    dataset = build_dataset(attempt_download=not args.no_download, force_download=args.refresh)

    if dataset.empty:
        print("No data parsed -- nothing written.", file=sys.stderr)
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_PATH, index=False)
    print(f"Wrote {len(dataset):,} matches across {dataset['league'].nunique()} league(s) to {PROCESSED_PATH}")
    print(dataset.groupby("league")["season"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
