"""
Optional external data loader.

If the files in data/external/ exist they are loaded; otherwise the pipeline
falls back to the seeded ratings in ratings.py.  If online mode is enabled
and requests is available, download hooks are called first.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import EXTERNAL_FILES

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_external_data(online: bool = False) -> dict[str, Optional[pd.DataFrame]]:
    """
    Returns a dict keyed by dataset name.  Values are DataFrames or None.
    """
    if online:
        _try_downloads()

    result: dict[str, Optional[pd.DataFrame]] = {}
    for name, path in EXTERNAL_FILES.items():
        result[name] = _load_csv_safe(path, name)

    loaded = [k for k, v in result.items() if v is not None]
    missing = [k for k, v in result.items() if v is None]

    if loaded:
        print(f"[external_data] Loaded: {loaded}")
    if missing:
        print(f"[external_data] Not found (using seeded fallback): {missing}")

    return result


# ---------------------------------------------------------------------------
# Download hooks (optional; safe to fail)
# ---------------------------------------------------------------------------

def _try_downloads() -> None:
    try:
        import requests  # noqa: F401
    except ImportError:
        warnings.warn("[external_data] requests not installed; skipping downloads.")
        return

    _download_elo_ratings()


def _download_elo_ratings() -> None:
    """
    World Football Elo ratings from eloratings.net (CSV endpoint).
    Replace URL with an authoritative source if available.
    """
    url = "https://www.eloratings.net/World.tsv"
    dest = EXTERNAL_FILES["world_football_elo"]
    if dest.exists():
        return
    try:
        import requests
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(r.text)
        print(f"[external_data] Downloaded Elo ratings to {dest}")
    except Exception as exc:
        print(f"[external_data] Could not download Elo ratings: {exc}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_csv_safe(path: Path, name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as exc:
        warnings.warn(f"[external_data] Failed to load {name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Parsers for specific external datasets (called from ratings.py)
# ---------------------------------------------------------------------------

def parse_elo_ratings(df: Optional[pd.DataFrame]) -> dict[str, float]:
    """
    Expects columns: team, elo  (or country, rating).
    Returns {team_name: elo_value}.
    """
    if df is None:
        return {}
    for team_col in ("team", "country", "Team", "Country"):
        for elo_col in ("elo", "rating", "Elo", "Rating"):
            if team_col in df.columns and elo_col in df.columns:
                return dict(zip(df[team_col].astype(str), df[elo_col].astype(float)))
    return {}


def parse_fifa_rankings(df: Optional[pd.DataFrame]) -> dict[str, float]:
    """
    Expects columns: team, points  (or country, total_points).
    Returns {team_name: fifa_points}.
    """
    if df is None:
        return {}
    for team_col in ("team", "country", "Team", "Country"):
        for pts_col in ("points", "total_points", "Points"):
            if team_col in df.columns and pts_col in df.columns:
                return dict(zip(df[team_col].astype(str), df[pts_col].astype(float)))
    return {}


def parse_squad_values(df: Optional[pd.DataFrame]) -> dict[str, float]:
    """
    Expects columns: team, value_eur (or similar).
    Returns {team_name: squad_value_EUR}.
    """
    if df is None:
        return {}
    for team_col in ("team", "country", "Team"):
        for val_col in ("value_eur", "squad_value", "value", "Value"):
            if team_col in df.columns and val_col in df.columns:
                return dict(zip(df[team_col].astype(str), df[val_col].astype(float)))
    return {}
