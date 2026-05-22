"""
World Football Elo ratings ingestion.

Produces: data/processed/current_world_elo.csv
Columns:  rank, team, elo

Live fetch is attempted first; on failure the hardcoded May 2026
snapshot is used.  The pipeline always has data.
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR

OUT_PATH = PROCESSED_DIR / "current_world_elo.csv"

# ---------------------------------------------------------------------------
# Hardcoded May 2026 snapshot (source: World Football Elo approximation)
# ---------------------------------------------------------------------------
_SNAPSHOT: list[dict] = [
    {"rank":  1, "team": "France",               "elo": 2072},
    {"rank":  2, "team": "Argentina",            "elo": 2045},
    {"rank":  3, "team": "Brazil",               "elo": 2038},
    {"rank":  4, "team": "Spain",                "elo": 2032},
    {"rank":  5, "team": "England",              "elo": 2016},
    {"rank":  6, "team": "Germany",              "elo": 1994},
    {"rank":  7, "team": "Portugal",             "elo": 1980},
    {"rank":  8, "team": "Netherlands",          "elo": 1962},
    {"rank":  9, "team": "Belgium",              "elo": 1942},
    {"rank": 10, "team": "Uruguay",              "elo": 1908},
    {"rank": 11, "team": "Colombia",             "elo": 1886},
    {"rank": 12, "team": "Morocco",              "elo": 1865},
    {"rank": 13, "team": "Mexico",               "elo": 1852},
    {"rank": 14, "team": "Japan",                "elo": 1843},
    {"rank": 15, "team": "USA",                  "elo": 1834},
    {"rank": 16, "team": "Canada",               "elo": 1818},
    {"rank": 17, "team": "Switzerland",          "elo": 1806},
    {"rank": 18, "team": "Croatia",              "elo": 1800},
    {"rank": 19, "team": "South Korea",          "elo": 1788},
    {"rank": 20, "team": "Austria",              "elo": 1774},
    {"rank": 21, "team": "Senegal",              "elo": 1757},
    {"rank": 22, "team": "Ecuador",              "elo": 1742},
    {"rank": 23, "team": "Norway",               "elo": 1736},
    {"rank": 24, "team": "Sweden",               "elo": 1724},
    {"rank": 25, "team": "Bosnia and Herzegovina","elo": 1712},
    {"rank": 26, "team": "Turkey",               "elo": 1702},
    {"rank": 27, "team": "Scotland",             "elo": 1695},
    {"rank": 28, "team": "Algeria",              "elo": 1682},
    {"rank": 29, "team": "Czechia",              "elo": 1668},
    {"rank": 30, "team": "Australia",            "elo": 1655},
    {"rank": 31, "team": "Iraq",                 "elo": 1642},
    {"rank": 32, "team": "Saudi Arabia",         "elo": 1635},
    {"rank": 33, "team": "Egypt",                "elo": 1622},
    {"rank": 34, "team": "Iran",                 "elo": 1615},
    {"rank": 35, "team": "Paraguay",             "elo": 1608},
    {"rank": 36, "team": "Tunisia",              "elo": 1592},
    {"rank": 37, "team": "Ghana",                "elo": 1578},
    {"rank": 38, "team": "DR Congo",             "elo": 1564},
    {"rank": 39, "team": "Uzbekistan",           "elo": 1553},
    {"rank": 40, "team": "South Africa",         "elo": 1542},
    {"rank": 41, "team": "Panama",               "elo": 1525},
    {"rank": 42, "team": "Jordan",               "elo": 1508},
    {"rank": 43, "team": "Cabo Verde",           "elo": 1485},
    {"rank": 44, "team": "Qatar",                "elo": 1470},
    {"rank": 45, "team": "New Zealand",          "elo": 1462},
    {"rank": 46, "team": "Haiti",                "elo": 1435},
    {"rank": 47, "team": "Curaçao",              "elo": 1418},
]

# Normalisation bounds
ELO_MIN = 1300.0
ELO_MAX = 2150.0


def _try_live_fetch() -> pd.DataFrame | None:
    """
    Attempt to download World Elo ratings.

    eloratings.net TSV format: rank<tab>team_name<tab>elo<tab>...
    We validate the result before returning it.
    """
    try:
        import requests
        url = "https://www.eloratings.net/World.tsv"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        from io import StringIO
        raw = pd.read_csv(StringIO(r.text), sep="\t", header=None)
        # Expect at least 3 cols: rank (int), team (str), elo (int)
        if raw.shape[1] < 3:
            return None
        raw.columns = ["rank", "team", "elo"] + list(raw.columns[3:])
        df = raw[["rank", "team", "elo"]].copy()
        df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
        df = df.dropna(subset=["elo"])
        # Sanity: Elo values should be in a realistic range
        if df["elo"].max() < 1500 or df["elo"].min() < 100:
            return None
        # Sanity: team column should contain strings, not numbers
        if pd.to_numeric(df["team"], errors="coerce").notna().all():
            return None
        return df[["rank", "team", "elo"]].reset_index(drop=True)
    except Exception:
        return None


def run(force_refresh: bool = False) -> pd.DataFrame:
    """Returns (and saves) the Elo ratings DataFrame."""
    if OUT_PATH.exists() and not force_refresh:
        try:
            df = pd.read_csv(OUT_PATH)
            if not df.empty:
                print(f"[world_elo] Loaded cached Elo ratings ({len(df)} teams)")
                return df
        except Exception:
            pass

    live = _try_live_fetch()
    df   = live if live is not None else pd.DataFrame(_SNAPSHOT)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    source = "live" if live is not None else "snapshot"
    print(f"[world_elo] Saved {len(df)} teams to {OUT_PATH} ({source})")
    return df


def load_normalised() -> dict[str, float]:
    """Returns {team: normalised_0_100_score} from the processed CSV."""
    if not OUT_PATH.exists():
        run()
    df = pd.read_csv(OUT_PATH)
    result = {}
    for _, row in df.iterrows():
        team   = str(row["team"])
        elo    = float(row["elo"])
        normed = max(0.0, min(100.0, (elo - ELO_MIN) / (ELO_MAX - ELO_MIN) * 100.0))
        result[team] = normed
    return result
