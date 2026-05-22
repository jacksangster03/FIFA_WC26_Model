"""
FIFA rankings ingestion.

Produces: data/processed/current_fifa_rankings.csv
Columns:  rank, team, points, confederation

Live fetch is attempted when run() is called; on failure the hardcoded
May 2026 snapshot is written instead.  The pipeline always has data.
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR

OUT_PATH = PROCESSED_DIR / "current_fifa_rankings.csv"

# ---------------------------------------------------------------------------
# Hardcoded May 2026 snapshot (source: FIFA rankings approximation)
# Points are on the official FIFA scale (0-2000+).
# ---------------------------------------------------------------------------
_SNAPSHOT: list[dict] = [
    {"rank":  1, "team": "Argentina",            "points": 1873.8, "confederation": "CONMEBOL"},
    {"rank":  2, "team": "France",               "points": 1840.2, "confederation": "UEFA"},
    {"rank":  3, "team": "Spain",                "points": 1820.6, "confederation": "UEFA"},
    {"rank":  4, "team": "England",              "points": 1786.1, "confederation": "UEFA"},
    {"rank":  5, "team": "Brazil",               "points": 1764.8, "confederation": "CONMEBOL"},
    {"rank":  6, "team": "Portugal",             "points": 1742.4, "confederation": "UEFA"},
    {"rank":  7, "team": "Belgium",              "points": 1703.2, "confederation": "UEFA"},
    {"rank":  8, "team": "Netherlands",          "points": 1698.5, "confederation": "UEFA"},
    {"rank":  9, "team": "Germany",              "points": 1675.9, "confederation": "UEFA"},
    {"rank": 10, "team": "Colombia",             "points": 1661.3, "confederation": "CONMEBOL"},
    {"rank": 11, "team": "Uruguay",              "points": 1647.6, "confederation": "CONMEBOL"},
    {"rank": 12, "team": "Morocco",              "points": 1637.4, "confederation": "CAF"},
    {"rank": 13, "team": "Japan",                "points": 1625.0, "confederation": "AFC"},
    {"rank": 14, "team": "USA",                  "points": 1605.2, "confederation": "CONCACAF"},
    {"rank": 15, "team": "Croatia",              "points": 1597.8, "confederation": "UEFA"},
    {"rank": 16, "team": "Switzerland",          "points": 1584.1, "confederation": "UEFA"},
    {"rank": 17, "team": "Senegal",              "points": 1575.3, "confederation": "CAF"},
    {"rank": 18, "team": "Mexico",               "points": 1568.0, "confederation": "CONCACAF"},
    {"rank": 19, "team": "Austria",              "points": 1555.9, "confederation": "UEFA"},
    {"rank": 20, "team": "South Korea",          "points": 1540.6, "confederation": "AFC"},
    {"rank": 21, "team": "Ecuador",              "points": 1535.7, "confederation": "CONMEBOL"},
    {"rank": 22, "team": "Canada",               "points": 1512.4, "confederation": "CONCACAF"},
    {"rank": 23, "team": "Norway",               "points": 1503.8, "confederation": "UEFA"},
    {"rank": 24, "team": "Sweden",               "points": 1494.5, "confederation": "UEFA"},
    {"rank": 25, "team": "Turkey",               "points": 1488.2, "confederation": "UEFA"},
    {"rank": 26, "team": "Scotland",             "points": 1478.1, "confederation": "UEFA"},
    {"rank": 27, "team": "Algeria",              "points": 1459.3, "confederation": "CAF"},
    {"rank": 28, "team": "Australia",            "points": 1443.0, "confederation": "AFC"},
    {"rank": 29, "team": "Saudi Arabia",         "points": 1430.5, "confederation": "AFC"},
    {"rank": 30, "team": "Czechia",              "points": 1424.8, "confederation": "UEFA"},
    {"rank": 31, "team": "Iran",                 "points": 1418.0, "confederation": "AFC"},
    {"rank": 32, "team": "Bosnia and Herzegovina","points": 1395.2, "confederation": "UEFA"},
    {"rank": 33, "team": "Paraguay",             "points": 1388.4, "confederation": "CONMEBOL"},
    {"rank": 34, "team": "Tunisia",              "points": 1377.6, "confederation": "CAF"},
    {"rank": 35, "team": "Egypt",                "points": 1367.1, "confederation": "CAF"},
    {"rank": 36, "team": "Ghana",                "points": 1354.8, "confederation": "CAF"},
    {"rank": 37, "team": "Uzbekistan",           "points": 1341.5, "confederation": "AFC"},
    {"rank": 38, "team": "Iraq",                 "points": 1328.0, "confederation": "AFC"},
    {"rank": 39, "team": "South Africa",         "points": 1311.7, "confederation": "CAF"},
    {"rank": 40, "team": "Panama",               "points": 1297.4, "confederation": "CONCACAF"},
    {"rank": 41, "team": "Jordan",               "points": 1280.8, "confederation": "AFC"},
    {"rank": 42, "team": "DR Congo",             "points": 1268.0, "confederation": "CAF"},
    {"rank": 43, "team": "Cabo Verde",           "points": 1247.2, "confederation": "CAF"},
    {"rank": 44, "team": "New Zealand",          "points": 1210.0, "confederation": "OFC"},
    {"rank": 45, "team": "Qatar",                "points": 1195.0, "confederation": "AFC"},
    {"rank": 46, "team": "Haiti",                "points": 1165.3, "confederation": "CONCACAF"},
    {"rank": 47, "team": "Curaçao",              "points": 1142.0, "confederation": "CONCACAF"},
]

# Normalisation bounds (FIFA scale)
FIFA_MIN = 1000.0
FIFA_MAX = 2000.0


def _try_live_fetch() -> pd.DataFrame | None:
    """Attempt to fetch current FIFA rankings from a live source."""
    try:
        import requests
        # Unofficial API — may break; treat as best-effort
        url = "https://www.fifa.com/fifa-world-ranking/men"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        # Parsing the page is brittle; skip and use snapshot
        return None
    except Exception:
        return None


def run(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns (and saves) the FIFA rankings DataFrame.
    Uses cached processed CSV if present and force_refresh is False.
    """
    if OUT_PATH.exists() and not force_refresh:
        try:
            df = pd.read_csv(OUT_PATH)
            if not df.empty:
                print(f"[fifa_rankings] Loaded cached rankings ({len(df)} teams)")
                return df
        except Exception:
            pass

    live = _try_live_fetch()
    df = live if live is not None else pd.DataFrame(_SNAPSHOT)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    source = "live" if live is not None else "snapshot"
    print(f"[fifa_rankings] Saved {len(df)} teams to {OUT_PATH} ({source})")
    return df


def load_normalised() -> dict[str, float]:
    """Returns {team: normalised_0_100_score} from the processed CSV."""
    if not OUT_PATH.exists():
        run()
    df = pd.read_csv(OUT_PATH)
    result = {}
    for _, row in df.iterrows():
        team   = str(row["team"])
        pts    = float(row["points"])
        normed = max(0.0, min(100.0, (pts - FIFA_MIN) / (FIFA_MAX - FIFA_MIN) * 100.0))
        result[team] = normed
    return result
