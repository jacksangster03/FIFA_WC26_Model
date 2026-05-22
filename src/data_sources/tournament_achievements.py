"""
Tournament pedigree scores.

Produces: data/processed/tournament_pedigree_scores.csv
Columns:  team, wc_titles, wc_finals, wc_semis, wc_quarters, pedigree_score

Score formula (0-100):
  wc_titles  * 20 pts  (capped at 100)
  wc_finals  * 8 pts
  wc_semis   * 4 pts
  wc_quarters* 1.5 pts
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR

OUT_PATH = PROCESSED_DIR / "tournament_pedigree_scores.csv"

# ---------------------------------------------------------------------------
# Historical World Cup data (all editions through 2022)
# ---------------------------------------------------------------------------
_PEDIGREE: list[dict] = [
    # team, titles, finals (incl. titles), semi (incl. finals), quarters (incl. semis)
    {"team": "Brazil",               "wc_titles": 5, "wc_finals": 7, "wc_semis": 11, "wc_quarters": 16},
    {"team": "Germany",              "wc_titles": 4, "wc_finals": 8, "wc_semis": 13, "wc_quarters": 16},
    {"team": "Italy",                "wc_titles": 4, "wc_finals": 6, "wc_semis": 8,  "wc_quarters": 13},
    {"team": "Argentina",            "wc_titles": 3, "wc_finals": 6, "wc_semis": 7,  "wc_quarters": 12},
    {"team": "France",               "wc_titles": 2, "wc_finals": 3, "wc_semis": 6,  "wc_quarters": 9},
    {"team": "Uruguay",              "wc_titles": 2, "wc_finals": 3, "wc_semis": 4,  "wc_quarters": 8},
    {"team": "England",              "wc_titles": 1, "wc_finals": 1, "wc_semis": 3,  "wc_quarters": 8},
    {"team": "Spain",                "wc_titles": 1, "wc_finals": 1, "wc_semis": 2,  "wc_quarters": 6},
    {"team": "Netherlands",          "wc_titles": 0, "wc_finals": 3, "wc_semis": 5,  "wc_quarters": 8},
    {"team": "Croatia",              "wc_titles": 0, "wc_finals": 1, "wc_semis": 2,  "wc_quarters": 4},
    {"team": "Sweden",               "wc_titles": 0, "wc_finals": 1, "wc_semis": 4,  "wc_quarters": 7},
    {"team": "Portugal",             "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 4},
    {"team": "Belgium",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 4},
    {"team": "Mexico",               "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 8},
    {"team": "South Korea",          "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 2},
    {"team": "USA",                  "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 2},
    {"team": "Turkey",               "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 1},
    {"team": "Morocco",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 2},
    {"team": "Austria",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 2,  "wc_quarters": 4},
    {"team": "Japan",                "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 2},
    {"team": "Ecuador",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Australia",            "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 1},
    {"team": "Colombia",             "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Senegal",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Paraguay",             "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 2},
    {"team": "Switzerland",          "wc_titles": 0, "wc_finals": 0, "wc_semis": 1,  "wc_quarters": 4},
    {"team": "Canada",               "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Norway",               "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Scotland",             "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Algeria",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Saudi Arabia",         "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Iran",                 "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Ghana",                "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Egypt",                "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Tunisia",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Uzbekistan",           "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "South Africa",         "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Panama",               "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Jordan",               "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Cabo Verde",           "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "New Zealand",          "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Haiti",                "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Curaçao",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Qatar",                "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
    {"team": "Bosnia and Herzegovina","wc_titles": 0, "wc_finals": 0, "wc_semis": 0, "wc_quarters": 0},
    {"team": "Czechia",              "wc_titles": 0, "wc_finals": 0, "wc_semis": 2,  "wc_quarters": 4},
    {"team": "DR Congo",             "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 1},
    {"team": "Iraq",                 "wc_titles": 0, "wc_finals": 0, "wc_semis": 0,  "wc_quarters": 0},
]


def _score(row: dict) -> float:
    raw = (row["wc_titles"]  * 20.0 +
           row["wc_finals"]  *  8.0 +
           row["wc_semis"]   *  4.0 +
           row["wc_quarters"]*  1.5)
    return min(100.0, raw)


def run(force_refresh: bool = False) -> pd.DataFrame:
    """Builds (and saves) pedigree scores. Always uses hardcoded data."""
    if OUT_PATH.exists() and not force_refresh:
        try:
            df = pd.read_csv(OUT_PATH)
            if not df.empty:
                print(f"[tournament_achievements] Loaded cached pedigree ({len(df)} teams)")
                return df
        except Exception:
            pass

    rows = []
    for entry in _PEDIGREE:
        rows.append({
            "team":           entry["team"],
            "wc_titles":      entry["wc_titles"],
            "wc_finals":      entry["wc_finals"],
            "wc_semis":       entry["wc_semis"],
            "wc_quarters":    entry["wc_quarters"],
            "pedigree_score": round(_score(entry), 2),
        })

    df = pd.DataFrame(rows).sort_values("pedigree_score", ascending=False).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[tournament_achievements] Saved pedigree for {len(df)} teams -> {OUT_PATH}")
    return df


def load_normalised() -> dict[str, float]:
    """Returns {team: pedigree_score_0_100}."""
    if not OUT_PATH.exists():
        run()
    df = pd.read_csv(OUT_PATH)
    return dict(zip(df["team"], df["pedigree_score"].astype(float)))
