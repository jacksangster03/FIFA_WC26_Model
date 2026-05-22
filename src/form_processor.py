"""
Form feature processor.

Reads data/external/international_results.csv, filters to historical
completed matches only (before the 2026 World Cup), applies tournament
weights, and produces:
  - data/processed/team_form_features.csv
  - data/processed/team_ratings_final.csv  (seeded + form adjustment)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXTERNAL_FILES,
    PROCESSED_DIR,
    HOST_COUNTRIES,
    HOST_BOOST,
)
from src.ratings import SEEDED_RATINGS, get_rating

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WC_CUTOFF_DATE = pd.Timestamp("2026-06-11")   # day of first WC group match

TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup":                    1.4,
    "FIFA World Cup qualification":      1.2,
    "UEFA Euro":                         1.1,
    "Copa América":                      1.1,
    "African Cup of Nations":            1.1,
    "AFC Asian Cup":                     1.1,
    "Gold Cup":                          1.1,
    "CONCACAF Gold Cup":                 1.1,
    "UEFA Nations League":               0.8,
    "UEFA Nations League A":             0.8,
    "CONCACAF Nations League":           0.8,
    "Friendly":                          0.4,
    "International Friendlies":          0.4,
}
DEFAULT_TOURNAMENT_WEIGHT = 0.7

# Max form adjustment (rating points) from form blending
FORM_BLEND_MAX = 3.0

# How many recent matches to use
FORM_WINDOW_LONG  = 10
FORM_WINDOW_SHORT = 5

# ---------------------------------------------------------------------------
# Team name normalisation
# The Kaggle dataset uses different names for some nations.
# Map dataset name -> our model name.
# ---------------------------------------------------------------------------
TEAM_NAME_MAP: dict[str, str] = {
    # CONMEBOL / CONCACAF
    "United States":                "USA",
    "Trinidad and Tobago":          "Trinidad and Tobago",
    # UEFA
    "Bosnia and Herzegovina":       "Bosnia and Herzegovina",
    # CAF
    "Ivory Coast":                  "Côte d'Ivoire",
    "Cape Verde":                   "Cabo Verde",
    "DR Congo":                     "DR Congo",
    # AFC
    "Curacao":                      "Curaçao",
    "North Korea":                  "North Korea",
    # Misc
    "Czech Republic":               "Czechia",
    "Kyrgyz Republic":              "Kyrgyzstan",
    "Republic of Ireland":          "Republic of Ireland",
    "Faroe Islands":                "Faroe Islands",
}


def _normalise_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Main pipeline functions
# ---------------------------------------------------------------------------

def load_and_filter_results() -> pd.DataFrame:
    """
    Loads international_results.csv, drops future rows, drops rows without
    completed scores, normalises team names, and returns a clean DataFrame.
    """
    path = EXTERNAL_FILES["international_results"]
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Drop future / unplayed rows (any score NaN)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Drop matches from the World Cup itself (2026-06-11 onwards)
    df = df[df["date"] < WC_CUTOFF_DATE]

    # Normalise team names
    df["home_team"] = df["home_team"].map(_normalise_name).fillna(df["home_team"])
    df["away_team"] = df["away_team"].map(_normalise_name).fillna(df["away_team"])

    df = df.sort_values("date").reset_index(drop=True)
    print(f"[form_processor] Loaded {len(df):,} completed matches "
          f"(up to {df['date'].max().date()})")
    return df


def _tournament_weight(tournament: str) -> float:
    if not isinstance(tournament, str):
        return DEFAULT_TOURNAMENT_WEIGHT
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament.lower():
            return weight
    return DEFAULT_TOURNAMENT_WEIGHT


def _team_match_history(
    df: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """
    Returns all matches involving `team`, with columns:
      date, goals_for, goals_against, won, drawn, lost, weight, points
    """
    home = df[df["home_team"] == team].copy()
    home["goals_for"]     = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["is_home"]       = True

    away = df[df["away_team"] == team].copy()
    away["goals_for"]     = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["is_home"]       = False

    combined = pd.concat([home, away], ignore_index=True)
    combined["weight"] = combined["tournament"].apply(_tournament_weight)
    combined["won"]  = (combined["goals_for"] > combined["goals_against"]).astype(int)
    combined["drawn"]= (combined["goals_for"] == combined["goals_against"]).astype(int)
    combined["lost"] = (combined["goals_for"] < combined["goals_against"]).astype(int)
    combined["points"] = combined["won"] * 3 + combined["drawn"]

    return combined.sort_values("date").reset_index(drop=True)


def _compute_form(history: pd.DataFrame, window: int) -> dict:
    """
    Computes form metrics over the last `window` matches.
    Returns a dict of metrics including recency-weighted and opponent-adjusted stats.
    """
    recent = history.tail(window)
    if recent.empty:
        return {
            "matches_used":               0,
            "latest_match_date":          None,
            "ppg":                        None,
            "goals_for":                  None,
            "goals_against":              None,
            "goal_difference":            None,
            "competitive_weighted_ppg":   None,
            "recency_weighted_ppg":       None,
            "recency_weighted_goal_diff": None,
            "opponent_adjusted_form":     None,
            "competitive_recency_score":  None,
        }

    n = len(recent)
    # Exponential recency weights (most recent match has weight 1, oldest ~ 0.5)
    decay = 0.85
    recency_w = np.array([decay ** (n - 1 - i) for i in range(n)])
    recency_w = recency_w / recency_w.sum()

    pts_arr = recent["points"].values.astype(float)
    gd_arr  = (recent["goals_for"] - recent["goals_against"]).values.astype(float)
    tw_arr  = recent["weight"].values.astype(float)

    # Competitive tournament-weighted PPG
    w_pts   = (pts_arr * tw_arr).sum()
    w_total = tw_arr.sum()
    comp_ppg = w_pts / w_total if w_total > 0 else None

    # Recency-weighted PPG and GD
    rec_ppg = float(np.dot(recency_w, pts_arr))
    rec_gd  = float(np.dot(recency_w, gd_arr))

    # Competitive recency: combine tournament weight and recency decay
    combined_w = recency_w * tw_arr
    combined_w_sum = combined_w.sum()
    comp_rec = float(np.dot(combined_w / combined_w_sum, pts_arr)) if combined_w_sum > 0 else comp_ppg

    # Opponent-adjusted form: simple boost if won against neutral/stronger opponents
    # (we don't have opponent ratings here, so approximate via tournament importance)
    opp_adj = comp_rec  # best approximation without opponent lookup

    return {
        "matches_used":               n,
        "latest_match_date":          recent["date"].max().date(),
        "ppg":                        float(np.mean(pts_arr)),
        "goals_for":                  int(recent["goals_for"].sum()),
        "goals_against":              int(recent["goals_against"].sum()),
        "goal_difference":            int(recent["goals_for"].sum() - recent["goals_against"].sum()),
        "competitive_weighted_ppg":   float(comp_ppg) if comp_ppg is not None else None,
        "recency_weighted_ppg":       round(rec_ppg, 4),
        "recency_weighted_goal_diff": round(rec_gd, 4),
        "opponent_adjusted_form":     round(opp_adj, 4),
        "competitive_recency_score":  round(comp_rec, 4),
    }


def compute_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes form features for every team appearing in the dataset.
    Returns a DataFrame saved to data/processed/team_form_features.csv.
    """
    all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    rows = []

    for team in all_teams:
        history = _team_match_history(df, team)
        if history.empty:
            continue

        short = _compute_form(history, FORM_WINDOW_SHORT)
        long  = _compute_form(history, FORM_WINDOW_LONG)

        def _r(v): return round(v, 3) if v is not None else None

        rows.append({
            "team":                        team,
            "total_matches_in_dataset":    len(history),
            "matches_used":                long["matches_used"],
            "latest_match_date":           long["latest_match_date"],
            "last_5_points_per_game":      _r(short["ppg"]),
            "last_10_points_per_game":     _r(long["ppg"]),
            "last_10_goal_difference":     long["goal_difference"],
            "last_10_goals_for":           long["goals_for"],
            "last_10_goals_against":       long["goals_against"],
            "competitive_weighted_form":   _r(long["competitive_weighted_ppg"]),
            "recency_weighted_ppg":        _r(long["recency_weighted_ppg"]),
            "recency_weighted_goal_diff":  _r(long["recency_weighted_goal_diff"]),
            "opponent_adjusted_form":      _r(long["opponent_adjusted_form"]),
            "competitive_recency_score":   _r(long["competitive_recency_score"]),
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "team_form_features.csv"
    out.to_csv(out_path, index=False)
    print(f"[form_processor] Saved form features for {len(out)} teams -> {out_path}")
    return out


def _form_score_for_blend(form_lookup: dict[str, float | None], team: str) -> float | None:
    """
    Converts competitive_recency_score (a PPG-like value in 0-3) to a 0-100 scale
    for use as a rating blend component.  Formula: score / 3 * 100.
    Returns None if no form data available.
    """
    val = form_lookup.get(team)
    if val is None:
        return None
    return float(np.clip(val / 3.0 * 100.0, 0.0, 100.0))


def run_form_processing(seeded_ratings: dict[str, float]) -> dict[str, float]:
    """
    Top-level entry point. Processes historical data, runs the full source-driven
    rating blend, saves processed files, and returns the final ratings dict.

    Blend (see ratings.blend_all_sources):
      FIFA 35% | Elo 25% | Form 15% | Pedigree 10% | Seeded 10% | Host 5%
    Missing sources are reweighted proportionally.
    """
    from src.data_sources import playoff_resolver
    from src.ratings import blend_all_sources, SEEDED_RATINGS as _SEEDS

    df = load_and_filter_results()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("[form_processor] No historical data available; using seeded ratings only")
        _save_seeded_only(seeded_ratings)
        return seeded_ratings

    # 1. Compute form features
    form_df = compute_form_features(df)

    # 2. Build form lookup for the blend (competitive_recency_score -> 0-100)
    form_raw = form_df.set_index("team")["competitive_recency_score"].to_dict()
    form_lookup = {t: _form_score_for_blend(form_raw, t) for t in form_raw}

    # 3. Determine the full team list.
    # IMPORTANT: restrict to WC competition teams only (seeded + resolved playoff teams).
    # Do NOT include every team from the form dataset (333 non-WC teams would corrupt ratings).
    resolved = playoff_resolver.get_resolved_teams()  # {placeholder: canonical}
    canonical_teams = list(resolved.values())
    all_teams = sorted(set(_SEEDS) | set(canonical_teams))

    # 4. Source-driven blend
    ratings, details_df = blend_all_sources(all_teams, form_lookup)

    # 4b. Apply capped manual adjustments (safe to skip if file absent)
    from src.ratings import apply_manual_adjustments
    ratings, details_df = apply_manual_adjustments(ratings, details_df)

    # 5. Merge in form detail columns for the output CSV
    form_index = form_df.set_index("team")
    extra_cols = ["matches_used", "latest_match_date", "last_5_points_per_game",
                  "last_10_points_per_game", "last_10_goal_difference",
                  "last_10_goals_for", "last_10_goals_against",
                  "competitive_weighted_form", "recency_weighted_ppg",
                  "recency_weighted_goal_diff", "opponent_adjusted_form",
                  "competitive_recency_score"]

    for col in extra_cols:
        details_df[col] = details_df["team"].map(
            lambda t, c=col: form_index.loc[t, c]
            if t in form_index.index and c in form_index.columns else None
        )

    out_path = PROCESSED_DIR / "team_ratings_final.csv"
    details_df.to_csv(out_path, index=False)
    print(f"[form_processor] Saved final ratings for {len(details_df)} teams -> {out_path}")

    return ratings


def _save_seeded_only(seeded_ratings: dict[str, float]) -> None:
    from src.ratings import SEEDED_RATINGS as _SEEDS
    rows = []
    for team in sorted(seeded_ratings, key=lambda t: -seeded_ratings[t]):
        rows.append({
            "team":          team,
            "final_rating":  round(seeded_ratings[team] + (HOST_BOOST if team in HOST_COUNTRIES else 0.0), 2),
            "seeded_component": seeded_ratings.get(team, 55.0),
            "host_component":   HOST_BOOST if team in HOST_COUNTRIES else 0.0,
            "sources_used":     "seeded_fallback",
            "rating_explanation": f"seeded={seeded_ratings.get(team, 55.0)}"
                                  + (f" | host+{HOST_BOOST}" if team in HOST_COUNTRIES else ""),
        })
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "team_ratings_final.csv", index=False)
