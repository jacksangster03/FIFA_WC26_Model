"""
Bracket difficulty analysis and calibration diagnostics.

Provides:
  knockout_win_probability(rating_a, rating_b, stage) -> float
      Logistic win probability with stage-scaled uncertainty.
      Used for diagnostics and simulation win-probability display.

  generate_bracket_diagnostics(knockout_preds, ratings) -> pd.DataFrame
      Produces outputs/bracket_path_diagnostics.csv.

  generate_xg_diagnostics(ratings) -> pd.DataFrame
      Tests specific elite matchups against target xG-gap ranges.
      Produces outputs/xg_calibration_diagnostics.csv.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, KNOCKOUT_PROBABILITY_SCALES
from src.goal_model import expected_goals

# ---------------------------------------------------------------------------
# Stage-scaled logistic win probability
# ---------------------------------------------------------------------------

def knockout_win_probability(rating_a: float, rating_b: float, stage: str) -> float:
    """
    Logistic win probability for team A vs team B in a given knockout stage.

    Scale increases with stage importance to reflect greater uncertainty.
    Result is clipped to [0.20, 0.80] so even the largest favourite cannot
    exceed 80% win probability in a single knockout match.

    Expected outputs:
      2-pt gap  -> ~52-53%
      5-pt gap  -> ~56-58%
      10-pt gap -> ~62-65%
      15-pt gap -> ~68-72%
    """
    diff  = rating_a - rating_b
    scale = KNOCKOUT_PROBABILITY_SCALES.get(stage, 17.0)
    p     = 1.0 / (1.0 + math.exp(-diff / scale))
    return float(np.clip(p, 0.20, 0.80))


# ---------------------------------------------------------------------------
# Bracket path diagnostics (deterministic)
# ---------------------------------------------------------------------------

def generate_bracket_diagnostics(
    knockout_preds: pd.DataFrame,
    ratings: dict[str, float],
) -> pd.DataFrame:
    """
    For each knockout match, compute model win probabilities and annotate
    the deterministic result.  Saves outputs/bracket_path_diagnostics.csv.
    """
    rows = []
    for _, row in knockout_preds.iterrows():
        home = str(row.get("predicted_home_team", ""))
        away = str(row.get("predicted_away_team", ""))
        rnd  = str(row.get("round", ""))

        if not home or not away:
            continue

        hr  = ratings.get(home, 55.0)
        ar  = ratings.get(away, 55.0)
        diff = hr - ar

        p_home = knockout_win_probability(hr, ar, rnd)
        p_away = 1.0 - p_home
        det_pen_winner = "home" if p_home >= 0.5 else "away"

        winner_side = str(row.get("match_winner", ""))
        winner = home if winner_side == "home" else away
        loser  = away if winner_side == "home" else home

        pens = bool(row.get("penalties", False))
        hg   = row.get("predicted_home_goals", "")
        ag   = row.get("predicted_away_goals", "")

        stage_draw_risk = (
            "high" if abs(diff) <= 3 else
            "medium" if abs(diff) <= 8 else
            "low"
        )
        note_parts = []
        if abs(diff) < 3:
            note_parts.append("coinflip")
        elif abs(diff) >= 15:
            note_parts.append("heavy-favourite")
        if pens:
            note_parts.append("pens")

        rows.append({
            "round":               rnd,
            "match_id":            int(row.get("match_id", 0)),
            "home_team":           home,
            "away_team":           away,
            "home_rating":         round(hr, 2),
            "away_rating":         round(ar, 2),
            "rating_diff":         round(diff, 2),
            "home_win_prob_model": round(p_home, 3),
            "away_win_prob_model": round(p_away, 3),
            "home_penalty_win_prob": round(p_home, 3),
            "away_penalty_win_prob": round(p_away, 3),
            "predicted_score":     f"{hg}-{ag}",
            "winner":              winner,
            "loser":               loser,
            "penalties":           pens,
            "deterministic_penalty_winner": det_pen_winner if pens else "",
            "deterministic_winner_consistent_with_prob": (
                (winner_side == "home" and p_home >= 0.5) or
                (winner_side == "away" and p_home <= 0.5)
            ),
            "stage_draw_risk":     stage_draw_risk,
            "path_difficulty_note": "; ".join(note_parts) if note_parts else "normal",
        })

    df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "bracket_path_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[bracket_difficulty] Bracket diagnostics saved -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# xG calibration diagnostics
# ---------------------------------------------------------------------------

_TEST_MATCHUPS = [
    ("France",      "Spain"),
    ("France",      "England"),
    ("France",      "Argentina"),
    ("France",      "Brazil"),
    ("Spain",       "Argentina"),
    ("Spain",       "England"),
    ("Portugal",    "Germany"),
    ("Belgium",     "France"),
    ("Belgium",     "Netherlands"),
    ("Uruguay",     "Portugal"),
]

# Target xG-diff bands by rating-diff bucket
_XG_TARGET_BANDS: list[tuple[float, float, float, float]] = [
    # (min_diff, max_diff, target_min_xg_diff, target_max_xg_diff)
    (0,  3,  0.00, 0.12),
    (3,  7,  0.12, 0.25),
    (7,  12, 0.25, 0.45),
    (12, 20, 0.40, 0.65),
]


def _xg_target(rating_diff: float) -> tuple[float, float]:
    """Return (min, max) expected xG-diff for a given rating gap."""
    for mn, mx, tmin, tmax in _XG_TARGET_BANDS:
        if mn <= abs(rating_diff) < mx:
            return tmin, tmax
    return 0.40, 0.70  # large gap


def generate_xg_diagnostics(
    ratings: dict[str, float],
    stage: str = "group",
) -> pd.DataFrame:
    """
    Tests elite matchups against target xG-gap ranges.
    Saves outputs/xg_calibration_diagnostics.csv.
    """
    rows = []
    for team_a, team_b in _TEST_MATCHUPS:
        ra = ratings.get(team_a, 55.0)
        rb = ratings.get(team_b, 55.0)
        diff = ra - rb

        xg_a, xg_b = expected_goals(team_a, team_b, ratings, stage=stage)
        xg_diff = xg_a - xg_b

        # Compute implicit win probability from xG via Poisson
        from scipy.stats import poisson
        max_g = 8
        pa = np.array([poisson.pmf(i, xg_a) for i in range(max_g)])
        pb = np.array([poisson.pmf(j, xg_b) for j in range(max_g)])
        mat = np.outer(pa, pb)
        p_a_wins   = float(np.tril(mat, k=-1).sum())
        p_b_wins   = float(np.triu(mat, k=1).sum())
        p_draw     = float(np.diag(mat).sum())

        tmin, tmax = _xg_target(diff)
        in_band = tmin <= abs(xg_diff) <= tmax

        rows.append({
            "match_context":    f"{team_a} vs {team_b} ({stage})",
            "team_a":           team_a,
            "team_b":           team_b,
            "rating_a":         round(ra, 2),
            "rating_b":         round(rb, 2),
            "rating_diff":      round(diff, 2),
            "xg_a":             round(xg_a, 3),
            "xg_b":             round(xg_b, 3),
            "xg_diff":          round(xg_diff, 3),
            "model_win_prob_a": round(p_a_wins, 3),
            "model_win_prob_b": round(p_b_wins, 3),
            "draw_prob":        round(p_draw, 3),
            "xg_target_min":    tmin,
            "xg_target_max":    tmax,
            "xg_in_target_band": in_band,
        })

    df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "xg_calibration_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[bracket_difficulty] xG diagnostics saved -> {out_path}")
    return df
