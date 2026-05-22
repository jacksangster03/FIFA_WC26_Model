"""
Expected-goals and scoreline prediction model.

Two operating modes controlled by the `mode` argument to predict_scoreline:

  "deterministic"  (default, used for the submission)
      Step 1 — determine expected outcome using rating difference:
        |diff| <= DRAW_RATING_THRESHOLD  → draw
        diff >  DRAW_RATING_THRESHOLD    → home win
        diff < -DRAW_RATING_THRESHOLD    → away win
      Step 2 — conditional argmax: pick the most likely scoreline
        within the expected outcome region (upper-triangle / diagonal /
        lower-triangle of the joint Poisson PMF).
      This approach gives strong favourites 2-0, 3-0 results while
      still allowing realistic 1-1 draws for near-equal teams, and is
      fully deterministic with no randomness.

  "sample"  (used by Monte Carlo simulation)
      Samples from the raw joint Poisson PMF using a provided numpy
      Generator.  Natural draw rate ~25-27%; no artificial discount.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.stats import poisson

from src.config import (
    BASE_XG,
    HOME_XG_BONUS,
    ACTUAL_HOME_BONUS,
    STRENGTH_SCALE,
    STRENGTH_XG_FACTOR,
    MIN_XG,
    MAX_XG,
    MAX_GOALS_GRID,
    WC_CONSERVATISM,
    KNOCKOUT_XG_FACTORS,
    DRAW_RATING_THRESHOLD_GROUP,
    DRAW_RATING_THRESHOLD_KNOCKOUT,
    KNOCKOUT_PROBABILITY_SCALES,
    PENALTY_DETERMINISTIC_THRESHOLD,
    WINNER_BIAS,
    LOSER_XG_THRESHOLD,
)
from src.features import build_match_features
from src.ratings import get_rating


def expected_goals(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    stage: str = "group",
    venue: str = "",
) -> tuple[float, float]:
    """Returns (home_xg, away_xg) for a match."""
    features    = build_match_features(home_team, away_team, ratings, stage, venue)
    diff        = features["rating_diff"]
    xg_shift    = (diff / STRENGTH_SCALE) * STRENGTH_XG_FACTOR
    venue_bonus = ACTUAL_HOME_BONUS if features["home_is_host"] else HOME_XG_BONUS

    home_xg = BASE_XG + venue_bonus + xg_shift
    away_xg = BASE_XG               - xg_shift

    stage_factor = KNOCKOUT_XG_FACTORS.get(stage, 1.0)
    home_xg = float(np.clip(home_xg * stage_factor * WC_CONSERVATISM, MIN_XG, MAX_XG))
    away_xg = float(np.clip(away_xg * stage_factor * WC_CONSERVATISM, MIN_XG, MAX_XG))

    return home_xg, away_xg


def scoreline_probability_matrix(
    home_xg: float,
    away_xg: float,
    max_goals: int = MAX_GOALS_GRID,
) -> np.ndarray:
    """
    Joint Poisson PMF matrix (no draw discount — used by both modes).
    Row index = home goals, column index = away goals.
    """
    home_probs = np.array([poisson.pmf(i, home_xg) for i in range(max_goals + 1)])
    away_probs = np.array([poisson.pmf(j, away_xg) for j in range(max_goals + 1)])
    return np.outer(home_probs, away_probs)


def _conditional_argmax(matrix: np.ndarray, outcome: str) -> tuple[int, int]:
    """
    Returns the (home_goals, away_goals) tuple that maximises the PMF
    within the requested outcome region.

    outcome:
      "home"  → home_goals > away_goals (strictly below diagonal, row > col)
      "away"  → away_goals > home_goals (strictly above diagonal, row < col)
      "draw"  → home_goals == away_goals (main diagonal only)
    """
    if outcome == "home":
        masked = np.tril(matrix, k=-1)          # row > col
    elif outcome == "away":
        masked = np.triu(matrix, k=1)           # col > row
    else:
        masked = np.diag(np.diag(matrix))       # diagonal only

    if masked.max() == 0:
        # Fallback: unrestricted argmax (should be extremely rare)
        idx = int(np.argmax(matrix.ravel()))
    else:
        idx = int(np.argmax(masked.ravel()))

    return tuple(int(x) for x in np.unravel_index(idx, matrix.shape))  # type: ignore[return-value]


def predict_scoreline(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    stage: str = "group",
    venue: str = "",
    mode: Literal["deterministic", "sample"] = "deterministic",
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, int, float, float]:
    """
    Returns (home_goals, away_goals, home_xg, away_xg).

    deterministic (default):
        Compares ratings to decide the expected winner, then picks the
        most probable scoreline within that winner's outcome region.
        No randomness; fully reproducible.

    sample:
        Samples a scoreline from the joint Poisson PMF using `rng`.
        `rng` must be provided.
    """
    home_xg, away_xg = expected_goals(home_team, away_team, ratings, stage, venue)
    matrix           = scoreline_probability_matrix(home_xg, away_xg)

    if mode == "sample":
        if rng is None:
            raise ValueError("rng must be supplied when mode='sample'")
        flat = matrix.ravel()
        flat = flat / flat.sum()
        idx  = int(rng.choice(len(flat), p=flat))
        home_goals, away_goals = (int(x) for x in np.unravel_index(idx, matrix.shape))

    else:
        # Deterministic: decide outcome from rating gap, then formula-based scoreline
        home_r      = get_rating(home_team, ratings)
        away_r      = get_rating(away_team, ratings)
        rating_diff = home_r - away_r

        draw_threshold = (
            DRAW_RATING_THRESHOLD_GROUP if stage == "group"
            else DRAW_RATING_THRESHOLD_KNOCKOUT
        )

        if stage != "group" and 5.0 <= abs(rating_diff) < draw_threshold:
            # In knockout deterministic mode, avoid forcing too many 90-min draws
            # when there is a meaningful (but not huge) quality edge.
            avg_rating = (home_r + away_r) / 2.0
            if not (avg_rating >= 74.0 and abs(rating_diff) < 6.0):
                outcome = "home" if rating_diff > 0 else "away"
            else:
                outcome = "draw"
        elif rating_diff > draw_threshold:
            outcome = "home"
        elif rating_diff < -draw_threshold:
            outcome = "away"
        else:
            outcome = "draw"

        if outcome == "draw":
            avg = (home_xg + away_xg) / 2
            g   = max(0, round(avg))
            home_goals, away_goals = g, g
        else:
            w_xg = home_xg if outcome == "home" else away_xg
            l_xg = away_xg if outcome == "home" else home_xg
            w_goals = max(1, round(w_xg + WINNER_BIAS))
            l_goals = 0 if l_xg < LOSER_XG_THRESHOLD else round(l_xg)
            if w_goals <= l_goals:
                w_goals = l_goals + 1
            home_goals = w_goals if outcome == "home" else l_goals
            away_goals = l_goals if outcome == "home" else w_goals

    return int(home_goals), int(away_goals), home_xg, away_xg


def predict_winner_from_scoreline(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def knockout_winner(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    ratings: dict[str, float],
    stage: str = "",
    rng: Optional[np.random.Generator] = None,
) -> tuple[str, bool]:
    """
    Determines knockout winner including extra time / penalties.
    Returns (match_winner_side, penalties_bool) where side is "home" or "away".
    """
    if home_goals != away_goals:
        return ("home" if home_goals > away_goals else "away"), False

    # Tied at 90 min → penalties
    home_r = get_rating(home_team, ratings)
    away_r = get_rating(away_team, ratings)
    diff   = home_r - away_r

    if rng is not None:
        total  = home_r + away_r
        p_home = home_r / total if total > 0 else 0.5
        winner = "home" if rng.random() < p_home else "away"
    else:
        scale = KNOCKOUT_PROBABILITY_SCALES.get(stage, 17.0)
        p_home = float(np.clip(1.0 / (1.0 + np.exp(-diff / scale)), 0.20, 0.80))
        if p_home > 0.5:
            winner = "home"
        elif p_home < 0.5:
            winner = "away"
        elif abs(diff) >= PENALTY_DETERMINISTIC_THRESHOLD:
            winner = "home" if diff > 0 else "away"
        else:
            winner = "home" if hash((home_team, away_team)) % 2 == 0 else "away"

    return winner, True
