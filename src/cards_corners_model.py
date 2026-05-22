"""
Side model for corners, yellow cards, and red cards.

Designed to be conservative and realistic:
  - Corners: 7-12 range, sensitive to attacking style and match tempo
  - Yellows: 2-6 range, higher in knockout and tight matches
  - Reds: almost always 0; occasional 1 in high-friction knockouts
"""

from __future__ import annotations

import math

from src.config import (
    BASE_CORNERS,
    BASE_YELLOW,
    BASE_RED,
    KNOCKOUT_YELLOW_BOOST,
    KNOCKOUT_RED_BOOST,
    CLOSE_GAME_YELLOW_BOOST,
    MISMATCH_CORNER_PENALTY,
)


def predict_corners(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    stage: str,
    home_xg: float,
    away_xg: float,
    home_goals: int,
    away_goals: int,
) -> int:
    """
    Predicts total match corners (not split by team).
    Returned as an integer per competition convention.
    """
    total_xg = home_xg + away_xg
    corners = BASE_CORNERS

    # Higher xG -> more attacking intent -> more corners
    xg_boost = (total_xg - 2.0) * 0.8
    corners += xg_boost

    # Mismatch: weaker team may park the bus -> fewer corners
    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    rating_diff = abs(home_r - away_r)
    if rating_diff > 20:
        corners += MISMATCH_CORNER_PENALTY

    # Knockout games slightly fewer (more cautious play)
    if stage != "group":
        corners -= 0.5

    # Round and clamp
    corners = max(7, min(13, round(corners)))
    return int(corners)


def predict_yellow_cards(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    stage: str,
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
) -> int:
    """Predicts total yellow cards in the match."""
    yellows = BASE_YELLOW

    is_knockout = stage != "group"
    if is_knockout:
        yellows += KNOCKOUT_YELLOW_BOOST

    # Close matches -> more fouls -> more yellows
    xg_diff = abs(home_xg - away_xg)
    if xg_diff < 0.30:
        yellows += CLOSE_GAME_YELLOW_BOOST

    # Late-stage finals tend to be slightly more disciplined (higher stakes = more careful)
    if stage in ("Final",):
        yellows -= 0.3

    yellows = max(2, min(7, round(yellows)))
    return int(yellows)


def predict_red_cards(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    stage: str,
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
) -> int:
    """
    Predicts total red cards.  Almost always 0.
    A red is only predicted in high-friction knockout matches where
    the game is tight and both sides are under significant pressure.
    """
    prob_red = BASE_RED

    # Small base probability in knockout ties
    if stage != "group":
        prob_red += KNOCKOUT_RED_BOOST

    # Tight knockout match between sides of similar strength
    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    rating_diff = abs(home_r - away_r)
    xg_diff = abs(home_xg - away_xg)

    if stage not in ("group", "Round of 32") and rating_diff < 5 and xg_diff < 0.20:
        prob_red += 0.08

    # Deterministically assign 0 or 1 using match fingerprint
    seed = hash((home_team, away_team, stage)) % 1000
    threshold = int(prob_red * 1000)

    return 1 if seed < threshold else 0
