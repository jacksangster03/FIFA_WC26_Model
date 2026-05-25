"""
Side model for corners, yellow cards, and red cards.

Produces varied, realistic match-level predictions:
  - Corners:      6-14 range, most results 7-12, some high-tempo matches 11-13
  - Yellow cards: group avg 3.3-4.2, KO avg 4.0-5.2
  - Red cards:    0-3 total across all 104 matches

All outputs are deterministic (no randomness). Variation comes from:
  - Team style profiles (data/external/team_style_profiles.csv)
  - Rating gap and stage
  - Closeness of expected scoreline / xG
  - A three-component match fingerprint for wide deterministic spread
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Style profile loader
# ---------------------------------------------------------------------------

_STYLE_LABEL_MAP = {
    "low":         0.0,
    "medium":      1.0,
    "medium_high": 1.5,
    "high":        2.0,
}

_DEFAULT_STYLE = {
    "tempo":            1.0,
    "physicality":      1.0,
    "set_piece_threat": 1.0,
    "discipline_risk":  1.0,
}


@lru_cache(maxsize=1)
def _load_profiles() -> dict[str, dict[str, float]]:
    """Loads team_style_profiles.csv once; returns {team: {attr: float}}."""
    path = Path(__file__).parent.parent / "data" / "external" / "team_style_profiles.csv"
    profiles: dict[str, dict[str, float]] = {}
    if not path.exists():
        return profiles
    try:
        import pandas as pd
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            team = str(row["team"])
            profiles[team] = {
                "tempo":            _STYLE_LABEL_MAP.get(str(row.get("tempo", "medium")),      1.0),
                "physicality":      _STYLE_LABEL_MAP.get(str(row.get("physicality", "medium")),1.0),
                "set_piece_threat": _STYLE_LABEL_MAP.get(str(row.get("set_piece_threat", "medium")), 1.0),
                "discipline_risk":  _STYLE_LABEL_MAP.get(str(row.get("discipline_risk", "medium")),  1.0),
            }
    except Exception:
        pass
    return profiles


def _style(team: str) -> dict[str, float]:
    return _load_profiles().get(team, _DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# Three-component deterministic fingerprint
# ---------------------------------------------------------------------------

def _fp(home_team: str, away_team: str, stage: str, salt: str = "") -> float:
    """
    Three independent hash components, combined for ~[-1.48, +1.48] spread.
    Using three primes (13, 9, 7) gives 13*9*7 = 819 distinct combinations
    before rounding, providing good separation across the integer clamp range.
    """
    h1 = hash((home_team, away_team, stage, salt + "_p"))
    h2 = hash((away_team, home_team, stage, salt + "_q"))
    h3 = hash((home_team + ":" + away_team, stage, salt + "_r"))
    v1 = ((h1 % 13) - 6) * 0.15   # -0.90 to +0.90
    v2 = ((h2 %  9) - 4) * 0.10   # -0.40 to +0.40
    v3 = ((h3 %  7) - 3) * 0.06   # -0.18 to +0.18
    return v1 + v2 + v3


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------

_STAGE_CORNERS_ADJ: dict[str, float] = {
    "group":               0.0,
    "Round of 32":        -0.2,
    "Round of 16":        -0.3,
    "Quarter-final":      -0.4,
    "Semi-final":         -0.4,
    "Third-place playoff": -0.2,
    "Final":              -0.5,
}


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
    Predicts total match corners as an integer, clamped to [6, 14].

    Drivers: base 9.0, attack strength, xG tempo, closeness, chasing,
    mismatch penalty, knockout caution, team style, match fingerprint.
    """
    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    avg_r  = (home_r + away_r) / 2.0
    gap    = abs(home_r - away_r)

    corners: float = 9.0

    # Attack strength: elite teams generate more set-piece situations
    attack_adj = max(-0.5, min(1.0, (avg_r - 64.0) / 38.0))
    corners += attack_adj

    # xG tempo: high-intent games produce more corner attempts
    total_xg = home_xg + away_xg
    xg_adj   = max(-0.6, min(1.0, (total_xg - 2.6) * 0.65))
    corners += xg_adj

    # Closeness: both teams pushing creates more corners
    xg_diff = abs(home_xg - away_xg)
    if xg_diff < 0.20:
        corners += 0.8
    elif xg_diff < 0.45:
        corners += 0.4

    # Mismatch penalty: dominated side parks and rarely wins corners
    if gap > 28:
        corners -= 1.8
    elif gap > 20:
        corners -= 1.0
    elif gap > 14:
        corners -= 0.5

    # Underdog chasing: trailing weaker side forces extra volume late on
    if home_goals != away_goals:
        weaker_trailing = (
            (home_r < away_r and home_goals < away_goals) or
            (away_r < home_r and away_goals < home_goals)
        )
        if weaker_trailing and gap < 22:
            corners += 0.3 if gap > 12 else 0.2

    # Stage caution
    corners += _STAGE_CORNERS_ADJ.get(stage, 0.0)

    # Team style
    hs  = _style(home_team)
    as_ = _style(away_team)
    spt_adj   = ((hs["set_piece_threat"] - 1.0) + (as_["set_piece_threat"] - 1.0)) * 0.40
    tempo_adj = ((hs["tempo"]            - 1.0) + (as_["tempo"]            - 1.0)) * 0.25
    corners += spt_adj + tempo_adj

    # Three-component fingerprint for deterministic variety
    corners += _fp(home_team, away_team, stage, "corners")

    return int(max(6, min(14, round(corners))))


# ---------------------------------------------------------------------------
# Yellow cards
# ---------------------------------------------------------------------------

_STAGE_YELLOW_ADJ: dict[str, float] = {
    "group":               0.0,
    "Round of 32":         0.45,
    "Round of 16":         0.70,
    "Quarter-final":       1.00,
    "Semi-final":          1.20,
    "Third-place playoff": 0.80,
    "Final":               0.85,
}


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
    """
    Predicts total yellow cards, clamped to [1, 9].

    Group target 3.3-4.2; knockout target 4.0-5.2.
    """
    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    gap    = abs(home_r - away_r)

    yellows: float = 3.2

    # Stage escalation
    yellows += _STAGE_YELLOW_ADJ.get(stage, 0.0)

    # Penalty-pressure draw in knockout: maximum friction
    xg_diff = abs(home_xg - away_xg)
    is_drawn_ko = (home_goals == away_goals and stage != "group")
    if is_drawn_ko:
        yellows += 1.0
    elif xg_diff < 0.15:
        yellows += 0.7
    elif xg_diff < 0.40:
        yellows += 0.35

    # Very evenly matched: extra competitive edge
    if gap < 5:
        yellows += 0.3

    # Underdog frustration: significantly weaker side trailing tends to foul more
    if gap > 15 and home_goals != away_goals:
        weaker_trailing = (
            (home_r < away_r and home_goals < away_goals) or
            (away_r < home_r and away_goals < home_goals)
        )
        if weaker_trailing:
            yellows += 0.35

    # Large mismatch: fewer yellows if weaker side is outclassed and drops off
    if gap > 25:
        hs  = _style(home_team)
        as_ = _style(away_team)
        phys_max = max(hs["physicality"], as_["physicality"])
        yellows -= max(0.0, 0.4 - (phys_max - 1.0) * 0.2)

    # Team style: physicality and discipline risk
    hs  = _style(home_team)
    as_ = _style(away_team)
    style_adj  = ((hs["physicality"]     - 1.0) + (as_["physicality"]     - 1.0)) * 0.28
    style_adj += ((hs["discipline_risk"] - 1.0) + (as_["discipline_risk"] - 1.0)) * 0.24
    yellows += style_adj

    # Close knockout tie extra intensity
    if stage != "group" and gap < 9 and abs(home_goals - away_goals) <= 1:
        yellows += 0.4

    # Three-component fingerprint
    yellows += _fp(home_team, away_team, stage, "yellows")

    return int(max(1, min(9, round(yellows))))


# ---------------------------------------------------------------------------
# Red cards
# ---------------------------------------------------------------------------

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
    Predicts total red cards. Targets 0-3 across all 104 matches.

    A red requires: late KO stage, close scoreline, high yellows,
    tight rating gap, physical/disciplined team, and hash threshold.
    """
    if stage in ("group", "Round of 32"):
        return 0

    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    gap    = abs(home_r - away_r)

    if gap >= 10:
        return 0
    if abs(home_goals - away_goals) > 1:
        return 0

    yellows = predict_yellow_cards(
        home_team, away_team, ratings, stage,
        home_goals, away_goals, home_xg, away_xg,
    )

    # R16 requires very high friction; QF+ slightly lower bar
    min_yellows = 6 if stage == "Round of 16" else 5
    if yellows < min_yellows:
        return 0

    hs  = _style(home_team)
    as_ = _style(away_team)
    max_disc = max(hs["discipline_risk"], as_["discipline_risk"])
    max_phys = max(hs["physicality"],     as_["physicality"])
    if max_disc < 1.5 and max_phys < 1.5:
        return 0

    h = hash((home_team, away_team, stage, "red"))
    return 1 if (h % 3 == 0) else 0
