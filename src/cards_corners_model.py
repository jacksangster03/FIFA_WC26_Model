"""
Side model for corners, yellow cards, and red cards.

Produces varied, realistic match-level predictions:
  - Corners:      6-14 range, most results 7-12
  - Yellow cards: 1-8 range, most results 2-5
  - Red cards:    almost always 0; 1 only in specific high-friction knockout scenarios

All outputs are deterministic (no randomness). Variation comes from:
  - Team style profiles (data/external/team_style_profiles.csv)
  - Rating gap and stage
  - Closeness of expected scoreline
  - A small match-fingerprint based on team name hash
"""

from __future__ import annotations

import math
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
# Deterministic match fingerprint (small, bounded perturbation)
# ---------------------------------------------------------------------------

def _fingerprint(home_team: str, away_team: str, stage: str, salt: str = "") -> float:
    """
    Returns a value in [-0.8, +0.8] unique to this match.
    Uses 17 distinct steps for wider distribution spread.
    """
    h = hash((home_team, away_team, stage, salt))
    return ((h % 17) - 8) * 0.1   # -0.8 … 0 … +0.8  (17 steps)


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------

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
    Predicts total match corners as an integer.

    Drivers:
      base 8.5
      + attacking strength of both teams (avg rating)
      + xG tempo signal
      + closeness boost (both teams pushing)
      + team style (set-piece threat, tempo)
      - mismatch penalty (dominant team, weaker team parks)
      - knockout caution factor
      + small match fingerprint for deterministic variation
    Clamped to [6, 14].
    """
    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    avg_r  = (home_r + away_r) / 2.0
    gap    = abs(home_r - away_r)

    # Base
    corners: float = 8.5

    # Attacking strength: teams rated above 65 generate more corners on average
    attack_adj = min(1.0, max(-0.3, (avg_r - 65.0) / 55.0))
    corners += attack_adj

    # xG tempo: more attacking intent -> more corners
    total_xg = home_xg + away_xg
    xg_adj   = (total_xg - 2.7) * 0.5
    corners += max(-0.5, min(0.8, xg_adj))

    # Closeness: close games see both teams push forward
    xg_diff = abs(home_xg - away_xg)
    if xg_diff < 0.2:
        corners += 0.6
    elif xg_diff < 0.45:
        corners += 0.3

    # Mismatch penalty: weak team parks -> fewer set-pieces
    if gap > 28:
        corners -= 1.0
    elif gap > 16:
        corners -= 0.5

    # Knockout caution
    if stage != "group":
        corners -= 0.4

    # Team style: set-piece threat and tempo
    hs = _style(home_team)
    as_ = _style(away_team)
    spt_adj  = ((hs["set_piece_threat"] - 1.0) + (as_["set_piece_threat"] - 1.0)) * 0.20
    tempo_adj = ((hs["tempo"] - 1.0)           + (as_["tempo"] - 1.0))           * 0.15
    corners += spt_adj + tempo_adj

    # Primary match fingerprint for deterministic variation
    corners += _fingerprint(home_team, away_team, stage, "corners")
    # Secondary fingerprint (different salt) adds finer variation
    corners += _fingerprint(away_team, home_team, stage, "corners2") * 0.4

    return int(max(6, min(14, round(corners))))


# ---------------------------------------------------------------------------
# Yellow cards
# ---------------------------------------------------------------------------

_STAGE_YELLOW_ADJ: dict[str, float] = {
    "group":               0.0,
    "Round of 32":         0.35,
    "Round of 16":         0.55,
    "Quarter-final":       0.75,
    "Semi-final":          0.85,
    "Third-place playoff": 0.65,
    "Final":               0.55,
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
    Predicts total yellow cards.

    Drivers:
      base 2.8
      + stage escalation
      + closeness (tight games -> more fouls)
      + team physicality / discipline_risk profiles
      + small match fingerprint
    Clamped to [1, 8], target 2-5 for most matches.
    """
    home_r = ratings.get(home_team, 55.0)
    away_r = ratings.get(away_team, 55.0)
    gap    = abs(home_r - away_r)

    yellows: float = 2.8

    # Stage escalation
    yellows += _STAGE_YELLOW_ADJ.get(stage, 0.0)

    # Closeness
    xg_diff = abs(home_xg - away_xg)
    is_drawn_ko = (home_goals == away_goals and stage != "group")
    if is_drawn_ko:
        yellows += 0.7    # penalty-pressure draw
    elif xg_diff < 0.2:
        yellows += 0.55
    elif xg_diff < 0.5:
        yellows += 0.25

    # Very close teams: extra friction
    if gap < 5:
        yellows += 0.2

    # Large mismatch: fewer yellows unless underdog is defending deep
    # (physicality compensates slightly)
    if gap > 25:
        hs = _style(home_team)
        as_ = _style(away_team)
        phys_max = max(hs["physicality"], as_["physicality"])
        yellows -= max(0.0, 0.4 - (phys_max - 1.0) * 0.2)

    # Team style: physicality and discipline risk
    hs  = _style(home_team)
    as_ = _style(away_team)
    style_adj  = ((hs["physicality"]     - 1.0) + (as_["physicality"]     - 1.0)) * 0.18
    style_adj += ((hs["discipline_risk"] - 1.0) + (as_["discipline_risk"] - 1.0)) * 0.18
    yellows += style_adj

    # Primary match fingerprint
    yellows += _fingerprint(home_team, away_team, stage, "yellows")
    # Secondary fingerprint adds finer spread
    yellows += _fingerprint(away_team, home_team, stage, "yellows2") * 0.35

    return int(max(1, min(8, round(yellows))))


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
    Predicts total red cards.

    Almost always 0.  A red is predicted only when ALL of these hold:
      - Late knockout stage (QF, SF, Final, Third-place playoff)
      - Scoreline is close (|goal diff| <= 1)
      - Predicted yellows >= 5 (high-friction match)
      - Rating gap < 10 (evenly matched)
      - At least one team has high physicality / discipline risk profile
      - Match fingerprint hits the threshold (1 in 4 qualifying matches)

    Targets 0-3 total reds across all 104 matches.
    """
    # Group and R32 almost never have reds
    if stage in ("group", "Round of 32", "Round of 16"):
        return 0

    home_r  = ratings.get(home_team, 55.0)
    away_r  = ratings.get(away_team, 55.0)
    gap     = abs(home_r - away_r)
    scoreline_diff = abs(home_goals - away_goals)

    if scoreline_diff > 1:
        return 0
    if gap >= 10:
        return 0

    # Check yellows for this match
    yellows = predict_yellow_cards(
        home_team, away_team, ratings, stage,
        home_goals, away_goals, home_xg, away_xg,
    )
    if yellows < 5:
        return 0

    # Check team friction profile
    hs  = _style(home_team)
    as_ = _style(away_team)
    max_disc = max(hs["discipline_risk"], as_["discipline_risk"])
    max_phys = max(hs["physicality"],     as_["physicality"])
    if max_disc < 1.5 and max_phys < 1.5:
        return 0

    # Deterministic threshold: 1 in 4 qualifying matches gets a red
    h = hash((home_team, away_team, stage, "red"))
    return 1 if (h % 4 == 0) else 0
