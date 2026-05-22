"""
Knockout bracket resolution and prediction.

Slot resolution order:
  1. Parse slot strings from knockout_slots.csv
  2. Resolve group-based slots (Winner/Runner-up Group X)
  3. Assign Best 3rd teams via group_stage.assign_best_third_teams
  4. Simulate matches round by round, propagating winners and losers
  5. Handle third-place playoff (Loser Match X slots)
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from src.goal_model import predict_scoreline, knockout_winner
from src.cards_corners_model import (
    predict_corners,
    predict_yellow_cards,
    predict_red_cards,
)
from src.group_stage import (
    GroupQualifiers,
    assign_best_third_teams,
)
from src.ratings import get_rating


# ---------------------------------------------------------------------------
# Slot parsing
# ---------------------------------------------------------------------------

def _parse_best_third_groups(slot: str) -> Optional[list[str]]:
    """
    Parses 'Best 3rd (Groups A/B/C/D/F)' -> ['A','B','C','D','F']
    Returns None if slot is not a Best 3rd slot.
    """
    m = re.search(r"Best 3rd \(Groups ([A-Z/]+)\)", slot)
    if m:
        return m.group(1).split("/")
    return None


def _parse_winner_group(slot: str) -> Optional[str]:
    m = re.match(r"Winner Group ([A-L])", slot)
    return m.group(1) if m else None


def _parse_runner_up_group(slot: str) -> Optional[str]:
    m = re.match(r"Runner-up Group ([A-L])", slot)
    return m.group(1) if m else None


def _parse_winner_match(slot: str) -> Optional[int]:
    m = re.match(r"Winner Match (\d+)", slot)
    return int(m.group(1)) if m else None


def _parse_loser_match(slot: str) -> Optional[int]:
    m = re.match(r"Loser Match (\d+)", slot)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main bracket runner
# ---------------------------------------------------------------------------

def simulate_knockout_bracket(
    knockout_slots: pd.DataFrame,
    qualifiers: GroupQualifiers,
    ratings: dict[str, float],
) -> pd.DataFrame:
    """
    Resolves all 32 knockout match slots and predicts each match.
    Returns a DataFrame with one row per knockout match.
    """
    # Build the Best 3rd slot map first
    best_third_slot_defs: list[tuple[int, list[str]]] = []
    for _, row in knockout_slots.iterrows():
        for slot_col in ("slot_home", "slot_away"):
            groups = _parse_best_third_groups(row[slot_col])
            if groups:
                best_third_slot_defs.append((int(row["match_id"]), groups))

    # Remove duplicates (each match_id appears once)
    seen_ids: set[int] = set()
    unique_best_third = []
    for mid, grps in best_third_slot_defs:
        if mid not in seen_ids:
            unique_best_third.append((mid, grps))
            seen_ids.add(mid)

    best_third_assignment = assign_best_third_teams(
        qualifiers.third_placed,
        unique_best_third,
    )

    # Match results store: {match_id: {winner, loser, home_goals, away_goals, penalties}}
    match_results: dict[int, dict] = {}

    # Process matches in match_id order (guarantees dependencies are resolved first)
    sorted_matches = knockout_slots.sort_values("match_id")

    rows = []
    for _, ko in sorted_matches.iterrows():
        mid  = int(ko["match_id"])
        rnd  = ko["round"]
        mult = int(ko["multiplier"])
        date = ko["date_utc"]
        venue = ko["venue"]
        slot_h = ko["slot_home"]
        slot_a = ko["slot_away"]

        home_team = _resolve_slot(
            slot_h, qualifiers, match_results, best_third_assignment, mid
        )
        away_team = _resolve_slot(
            slot_a, qualifiers, match_results, best_third_assignment, mid
        )

        # Predict the match (deterministic: argmax of draw-discounted PMF)
        hg, ag, hxg, axg = predict_scoreline(
            home_team, away_team, ratings, rnd, venue, mode="deterministic"
        )
        winner, penalties = knockout_winner(
            home_team, away_team, hg, ag, ratings, stage=rnd
        )
        loser = away_team if winner == "home" else home_team

        corners = predict_corners(
            home_team, away_team, ratings, rnd, hxg, axg, hg, ag
        )
        yellows = predict_yellow_cards(
            home_team, away_team, ratings, rnd, hg, ag, hxg, axg
        )
        reds = predict_red_cards(
            home_team, away_team, ratings, rnd, hg, ag, hxg, axg
        )

        match_results[mid] = {
            "winner":    home_team if winner == "home" else away_team,
            "loser":     loser,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": hg,
            "away_goals": ag,
            "penalties":  penalties,
        }

        rows.append({
            "match_id":              mid,
            "round":                 rnd,
            "multiplier":            mult,
            "date_utc":              date,
            "venue":                 venue,
            "slot_home":             slot_h,
            "slot_away":             slot_a,
            "predicted_home_team":   home_team,
            "predicted_away_team":   away_team,
            "predicted_home_goals":  hg,
            "predicted_away_goals":  ag,
            "corners":               corners,
            "yellow_cards":          yellows,
            "red_cards":             reds,
            "match_winner":          winner,
            "penalties":             penalties,
            # Internal helpers
            "_home_xg":              round(hxg, 3),
            "_away_xg":              round(axg, 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Slot resolver
# ---------------------------------------------------------------------------

def _resolve_slot(
    slot: str,
    qualifiers: GroupQualifiers,
    match_results: dict[int, dict],
    best_third_assignment: dict[int, str],
    current_match_id: int,
) -> str:
    """Resolves a slot string to a team name."""

    # Winner/Runner-up Group
    g = _parse_winner_group(slot)
    if g:
        return qualifiers.winners.get(g, f"Winner Group {g}")

    g = _parse_runner_up_group(slot)
    if g:
        return qualifiers.runners_up.get(g, f"Runner-up Group {g}")

    # Best 3rd
    if _parse_best_third_groups(slot):
        return best_third_assignment.get(current_match_id, "Best 3rd Unknown")

    # Winner/Loser of previous match
    mid = _parse_winner_match(slot)
    if mid is not None:
        res = match_results.get(mid)
        if res:
            return res["winner"]
        return f"Winner Match {mid}"

    mid = _parse_loser_match(slot)
    if mid is not None:
        res = match_results.get(mid)
        if res:
            return res["loser"]
        return f"Loser Match {mid}"

    return slot  # unrecognised: return as-is


# ---------------------------------------------------------------------------
# Bracket summary
# ---------------------------------------------------------------------------

def build_bracket_summary(
    knockout_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a compact bracket summary table for human review.
    """
    cols = [
        "match_id", "round", "multiplier",
        "predicted_home_team", "predicted_away_team",
        "predicted_home_goals", "predicted_away_goals",
        "match_winner", "penalties",
    ]
    summary = knockout_predictions[cols].copy()
    summary["winner_team"] = summary.apply(
        lambda r: r["predicted_home_team"] if r["match_winner"] == "home"
                  else r["predicted_away_team"],
        axis=1,
    )
    return summary
