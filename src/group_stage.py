"""
Group-stage prediction and table construction.

Produces:
  - group_predictions: one row per match with all required output columns
  - group_tables: ranked standings for all 12 groups
  - qualifier lists: winners, runners-up, third-placed teams
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import pandas as pd

from src.goal_model import predict_scoreline, predict_winner_from_scoreline
from src.cards_corners_model import (
    predict_corners,
    predict_yellow_cards,
    predict_red_cards,
)
from src.ratings import get_rating


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_all_group_matches(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
) -> pd.DataFrame:
    """
    Returns a DataFrame with one prediction row per group-stage match.
    """
    rows = []
    for _, match in fixtures.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        stage = "group"
        venue = match.get("venue", "")

        hg, ag, hxg, axg = predict_scoreline(
            home, away, ratings, stage, venue, mode="deterministic"
        )
        winner = predict_winner_from_scoreline(hg, ag)

        corners = predict_corners(home, away, ratings, stage, hxg, axg, hg, ag)
        yellows = predict_yellow_cards(home, away, ratings, stage, hg, ag, hxg, axg)
        reds    = predict_red_cards(home, away, ratings, stage, hg, ag, hxg, axg)

        rows.append({
            "match_id":            int(match["match_id"]),
            "group":               match["group"],
            "home_team":           home,
            "away_team":           away,
            "date_utc":            match["date_utc"],
            "venue":               venue,
            "predicted_home_goals": hg,
            "predicted_away_goals": ag,
            "corners":             corners,
            "yellow_cards":        yellows,
            "red_cards":           reds,
            "winning_team":        winner,
            # internal helpers (dropped before final export)
            "_home_xg":            round(hxg, 3),
            "_away_xg":            round(axg, 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

class TeamRecord(NamedTuple):
    team:   str
    group:  str
    pts:    int
    gf:     int
    ga:     int
    gd:     int
    wins:   int
    draws:  int
    losses: int
    rating: float


def build_group_tables(
    predictions: pd.DataFrame,
    ratings: dict[str, float],
) -> pd.DataFrame:
    """
    Builds a complete standings table for all 12 groups.
    Tiebreakers: points > GD > GF > rating.
    Returns a DataFrame with position 1-4 within each group.
    """
    # Accumulate stats per team
    stats: dict[str, dict] = defaultdict(lambda: {
        "pts": 0, "gf": 0, "ga": 0, "wins": 0, "draws": 0, "losses": 0,
    })

    team_group: dict[str, str] = {}

    for _, row in predictions.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["predicted_home_goals"], row["predicted_away_goals"]
        grp    = row["group"]

        team_group[home] = grp
        team_group[away] = grp

        stats[home]["gf"] += hg
        stats[home]["ga"] += ag
        stats[away]["gf"] += ag
        stats[away]["ga"] += hg

        result = row["winning_team"]
        if result == "home":
            stats[home]["pts"]   += 3
            stats[home]["wins"]  += 1
            stats[away]["losses"] += 1
        elif result == "away":
            stats[away]["pts"]   += 3
            stats[away]["wins"]  += 1
            stats[home]["losses"] += 1
        else:
            stats[home]["pts"]   += 1
            stats[home]["draws"] += 1
            stats[away]["pts"]   += 1
            stats[away]["draws"] += 1

    records = []
    for team, s in stats.items():
        gd = s["gf"] - s["ga"]
        records.append(TeamRecord(
            team=team,
            group=team_group[team],
            pts=s["pts"],
            gf=s["gf"],
            ga=s["ga"],
            gd=gd,
            wins=s["wins"],
            draws=s["draws"],
            losses=s["losses"],
            rating=get_rating(team, ratings),
        ))

    df = pd.DataFrame(records)

    # Sort within each group: pts desc, gd desc, gf desc, rating desc
    df = df.sort_values(
        ["group", "pts", "gd", "gf", "rating"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    # Assign position
    df["position"] = df.groupby("group").cumcount() + 1

    # Reorder columns for readability
    cols = ["group", "position", "team", "pts", "wins", "draws", "losses", "gf", "ga", "gd", "rating"]
    return df[cols]


# ---------------------------------------------------------------------------
# Qualifier extraction
# ---------------------------------------------------------------------------

class GroupQualifiers(NamedTuple):
    winners:       dict[str, str]   # {group: team}
    runners_up:    dict[str, str]   # {group: team}
    third_placed:  list[dict]       # sorted by quality (points, GD, GF, rating)


def extract_qualifiers(
    group_tables: pd.DataFrame,
    ratings: dict[str, float],
) -> GroupQualifiers:
    """
    Extracts:
      - 12 group winners
      - 12 runners-up
      - All 12 third-place teams, sorted by qualification ranking
        (top 8 qualify for the Round of 32)
    """
    winners:    dict[str, str] = {}
    runners_up: dict[str, str] = {}
    third_list: list[dict]     = []

    for grp, grp_df in group_tables.groupby("group"):
        grp_sorted = grp_df.sort_values("position")
        positions  = grp_sorted.set_index("position")

        winners[grp]    = positions.loc[1, "team"]
        runners_up[grp] = positions.loc[2, "team"]

        if 3 in positions.index:
            third = positions.loc[3]
            third_list.append({
                "group":  grp,
                "team":   third["team"],
                "pts":    third["pts"],
                "gd":     third["gd"],
                "gf":     third["gf"],
                "rating": third["rating"],
            })

    # Sort third-placed teams: pts desc, gd desc, gf desc, rating desc
    third_list.sort(
        key=lambda x: (x["pts"], x["gd"], x["gf"], x["rating"]),
        reverse=True,
    )

    return GroupQualifiers(
        winners=winners,
        runners_up=runners_up,
        third_placed=third_list,
    )


def assign_best_third_teams(
    third_placed: list[dict],
    best_third_slots: list[tuple[int, list[str]]],
) -> dict[int, str]:
    """
    Assigns the top-8 qualifying third-placed teams to Best 3rd slots using
    constraint-respecting backtracking.

    Each slot specifies a list of allowed groups.  Each qualifying team can
    be assigned to at most one slot.  Backtracking guarantees a valid
    assignment is found if one exists, without ever producing group-constraint
    violations.  Candidates are tried in quality order (best team first) so
    the highest-ranked eligible team is preferred at each slot.

    Args:
        third_placed:     all 12 third-placed teams, sorted best-first
        best_third_slots: list of (match_id, [allowed_group_letters])

    Returns:
        {match_id: team_name}
    """
    qualifying = third_placed[:8]   # top 8 qualify for the Round of 32

    # Sort slots by fewest options first (fail-fast: most constrained first)
    def _count_options(allowed: list[str]) -> int:
        return sum(1 for t in qualifying if t["group"] in allowed)

    slots_sorted = sorted(best_third_slots, key=lambda s: _count_options(s[1]))

    assignment: dict[int, str] = {}

    def _backtrack(slot_idx: int, used: set[str]) -> bool:
        if slot_idx == len(slots_sorted):
            return True   # all slots filled

        match_id, allowed_groups = slots_sorted[slot_idx]

        for candidate in qualifying:
            if candidate["team"] in used:
                continue
            if candidate["group"] not in allowed_groups:
                continue
            # Try this candidate
            used.add(candidate["team"])
            assignment[match_id] = candidate["team"]
            if _backtrack(slot_idx + 1, used):
                return True
            # Backtrack
            used.discard(candidate["team"])
            del assignment[match_id]

        return False   # no valid candidate for this slot

    success = _backtrack(0, set())

    if not success:
        # Should not happen with the official FIFA bracket structure,
        # but provide a graceful fallback just in case.
        print("[group_stage] WARNING: backtracking found no valid Best 3rd assignment; "
              "using best-available fallback")
        used: set[str] = set()
        for match_id, allowed_groups in best_third_slots:
            if match_id in assignment:
                continue
            for candidate in qualifying:
                if candidate["team"] not in used:
                    assignment[match_id] = candidate["team"]
                    used.add(candidate["team"])
                    break
            if match_id not in assignment:
                assignment[match_id] = "Unknown"

    return assignment
