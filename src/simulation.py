"""
Optional Monte Carlo simulation.

Simulates the full tournament N times using Poisson sampling to estimate:
  - Group advancement probabilities
  - Champion probabilities

Run via:
    from src.simulation import run_simulation
    run_simulation(fixtures, knockout_slots, ratings, n=1000)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DEFAULT_SIM_N, OUTPUT_FILES
from src.goal_model import expected_goals, knockout_winner
from src.group_stage import build_group_tables, extract_qualifiers, assign_best_third_teams
from src.knockout_stage import (
    _parse_winner_group, _parse_runner_up_group, _parse_best_third_groups,
    _parse_winner_match, _parse_loser_match,
)
from src.ratings import get_rating


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _simulate_group_stage(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict, dict, list]:
    from src.goal_model import predict_scoreline as _ps
    rows = []
    for _, match in fixtures.iterrows():
        home, away = match["home_team"], match["away_team"]
        hg, ag, hxg, axg = _ps(
            home, away, ratings, "group", match.get("venue", ""),
            mode="sample", rng=rng,
        )
        if hg > ag:
            winner = "home"
        elif ag > hg:
            winner = "away"
        else:
            winner = "draw"
        rows.append({
            "match_id":             match["match_id"],
            "group":                match["group"],
            "home_team":            home,
            "away_team":            away,
            "predicted_home_goals": hg,
            "predicted_away_goals": ag,
            "winning_team":         winner,
            "_home_xg":             hxg,
            "_away_xg":             axg,
        })
    pred_df = pd.DataFrame(rows)
    tables  = build_group_tables(pred_df, ratings)
    quals   = extract_qualifiers(tables, ratings)
    return pred_df, quals.winners, quals.runners_up, quals.third_placed


def _simulate_ko_match(
    home: str,
    away: str,
    ratings: dict[str, float],
    stage: str,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Returns (winner_team_name, loser_team_name)."""
    from src.goal_model import predict_scoreline as _ps
    hg, ag, _, _ = _ps(home, away, ratings, stage, mode="sample", rng=rng)
    winner_side, _ = knockout_winner(home, away, hg, ag, ratings, rng=rng)
    winner = home if winner_side == "home" else away
    loser  = away if winner_side == "home" else home
    return winner, loser


def _build_best_third_slot_defs(knockout_slots: pd.DataFrame) -> list[tuple[int, list[str]]]:
    seen: set[int] = set()
    result = []
    for _, row in knockout_slots.iterrows():
        mid = int(row["match_id"])
        if mid in seen:
            continue
        for col in ("slot_home", "slot_away"):
            grps = _parse_best_third_groups(row[col])
            if grps:
                result.append((mid, grps))
                seen.add(mid)
                break
    return result


def _simulate_knockout_bracket(
    knockout_slots: pd.DataFrame,
    winners: dict[str, str],
    runners_up: dict[str, str],
    third_placed: list[dict],
    ratings: dict[str, float],
    rng: np.random.Generator,
    slot_defs: list[tuple[int, list[str]]],
    advancement_counts: Optional[dict] = None,
    penalty_counts: Optional[dict] = None,
) -> Optional[str]:
    bt = assign_best_third_teams(third_placed, slot_defs)
    match_results: dict[int, dict] = {}

    for _, ko in knockout_slots.sort_values("match_id").iterrows():
        mid = int(ko["match_id"])
        rnd = ko["round"]

        def resolve(slot: str) -> str:
            g = _parse_winner_group(slot)
            if g:
                return winners.get(g, "Unknown")
            g = _parse_runner_up_group(slot)
            if g:
                return runners_up.get(g, "Unknown")
            if _parse_best_third_groups(slot):
                return bt.get(mid, "Unknown")
            pm = _parse_winner_match(slot)
            if pm is not None:
                return match_results.get(pm, {}).get("winner", "Unknown")
            pm = _parse_loser_match(slot)
            if pm is not None:
                return match_results.get(pm, {}).get("loser", "Unknown")
            return slot

        home = resolve(ko["slot_home"])
        away = resolve(ko["slot_away"])
        if "Unknown" in (home, away):
            return None

        from src.goal_model import predict_scoreline as _ps
        hg, ag, _, _ = _ps(home, away, ratings, rnd, mode="sample", rng=rng)
        from src.goal_model import knockout_winner as _kw
        winner_side, pens = _kw(home, away, hg, ag, ratings, rng=rng)
        w = home if winner_side == "home" else away
        l = away if winner_side == "home" else home

        match_results[mid] = {"winner": w, "loser": l}

        # Track advancement and penalty stats
        if advancement_counts is not None and rnd != "Third-place playoff":
            advancement_counts[rnd][w] = advancement_counts[rnd].get(w, 0) + 1
        if pens and penalty_counts is not None:
            penalty_counts["matches"][w] = penalty_counts["matches"].get(w, 0) + 1
            penalty_counts["matches"][l] = penalty_counts["matches"].get(l, 0) + 1
            # Loser eliminated by penalties
            penalty_counts["elims"][l] = penalty_counts["elims"].get(l, 0) + 1

    final_id = knockout_slots["match_id"].max()
    return match_results.get(int(final_id), {}).get("winner")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_KO_ROUNDS_ORDERED = [
    "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"
]


def run_simulation(
    fixtures: pd.DataFrame,
    knockout_slots: pd.DataFrame,
    ratings: dict[str, float],
    n: int = DEFAULT_SIM_N,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run n full tournament simulations.
    Returns a DataFrame with champion probability per team.
    Also saves simulation_advancement_summary.csv with per-round counts.
    """
    print(f"[simulation] Running {n:,} Monte Carlo simulations...")
    rng = np.random.default_rng(seed)

    slot_defs = _build_best_third_slot_defs(knockout_slots)

    champion_counts: dict[str, int] = defaultdict(int)
    # Per-round advancement: {round_name: {team: count}}
    advancement: dict[str, dict[str, int]] = {r: defaultdict(int) for r in _KO_ROUNDS_ORDERED}
    r32_entry: dict[str, int] = defaultdict(int)   # teams entering Round of 32
    penalty_matches: dict[str, int] = defaultdict(int)
    penalty_elims:   dict[str, int] = defaultdict(int)
    failures = 0

    # Track avg opponent ratings per team
    total_opp_rating:     dict[str, float] = defaultdict(float)
    total_opp_matches:    dict[str, int]   = defaultdict(int)
    elim_opp_rating:      dict[str, float] = defaultdict(float)

    for _ in range(n):
        try:
            _, wins, runs, thirds = _simulate_group_stage(fixtures, ratings, rng)

            # Track all teams entering R32
            for team in list(wins.values()) + list(runs.values()):
                r32_entry[team] += 1
            for t in thirds[:8]:
                r32_entry[t["team"]] += 1

            adv_counts  = {r: {} for r in _KO_ROUNDS_ORDERED}
            pen_counts  = {"matches": {}, "elims": {}}

            champion = _simulate_knockout_bracket(
                knockout_slots, wins, runs, thirds, ratings, rng, slot_defs,
                advancement_counts=adv_counts,
                penalty_counts=pen_counts,
            )

            if champion:
                champion_counts[champion] += 1
                # Champion also "won" the final
                advancement["Final"][champion] = advancement["Final"].get(champion, 0) + 1
            else:
                failures += 1
                continue

            # Merge per-sim advancement into totals
            for rnd in _KO_ROUNDS_ORDERED:
                for team, cnt in adv_counts.get(rnd, {}).items():
                    advancement[rnd][team] = advancement[rnd].get(team, 0) + cnt

            for team, cnt in pen_counts["matches"].items():
                penalty_matches[team] += cnt
            for team, cnt in pen_counts["elims"].items():
                penalty_elims[team] += cnt

        except Exception:
            failures += 1

    if failures:
        print(f"[simulation] {failures} simulations failed/skipped (bracket resolution errors)")

    total = n - failures
    champion_rows = [
        {
            "team":              team,
            "champion_count":    count,
            "champion_prob_pct": round(count / n * 100, 2),
        }
        for team, count in sorted(champion_counts.items(), key=lambda x: -x[1])
    ]

    df = pd.DataFrame(champion_rows)
    out_path = OUTPUT_FILES["simulation_summary"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.DataFrame([{"team": "_META_n_sims", "champion_count": n, "champion_prob_pct": None}])
    pd.concat([meta, df], ignore_index=True).to_csv(out_path, index=False)

    # Build and save advancement summary
    all_teams = sorted(set(r32_entry.keys()) | set(champion_counts.keys()))
    adv_rows = []
    for team in all_teams:
        entry_n = r32_entry.get(team, 0)
        r32_n   = advancement["Round of 32"].get(team, 0)
        r16_n   = advancement["Round of 16"].get(team, 0)
        qf_n    = advancement["Quarter-final"].get(team, 0)
        sf_n    = advancement["Semi-final"].get(team, 0)
        fin_n   = advancement["Final"].get(team, 0)
        champ_n = champion_counts.get(team, 0)
        adv_rows.append({
            "team":                          team,
            "r32_count":                     entry_n,
            "r16_count":                     r32_n,
            "qf_count":                      r16_n,
            "sf_count":                      qf_n,
            "final_count":                   sf_n,
            "champion_count":                champ_n,
            "r32_pct":                       round(entry_n / n * 100, 1),
            "r16_pct":                       round(r32_n   / n * 100, 1),
            "qf_pct":                        round(r16_n   / n * 100, 1),
            "sf_pct":                        round(qf_n    / n * 100, 1),
            "final_pct":                     round(sf_n    / n * 100, 1),
            "champion_pct":                  round(champ_n / n * 100, 2),
            "penalty_matches_count":         penalty_matches.get(team, 0),
            "penalty_eliminations_count":    penalty_elims.get(team, 0),
        })

    adv_df = pd.DataFrame(adv_rows).sort_values("champion_count", ascending=False).reset_index(drop=True)
    adv_path = out_path.parent / "simulation_advancement_summary.csv"
    adv_df.to_csv(adv_path, index=False)
    print(f"[simulation] Advancement summary saved -> {adv_path}")

    print(f"[simulation] Total simulations: {n}  |  Failures: {failures}  |  "
          f"champion_count sum: {sum(d['champion_count'] for d in champion_rows)}")
    print("\nTop-10 champion probabilities:")
    print(df.head(10).to_string(index=False))
    return df
