"""
FIFA World Cup 2026 Prediction Pipeline

Usage:
    python -m src.main                        # standard offline run
    python -m src.main --online               # download external data first
    python -m src.main --sim                  # + Monte Carlo (default 1000 sims)
    python -m src.main --sim --n-sims 5000    # + Monte Carlo with 5000 sims
    python -m src.main --online --sim --n-sims 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import OUTPUTS_DIR, OUTPUT_FILES, DEFAULT_SIM_N
from src.data_loader import load_group_fixtures, load_knockout_slots
from src.external_data import load_external_data
from src.form_processor import run_form_processing
from src.ratings import compute_team_ratings, SEEDED_RATINGS
from src.group_stage import (
    predict_all_group_matches,
    build_group_tables,
    extract_qualifiers,
)
from src.knockout_stage import simulate_knockout_bracket, build_bracket_summary
from src.manual_overrides import load_overrides, apply_overrides
from src.quality_checks import run_all_checks
from src.report import generate_model_report


# ---------------------------------------------------------------------------
# Column specs
# ---------------------------------------------------------------------------
_GROUP_OUTPUT_COLS = [
    "match_id", "group", "home_team", "away_team", "date_utc", "venue",
    "predicted_home_goals", "predicted_away_goals", "corners",
    "yellow_cards", "red_cards", "winning_team",
]

_KNOCKOUT_OUTPUT_COLS = [
    "match_id", "round", "multiplier", "date_utc", "venue",
    "slot_home", "slot_away",
    "predicted_home_team", "predicted_away_team",
    "predicted_home_goals", "predicted_away_goals", "corners",
    "yellow_cards", "red_cards", "match_winner", "penalties",
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    online: bool = False,
    refresh_data: bool = False,
    run_sim: bool = False,
    n_sims: int = DEFAULT_SIM_N,
) -> None:
    print("\n" + "=" * 60)
    print("FIFA WC 2026 Prediction Pipeline")
    print("=" * 60 + "\n")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Optional data refresh
    if refresh_data or online:
        print("--- [0/9] Refreshing external data sources ---")
        from src.data_sources.refresh_all import run_all
        run_all(force_refresh=refresh_data)
    else:
        # Always ensure processed data exists (uses cache if available)
        from src.data_sources import playoff_resolver, fifa_rankings, world_elo, tournament_achievements
        playoff_resolver.run(force_refresh=False)
        fifa_rankings.run(force_refresh=False)
        world_elo.run(force_refresh=False)
        tournament_achievements.run(force_refresh=False)

    # 1. Fixtures (prefers resolved if playoff_resolver ran)
    print("--- [1/9] Loading fixtures ---")
    group_fixtures = load_group_fixtures()
    knockout_slots = load_knockout_slots()

    # 2. External data (headers only; form_processor does the heavy lifting)
    print("\n--- [2/9] Loading external data ---")
    external_data = load_external_data(online=online)
    has_external  = any(v is not None for v in external_data.values())

    # 3. Form processing (writes data/processed/)
    print("\n--- [3/9] Processing historical form data ---")
    adjusted_ratings = run_form_processing(dict(SEEDED_RATINGS))
    # form_processor saves team_ratings_final.csv; ratings.py will load it
    # but we pass adjusted_ratings directly to avoid re-reading the file.
    ratings = adjusted_ratings

    print(f"[main] Ratings computed for {len(ratings)} teams")
    _print_top_ratings(ratings, n=10)

    # 4. Group stage predictions
    print("\n--- [4/9] Predicting group stage ---")
    group_predictions = predict_all_group_matches(group_fixtures, ratings)
    group_tables      = build_group_tables(group_predictions, ratings)
    qualifiers        = extract_qualifiers(group_tables, ratings)

    draws = (group_predictions["winning_team"] == "draw").sum()
    avg_g = (group_predictions["predicted_home_goals"] +
             group_predictions["predicted_away_goals"]).mean()
    print(f"[main] Group predictions: {len(group_predictions)} matches")
    print(f"[main] Draws: {draws}/72  |  Avg goals/match: {avg_g:.2f}")
    print(f"[main] Group winners:    {list(qualifiers.winners.values())}")
    print(f"[main] Group runners-up: {list(qualifiers.runners_up.values())}")
    print("[main] Third-place qualifiers (top 8):")
    for t in qualifiers.third_placed[:8]:
        print(f"         Group {t['group']}: {t['team']} ({t['pts']} pts, GD {t['gd']})")

    _sanity_check_groups(qualifiers, ratings, group_tables)

    # 5. Knockout bracket
    print("\n--- [5/9] Simulating knockout bracket ---")
    knockout_predictions = simulate_knockout_bracket(knockout_slots, qualifiers, ratings)
    bracket_summary      = build_bracket_summary(knockout_predictions)

    # 5b. Bracket path + xG calibration diagnostics
    try:
        from src.bracket_difficulty import generate_bracket_diagnostics, generate_xg_diagnostics
        generate_bracket_diagnostics(knockout_predictions, ratings)
        generate_xg_diagnostics(ratings)
    except Exception as _bde:
        print(f"[main] WARN: bracket diagnostics failed: {_bde}")

    # 6. Manual overrides
    print("\n--- [6/9] Applying manual overrides ---")
    overrides            = load_overrides()
    group_predictions    = apply_overrides(group_predictions,    overrides, "group")
    knockout_predictions = apply_overrides(knockout_predictions, overrides, "knockout")

    group_predictions["winning_team"] = group_predictions.apply(
        lambda r: ("home" if r["predicted_home_goals"] > r["predicted_away_goals"]
                   else ("away" if r["predicted_away_goals"] > r["predicted_home_goals"]
                         else "draw")),
        axis=1,
    )

    # 7. Quality checks
    print("\n--- [7/9] Running quality checks ---")
    knockout_predictions["penalties"] = knockout_predictions["penalties"].astype(bool)
    passed = run_all_checks(
        group_predictions[_GROUP_OUTPUT_COLS],
        knockout_predictions[_KNOCKOUT_OUTPUT_COLS],
        full_ko_preds=knockout_predictions,
        group_tables=group_tables,
        ratings=ratings,
    )

    # 8. Save outputs
    print("\n--- [8/9] Saving outputs ---")
    _save_outputs(
        group_predictions, knockout_predictions,
        group_tables, bracket_summary, ratings, has_external,
    )

    # 9. Optional simulation + post-run sanity check
    if run_sim:
        print(f"\n--- [9/9] Monte Carlo simulation ({n_sims:,} runs) ---")
        from src.simulation import run_simulation
        sim_df = run_simulation(group_fixtures, knockout_slots, ratings, n=n_sims)
        _sanity_check_champion(knockout_predictions, sim_df)
    else:
        print("\n--- [9/9] Simulation skipped (pass --sim to enable) ---")

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Outputs saved to: {OUTPUTS_DIR}")
    if not passed:
        print("WARNING: Quality checks found errors. See submission_quality_checks.txt")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanity_check_groups(
    qualifiers,
    ratings: dict,
    group_tables,
) -> None:
    """
    Warns if any top-10 rated team fails to finish in the top 2 of their group.
    """
    top10 = sorted(ratings.items(), key=lambda x: -x[1])[:10]
    top10_teams = {t for t, _ in top10}

    warnings_found = []
    for team in top10_teams:
        # Find this team's position in their group table
        row = group_tables[group_tables["team"] == team]
        if row.empty:
            continue
        pos = int(row["position"].values[0])
        grp = row["group"].values[0]
        if pos > 2:
            warnings_found.append(
                f"  SANITY WARN: {team} (top-10 rated) finished P{pos} in Group {grp}"
            )

    if warnings_found:
        print("\n[sanity] Group stage warnings:")
        for w in warnings_found:
            print(w)
    else:
        print("[sanity] All top-10 rated teams finished top 2 in their group.")


def _sanity_check_champion(knockout_predictions, sim_df) -> None:
    """
    Warns if the deterministic champion is outside the top-10 simulation probabilities.
    """
    final = knockout_predictions[knockout_predictions["round"] == "Final"]
    if final.empty:
        return
    row = final.iloc[0]
    det_champion = (row["predicted_home_team"] if row["match_winner"] == "home"
                    else row["predicted_away_team"])

    top10_sim = set(sim_df.head(10)["team"].tolist())
    print(f"\n[sanity] Deterministic champion: {det_champion}")
    if det_champion not in top10_sim:
        print(f"  SANITY WARN: {det_champion} is NOT in the top-10 simulation champions.")
        print(f"  Top-10 sim champions: {list(sim_df.head(10)['team'])}")
    else:
        rank = sim_df[sim_df["team"] == det_champion].index[0] + 1
        prob = sim_df[sim_df["team"] == det_champion]["champion_prob_pct"].values[0]
        print(f"  Champion is sim rank #{rank} with {prob:.1f}% probability. OK.")


def _print_top_ratings(ratings: dict, n: int = 10) -> None:
    top = sorted(ratings.items(), key=lambda x: -x[1])[:n]
    print(f"[main] Top {n} team ratings:")
    for team, rating in top:
        print(f"         {team:<25} {rating:.1f}")


def _save_outputs(
    group_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
    group_tables: pd.DataFrame,
    bracket_summary: pd.DataFrame,
    ratings: dict,
    has_external: bool,
) -> None:
    g_out = group_predictions[[c for c in _GROUP_OUTPUT_COLS if c in group_predictions.columns]]
    k_out = knockout_predictions[[c for c in _KNOCKOUT_OUTPUT_COLS if c in knockout_predictions.columns]]

    g_out.to_csv(OUTPUT_FILES["group_predictions"], index=False)
    print(f"[main] Saved -> {OUTPUT_FILES['group_predictions']}")

    k_out.to_csv(OUTPUT_FILES["knockout_predictions"], index=False)
    print(f"[main] Saved -> {OUTPUT_FILES['knockout_predictions']}")

    group_tables.to_csv(OUTPUT_FILES["group_tables"], index=False)
    print(f"[main] Saved -> {OUTPUT_FILES['group_tables']}")

    bracket_summary.to_csv(OUTPUT_FILES["bracket_summary"], index=False)
    print(f"[main] Saved -> {OUTPUT_FILES['bracket_summary']}")

    generate_model_report(g_out, k_out, group_tables, ratings, has_external)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FIFA WC 2026 prediction pipeline")
    parser.add_argument("--online",       action="store_true",
                        help="Attempt to download external data before running")
    parser.add_argument("--refresh-data", action="store_true", dest="refresh_data",
                        help="Force re-fetch/rebuild all processed data sources")
    parser.add_argument("--sim",          action="store_true",
                        help="Run Monte Carlo simulation after main predictions")
    parser.add_argument("--n-sims",       type=int, default=DEFAULT_SIM_N, dest="n_sims",
                        help=f"Number of Monte Carlo simulations (default: {DEFAULT_SIM_N})")
    args = parser.parse_args()
    run_pipeline(
        online=args.online,
        refresh_data=args.refresh_data,
        run_sim=args.sim,
        n_sims=args.n_sims,
    )


if __name__ == "__main__":
    main()
