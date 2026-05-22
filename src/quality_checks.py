"""
Submission quality checks and realism diagnostics.

Validates all required columns, types, value ranges, and counts.
Writes a human-readable report to outputs/submission_quality_checks.txt.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.config import (
    EXPECTED_GROUP_ROWS,
    EXPECTED_KNOCKOUT_ROWS,
    VALID_WINNER_VALUES,
    VALID_KO_WINNER_VALUES,
    OUTPUT_FILES,
)

_GROUP_REQUIRED_COLS = [
    "match_id", "group", "home_team", "away_team", "date_utc", "venue",
    "predicted_home_goals", "predicted_away_goals", "corners",
    "yellow_cards", "red_cards", "winning_team",
]

_KNOCKOUT_REQUIRED_COLS = [
    "match_id", "round", "multiplier", "date_utc", "venue",
    "slot_home", "slot_away",
    "predicted_home_team", "predicted_away_team",
    "predicted_home_goals", "predicted_away_goals", "corners",
    "yellow_cards", "red_cards", "match_winner", "penalties",
]


def _check(condition: bool, msg: str, errors: list, warnings_: list, *, warn: bool = False) -> None:
    if condition:
        if warn:
            warnings_.append(f"  WARN  {msg}")
        else:
            errors.append(f"  ERROR {msg}")


# ---------------------------------------------------------------------------
# Group stage checks
# ---------------------------------------------------------------------------

def check_group_predictions(df: pd.DataFrame) -> tuple[list, list]:
    errors:   list[str] = []
    warnings: list[str] = []

    _check(len(df) != EXPECTED_GROUP_ROWS,
           f"Expected {EXPECTED_GROUP_ROWS} group rows, got {len(df)}", errors, warnings)

    missing = set(_GROUP_REQUIRED_COLS) - set(df.columns)
    _check(bool(missing), f"Missing columns: {missing}", errors, warnings)
    if missing:
        return errors, warnings

    for col in _GROUP_REQUIRED_COLS:
        nulls = df[col].isna().sum()
        _check(nulls > 0, f"Column '{col}' has {nulls} null value(s)", errors, warnings)

    bad = df[~df["winning_team"].isin(VALID_WINNER_VALUES)]
    _check(len(bad) > 0,
           f"Invalid winning_team values: {bad['winning_team'].unique().tolist()}",
           errors, warnings)

    for col in ("predicted_home_goals", "predicted_away_goals"):
        _check((df[col] < 0).any(), f"{col} has negative values", errors, warnings)
        _check((df[col] > 8).any(), f"{col} has suspiciously large values (>8)",
               errors, warnings, warn=True)

    _check((df["corners"] < 3).any(), "corners < 3 in some matches", errors, warnings, warn=True)
    _check((df["corners"] > 20).any(), "corners > 20 in some matches", errors, warnings, warn=True)
    _check((df["yellow_cards"] < 0).any(), "yellow_cards < 0", errors, warnings)
    _check((df["yellow_cards"] > 12).any(), "yellow_cards > 12", errors, warnings, warn=True)
    _check((df["red_cards"] < 0).any(), "red_cards < 0", errors, warnings)
    _check((df["red_cards"] > 3).any(), "red_cards > 3", errors, warnings, warn=True)

    for _, row in df.iterrows():
        hg, ag, wt = row["predicted_home_goals"], row["predicted_away_goals"], row["winning_team"]
        exp = "home" if hg > ag else ("away" if ag > hg else "draw")
        _check(wt != exp,
               f"Match {row['match_id']}: {hg}-{ag} but winning_team='{wt}' (expected '{exp}')",
               errors, warnings)

    return errors, warnings


# ---------------------------------------------------------------------------
# Knockout stage checks
# ---------------------------------------------------------------------------

def check_knockout_predictions(df: pd.DataFrame) -> tuple[list, list]:
    errors:   list[str] = []
    warnings: list[str] = []

    _check(len(df) != EXPECTED_KNOCKOUT_ROWS,
           f"Expected {EXPECTED_KNOCKOUT_ROWS} knockout rows, got {len(df)}", errors, warnings)

    missing = set(_KNOCKOUT_REQUIRED_COLS) - set(df.columns)
    _check(bool(missing), f"Missing columns: {missing}", errors, warnings)
    if missing:
        return errors, warnings

    for col in _KNOCKOUT_REQUIRED_COLS:
        nulls = df[col].isna().sum()
        _check(nulls > 0, f"Column '{col}' has {nulls} null value(s)", errors, warnings)

    bad = df[~df["match_winner"].isin(VALID_KO_WINNER_VALUES)]
    _check(len(bad) > 0,
           f"Invalid match_winner values: {bad['match_winner'].unique().tolist()}",
           errors, warnings)

    for col in ("predicted_home_goals", "predicted_away_goals"):
        _check((df[col] < 0).any(), f"{col} has negative values", errors, warnings)
        _check((df[col] > 7).any(), f"{col} > 7 in knockout", errors, warnings, warn=True)

    drawn = df[df["predicted_home_goals"] == df["predicted_away_goals"]]
    _check((~drawn["penalties"]).any(),
           f"{(~drawn['penalties']).sum()} knockout draws have penalties=False",
           errors, warnings)

    for _, row in df.iterrows():
        hg, ag, mw = row["predicted_home_goals"], row["predicted_away_goals"], row["match_winner"]
        if hg != ag:
            exp_mw = "home" if hg > ag else "away"
            _check(mw != exp_mw,
                   f"Match {row['match_id']}: {hg}-{ag} but match_winner='{mw}' "
                   f"(expected '{exp_mw}')", errors, warnings)

    for rnd, rnd_df in df.groupby("round"):
        all_teams = list(rnd_df["predicted_home_team"]) + list(rnd_df["predicted_away_team"])
        seen: set = set()
        for team in all_teams:
            if team in seen and not team.startswith(
                ("Unknown", "Winner", "Runner", "Best", "Loser")
            ):
                _check(True, f"Team '{team}' appears >1× in {rnd}",
                       errors, warnings, warn=True)
            seen.add(team)

    return errors, warnings


# ---------------------------------------------------------------------------
# Realism diagnostics
# ---------------------------------------------------------------------------

def _scoreline_count(df: pd.DataFrame, h: int, a: int) -> int:
    """Count matches with this exact scoreline (or its mirror)."""
    exact  = ((df["predicted_home_goals"] == h) & (df["predicted_away_goals"] == a)).sum()
    mirror = ((df["predicted_home_goals"] == a) & (df["predicted_away_goals"] == h)).sum() if h != a else 0
    return int(exact + mirror)


def realism_diagnostics(
    group_df: pd.DataFrame,
    knockout_df: pd.DataFrame,
    full_ko_df: Optional[pd.DataFrame] = None,
    ratings: Optional[dict] = None,
) -> list[str]:
    """
    Extended realism diagnostics with scoreline distribution, elite-favourite
    win rate, corners/cards distribution, and red-card breakdown.
    """
    lines    = []
    warnings = []

    # ---- Group stage overview -------------------------------------------
    total_g     = len(group_df)
    draws_g     = (group_df["winning_team"] == "draw").sum()
    home_wins_g = (group_df["winning_team"] == "home").sum()
    away_wins_g = (group_df["winning_team"] == "away").sum()
    total_goals_g = (group_df["predicted_home_goals"] + group_df["predicted_away_goals"]).sum()
    avg_goals_g   = total_goals_g / total_g if total_g else 0

    lines.append("GROUP STAGE REALISM")
    lines.append(f"  Matches:          {total_g}")
    lines.append(f"  Home wins:        {home_wins_g}  ({home_wins_g/total_g*100:.1f}%)")
    lines.append(f"  Draws:            {draws_g}  ({draws_g/total_g*100:.1f}%)  [target: 18-25]")
    lines.append(f"  Away wins:        {away_wins_g}  ({away_wins_g/total_g*100:.1f}%)")
    lines.append(f"  Total goals:      {total_goals_g}")
    lines.append(f"  Avg goals/match:  {avg_goals_g:.2f}  [historical WC avg ~2.6]")

    # ---- Scoreline distribution ------------------------------------------
    lines.append("\n  SCORELINE DISTRIBUTION (group)")
    for label, (h, a) in [
        ("0-0",     (0, 0)), ("1-1",     (1, 1)), ("2-2",     (2, 2)),
        ("1-0/0-1", (1, 0)), ("2-0/0-2", (2, 0)), ("2-1/1-2", (2, 1)),
        ("3-0/0-3", (3, 0)), ("3-1/1-3", (3, 1)),
    ]:
        cnt = _scoreline_count(group_df, h, a)
        lines.append(f"    {label:<10}: {cnt}")

    # ---- Elite-favourite diagnostics ------------------------------------
    if ratings is not None and "home_team" in group_df.columns:
        lines.append("\n  ELITE FAVOURITE WIN RATE")
        elites, wins = 0, 0
        for _, row in group_df.iterrows():
            hr = ratings.get(row["home_team"], 55.0)
            ar = ratings.get(row["away_team"], 55.0)
            gap = abs(hr - ar)
            if gap >= 15:
                elites += 1
                fav_side = "home" if hr > ar else "away"
                if row["winning_team"] == fav_side:
                    wins += 1
        pct = wins / elites * 100 if elites else 0.0
        lines.append(f"    Elite matches (gap>=15): {elites}")
        lines.append(f"    Favourite wins:          {wins}  ({pct:.0f}%)")
        if elites > 0 and pct < 75:
            warnings.append(f"  WARN  Elite favourite win rate {pct:.0f}% < 75% target")

    # ---- Knockout overview ----------------------------------------------
    total_ko = len(knockout_df)
    pens     = int(knockout_df["penalties"].sum()) if "penalties" in knockout_df.columns else 0
    total_goals_ko = (knockout_df["predicted_home_goals"] + knockout_df["predicted_away_goals"]).sum()
    avg_goals_ko   = total_goals_ko / total_ko if total_ko else 0

    lines.append("\nKNOCKOUT STAGE REALISM")
    lines.append(f"  Matches:               {total_ko}")
    lines.append(f"  Avg goals/match:       {avg_goals_ko:.2f}  [historical KO avg ~2.3]")
    lines.append(f"  Penalty matches:       {pens}")

    # ---- R32 integrity --------------------------------------------------
    if full_ko_df is not None:
        r32 = full_ko_df[full_ko_df["round"] == "Round of 32"]
        if not r32.empty:
            r32_teams = list(r32["predicted_home_team"]) + list(r32["predicted_away_team"])
            real_teams = [t for t in r32_teams
                          if not t.startswith(("Unknown", "Winner", "Runner", "Best"))]
            unique_real = len(set(real_teams))
            placeholder_count = len(r32_teams) - len(real_teams)
            lines.append(f"\nROUND OF 32 INTEGRITY")
            lines.append(f"  Total team slots:     {len(r32_teams)}")
            lines.append(f"  Resolved teams:       {len(real_teams)}")
            lines.append(f"  Unique resolved:      {unique_real}  "
                         f"[expected 32 - placeholders = {32 - placeholder_count}]")
            dup_ok = unique_real == len(real_teams)
            lines.append(f"  No duplicates:        {'YES' if dup_ok else 'NO (check Best 3rd assignment)'}")

    # ---- Combined corners diagnostics -----------------------------------
    all_df = pd.concat([group_df, knockout_df], ignore_index=True) if not knockout_df.empty else group_df
    if "corners" in all_df.columns:
        lines.append("\nCORNERS DIAGNOSTICS")
        g_corners  = group_df["corners"].mean()    if "corners" in group_df.columns else float("nan")
        ko_corners = knockout_df["corners"].mean() if "corners" in knockout_df.columns and total_ko > 0 else float("nan")
        in_range   = all_df["corners"].between(7, 12).sum()
        too_low    = (all_df["corners"] < 6).sum()
        too_high   = (all_df["corners"] > 14).sum()
        pct_range  = in_range / len(all_df) * 100 if len(all_df) > 0 else 0

        lines.append(f"  Avg group corners:    {g_corners:.2f}")
        lines.append(f"  Avg KO corners:       {ko_corners:.2f}")
        lines.append(f"  In 7-12 range:        {in_range}/{len(all_df)} ({pct_range:.0f}%)")
        lines.append(f"  Below 6:              {too_low}")
        lines.append(f"  Above 14:             {too_high}")

        dist = all_df["corners"].value_counts().sort_index()
        lines.append("  Distribution:         " + "  ".join(f"{v}={cnt}" for v, cnt in dist.items()))

        if not math.isnan(g_corners) and not (7 <= g_corners <= 12):
            warnings.append(f"  WARN  Avg group corners {g_corners:.2f} outside 7-12 target")
        if pct_range < 70:
            warnings.append(f"  WARN  Only {pct_range:.0f}% of matches have corners 7-12 (target >=70%)")

    # ---- Yellow cards diagnostics ---------------------------------------
    if "yellow_cards" in all_df.columns:
        lines.append("\nYELLOW CARDS DIAGNOSTICS")
        g_yel  = group_df["yellow_cards"].mean()    if "yellow_cards" in group_df.columns else float("nan")
        ko_yel = knockout_df["yellow_cards"].mean() if "yellow_cards" in knockout_df.columns and total_ko > 0 else float("nan")
        in_yrange  = all_df["yellow_cards"].between(2, 6).sum()
        too_few    = (all_df["yellow_cards"] < 1).sum()
        too_many   = (all_df["yellow_cards"] > 8).sum()
        pct_yrange = in_yrange / len(all_df) * 100 if len(all_df) > 0 else 0

        lines.append(f"  Avg group yellows:    {g_yel:.2f}")
        lines.append(f"  Avg KO yellows:       {ko_yel:.2f}")
        lines.append(f"  In 2-6 range:         {in_yrange}/{len(all_df)} ({pct_yrange:.0f}%)")
        lines.append(f"  Below 1:              {too_few}")
        lines.append(f"  Above 8:              {too_many}")

        ydist = all_df["yellow_cards"].value_counts().sort_index()
        lines.append("  Distribution:         " + "  ".join(f"{v}={cnt}" for v, cnt in ydist.items()))

        if not math.isnan(g_yel) and not (2 <= g_yel <= 6):
            warnings.append(f"  WARN  Avg group yellow cards {g_yel:.2f} outside 2-6 target")
        if pct_yrange < 70:
            warnings.append(f"  WARN  Only {pct_yrange:.0f}% of matches have yellows 2-6 (target >=70%)")

    # ---- Red cards diagnostics ------------------------------------------
    if "red_cards" in all_df.columns:
        total_reds  = int(all_df["red_cards"].sum())
        group_reds  = int(group_df["red_cards"].sum())   if "red_cards" in group_df.columns else 0
        ko_reds     = int(knockout_df["red_cards"].sum()) if "red_cards" in knockout_df.columns else 0
        multi_reds  = int((all_df["red_cards"] > 1).sum())

        lines.append("\nRED CARDS DIAGNOSTICS")
        lines.append(f"  Total red cards:      {total_reds}")
        lines.append(f"  Group stage reds:     {group_reds}")
        lines.append(f"  Knockout reds:        {ko_reds}")
        lines.append(f"  Matches with >1 red:  {multi_reds}")

        if total_reds > 4:
            warnings.append(f"  WARN  Total red cards {total_reds} exceeds target of 0-4")
        if multi_reds > 0:
            warnings.append(f"  WARN  {multi_reds} match(es) have >1 red card")
        if group_reds > 2:
            warnings.append(f"  WARN  Group stage has {group_reds} red cards (target <=2)")

    if warnings:
        lines.append("\nDIAGNOSTIC WARNINGS")
        lines.extend(warnings)

    return lines


# ---------------------------------------------------------------------------
# Data source checks
# ---------------------------------------------------------------------------

_PLAYOFF_PLACEHOLDERS = {
    "UEFA Playoff A", "UEFA Playoff B", "UEFA Playoff C", "UEFA Playoff D",
    "FIFA Playoff 1", "FIFA Playoff 2",
}
_ACTUAL_PLAYOFF_TEAMS = {
    "Bosnia and Herzegovina", "Sweden", "Turkey", "Czechia", "DR Congo", "Iraq",
}


def check_data_sources() -> tuple[list, list]:
    """Warns if expected processed data files are absent or if sources are missing."""
    from src.config import PROCESSED_DIR
    errors:   list[str] = []
    warnings: list[str] = []

    required = {
        "current_fifa_rankings.csv": "FIFA rankings",
        "current_world_elo.csv":     "World Elo ratings",
        "tournament_pedigree_scores.csv": "Tournament pedigree",
        "playoff_mapping.csv":       "Playoff mapping",
        "group_fixtures_resolved.csv": "Resolved fixtures",
        "team_form_features.csv":    "Form features",
        "team_ratings_final.csv":    "Final ratings",
    }
    for fname, label in required.items():
        path = PROCESSED_DIR / fname
        _check(not path.exists(), f"Missing processed file: {fname} ({label})",
               errors, warnings, warn=True)

    # Check team_ratings_final for required component columns
    ratings_path = PROCESSED_DIR / "team_ratings_final.csv"
    if ratings_path.exists():
        try:
            df = pd.read_csv(ratings_path)
            for col in ("fifa_component", "elo_component", "form_component",
                        "tournament_pedigree_component", "sources_used", "rating_explanation"):
                _check(col not in df.columns,
                       f"team_ratings_final.csv missing column: {col}",
                       errors, warnings, warn=True)
        except Exception:
            pass

    # Check form features for actual playoff teams
    form_path = PROCESSED_DIR / "team_form_features.csv"
    if form_path.exists():
        try:
            form_df = pd.read_csv(form_path)
            form_teams = set(form_df["team"].astype(str))
            for team in _ACTUAL_PLAYOFF_TEAMS:
                _check(team not in form_teams,
                       f"Playoff team '{team}' missing from form features",
                       errors, warnings, warn=True)
        except Exception:
            pass

    return errors, warnings


def check_placeholder_leakage(
    group_preds: pd.DataFrame,
    group_tables: Optional[pd.DataFrame],
    knockout_preds: pd.DataFrame,
) -> tuple[list, list]:
    """Fails if resolved playoff placeholders appear in output files."""
    errors:   list[str] = []
    warnings: list[str] = []

    for col in ("home_team", "away_team"):
        if col in group_preds.columns:
            leaked = set(group_preds[col]).intersection(_PLAYOFF_PLACEHOLDERS)
            _check(bool(leaked),
                   f"Unresolved placeholders in group_predictions.csv [{col}]: {leaked}",
                   errors, warnings)

    if group_tables is not None:
        for col in ("team",):
            if col in group_tables.columns:
                leaked = set(group_tables[col]).intersection(_PLAYOFF_PLACEHOLDERS)
                _check(bool(leaked),
                       f"Unresolved placeholders in group_tables.csv [{col}]: {leaked}",
                       errors, warnings)

    for col in ("predicted_home_team", "predicted_away_team"):
        if col in knockout_preds.columns:
            leaked = set(knockout_preds[col]).intersection(_PLAYOFF_PLACEHOLDERS)
            _check(bool(leaked),
                   f"Unresolved placeholders in knockout_predictions.csv [{col}]: {leaked}",
                   errors, warnings)

    return errors, warnings


def check_rating_sanity(ratings: Optional[dict] = None) -> tuple[list, list]:
    """
    Warns if top-5 FIFA-ranked team is outside the top 8 final ratings,
    and if Spain is materially below closely-ranked peers without explanation.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    from src.config import PROCESSED_DIR

    # Check top-5 FIFA ranked teams appear in top 8 final ratings
    fifa_path = PROCESSED_DIR / "current_fifa_rankings.csv"
    ratings_path = PROCESSED_DIR / "team_ratings_final.csv"
    if fifa_path.exists() and ratings_path.exists():
        try:
            fifa_df    = pd.read_csv(fifa_path).head(5)
            ratings_df = pd.read_csv(ratings_path)
            top8_teams = set(ratings_df.head(8)["team"].tolist())
            for _, row in fifa_df.iterrows():
                team = str(row["team"])
                _check(team not in top8_teams,
                       f"FIFA top-5 team '{team}' (rank {int(row['rank'])}) "
                       f"is outside final rating top 8",
                       errors, warnings, warn=True)
        except Exception:
            pass

    # Spain sanity: check not > 5 points below Belgium or Netherlands
    if ratings is not None:
        spain = ratings.get("Spain")
        for peer in ("Belgium", "Netherlands"):
            peer_r = ratings.get(peer)
            if spain is not None and peer_r is not None:
                _check(peer_r - spain > 5.0,
                       f"Spain ({spain:.1f}) is >5 pts below {peer} ({peer_r:.1f}) "
                       f"without clear justification",
                       errors, warnings, warn=True)

    return errors, warnings


# ---------------------------------------------------------------------------
# Calibration report (soft warnings only, never fails pipeline)
# ---------------------------------------------------------------------------

def check_calibration() -> list[str]:
    """
    Reads simulation_summary.csv and champion_probability_targets.csv.
    Returns a list of calibration warning strings.  Never errors or fails.
    Warnings only; calibration issues are flagged, not fatal.
    """
    from src.config import OUTPUT_FILES, EXTERNAL_DIR
    lines: list[str] = []

    sim_path    = OUTPUT_FILES["simulation_summary"]
    target_path = EXTERNAL_DIR / "champion_probability_targets.csv"

    if not sim_path.exists():
        lines.append("CALIBRATION: No simulation_summary.csv found; run --sim to generate")
        return lines

    try:
        sim_df = pd.read_csv(sim_path)
        sim_df = sim_df[sim_df["team"] != "_META_n_sims"]
        sim_map = dict(zip(sim_df["team"], sim_df["champion_prob_pct"].astype(float)))
    except Exception as exc:
        lines.append(f"CALIBRATION: Could not read simulation_summary.csv: {exc}")
        return lines

    # Absolute calibration checks (no targets file required)
    total = sum(sim_map.values())
    top2  = sorted(sim_map.values(), reverse=True)[:2]
    top2_combined = sum(top2)

    if total > 0:
        france_pct   = sim_map.get("France",    0.0)
        arg_pct      = sim_map.get("Argentina", 0.0)
        spain_pct    = sim_map.get("Spain",     0.0)
        portugal_pct = sim_map.get("Portugal",  0.0)
        germany_pct  = sim_map.get("Germany",   0.0)

        if france_pct > 20:
            lines.append(f"CALIBRATION WARN: France {france_pct:.1f}% > 20% target ceiling")
        if arg_pct > 15:
            lines.append(f"CALIBRATION WARN: Argentina {arg_pct:.1f}% > 15% target ceiling")
        if spain_pct < 12:
            lines.append(f"CALIBRATION WARN: Spain {spain_pct:.1f}% < 12% target floor")
        if portugal_pct < 4:
            lines.append(f"CALIBRATION WARN: Portugal {portugal_pct:.1f}% < 4% target floor")
        if germany_pct > 9:
            lines.append(f"CALIBRATION WARN: Germany {germany_pct:.1f}% > 9% target ceiling")
        if top2_combined > 35:
            lines.append(f"CALIBRATION WARN: Top-2 combined {top2_combined:.1f}% > 35% (over-concentration)")

    # Target-band checks if file available
    if target_path.exists():
        try:
            targets = pd.read_csv(target_path)
            for _, row in targets.iterrows():
                team = str(row["team"])
                lo   = float(row["target_low_pct"])
                hi   = float(row["target_high_pct"])
                mid  = float(row["target_mid_pct"])
                pct  = sim_map.get(team, None)
                if pct is None:
                    continue
                deviation = abs(pct - mid)
                if deviation > 2.0 and not (lo <= pct <= hi):
                    lines.append(
                        f"CALIBRATION WARN: {team} {pct:.1f}% outside target band "
                        f"[{lo}-{hi}%] (mid={mid}%, dev={deviation:.1f}pp)"
                    )
        except Exception:
            pass

    if not lines:
        lines.append("CALIBRATION: All champion probabilities within target bands")

    return lines


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_all_checks(
    group_preds: pd.DataFrame,
    knockout_preds: pd.DataFrame,
    full_ko_preds: Optional[pd.DataFrame] = None,
    group_tables: Optional[pd.DataFrame] = None,
    ratings: Optional[dict] = None,
) -> bool:
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    g_errors, g_warnings   = check_group_predictions(group_preds)
    k_errors, k_warnings   = check_knockout_predictions(knockout_preds)
    ds_errors, ds_warnings = check_data_sources()
    ph_errors, ph_warnings = check_placeholder_leakage(group_preds, group_tables, knockout_preds)
    rs_errors, rs_warnings = check_rating_sanity(ratings)
    diag_lines             = realism_diagnostics(group_preds, knockout_preds, full_ko_preds, ratings)
    calib_lines            = check_calibration()

    all_errors = g_errors + k_errors + ph_errors

    print(f"\nGroup stage ({len(g_errors)} errors, {len(g_warnings)} warnings):")
    for msg in g_errors + g_warnings:
        print(msg)
    if not g_errors and not g_warnings:
        print("  All checks PASSED")

    print(f"\nKnockout stage ({len(k_errors)} errors, {len(k_warnings)} warnings):")
    for msg in k_errors + k_warnings:
        print(msg)
    if not k_errors and not k_warnings:
        print("  All checks PASSED")

    print(f"\nData sources ({len(ds_errors)} errors, {len(ds_warnings)} warnings):")
    for msg in ds_errors + ds_warnings:
        print(msg)
    if not ds_errors and not ds_warnings:
        print("  All sources present")

    print(f"\nPlaceholder leakage ({len(ph_errors)} errors, {len(ph_warnings)} warnings):")
    for msg in ph_errors + ph_warnings:
        print(msg)
    if not ph_errors and not ph_warnings:
        print("  No placeholder leakage detected")

    print(f"\nRating sanity ({len(rs_errors)} errors, {len(rs_warnings)} warnings):")
    for msg in rs_errors + rs_warnings:
        print(msg)
    if not rs_errors and not rs_warnings:
        print("  Rating sanity OK")

    print("\nRealism diagnostics:")
    for line in diag_lines:
        print("  " + line)

    print("\nCalibration report:")
    for line in calib_lines:
        print("  " + line)

    # Write report
    out_path = OUTPUT_FILES["quality_checks"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "FIFA WC 2026 Prediction - Submission Quality Checks\n",
        "=" * 60 + "\n\n",
        f"Group stage: {len(g_errors)} errors, {len(g_warnings)} warnings\n",
    ]
    for msg in g_errors + g_warnings:
        report_lines.append(msg + "\n")
    if not g_errors and not g_warnings:
        report_lines.append("  All checks PASSED\n")
    report_lines.append(f"\nKnockout stage: {len(k_errors)} errors, {len(k_warnings)} warnings\n")
    for msg in k_errors + k_warnings:
        report_lines.append(msg + "\n")
    if not k_errors and not k_warnings:
        report_lines.append("  All checks PASSED\n")
    report_lines.append(f"\nData sources: {len(ds_errors)} errors, {len(ds_warnings)} warnings\n")
    for msg in ds_errors + ds_warnings:
        report_lines.append(msg + "\n")
    report_lines.append(f"\nPlaceholder leakage: {len(ph_errors)} errors, {len(ph_warnings)} warnings\n")
    for msg in ph_errors + ph_warnings:
        report_lines.append(msg + "\n")
    report_lines.append(f"\nRating sanity: {len(rs_errors)} errors, {len(rs_warnings)} warnings\n")
    for msg in rs_errors + rs_warnings:
        report_lines.append(msg + "\n")
    report_lines.append("\n" + "\n".join(diag_lines) + "\n")
    report_lines.append("\nCalibration Report:\n" + "\n".join(calib_lines) + "\n")
    out_path.write_text("".join(report_lines))

    print(f"\n[quality_checks] Report written -> {out_path}")
    passed = len(all_errors) == 0
    print(f"\nOverall status: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60 + "\n")
    return passed
