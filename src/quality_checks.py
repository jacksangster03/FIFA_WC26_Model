"""
Submission quality checks and realism diagnostics.

Validates all required columns, types, value ranges, and counts.
Writes a human-readable report to outputs/submission_quality_checks.txt.
"""

from __future__ import annotations

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

def realism_diagnostics(
    group_df: pd.DataFrame,
    knockout_df: pd.DataFrame,
    full_ko_df: Optional[pd.DataFrame] = None,
) -> list[str]:
    """
    Computes descriptive statistics for human review.
    Returns a list of formatted lines.
    """
    lines = []

    # Group stage
    total_g     = len(group_df)
    draws_g     = (group_df["winning_team"] == "draw").sum()
    home_wins_g = (group_df["winning_team"] == "home").sum()
    away_wins_g = (group_df["winning_team"] == "away").sum()
    total_goals_g = (group_df["predicted_home_goals"] + group_df["predicted_away_goals"]).sum()
    avg_goals_g   = total_goals_g / total_g if total_g else 0
    zero_zero_g   = ((group_df["predicted_home_goals"] == 0) &
                     (group_df["predicted_away_goals"] == 0)).sum()
    one_zero_g    = (((group_df["predicted_home_goals"] == 1) &
                      (group_df["predicted_away_goals"] == 0)) |
                     ((group_df["predicted_home_goals"] == 0) &
                      (group_df["predicted_away_goals"] == 1))).sum()

    lines.append("GROUP STAGE REALISM")
    lines.append(f"  Matches:          {total_g}")
    lines.append(f"  Home wins:        {home_wins_g}  ({home_wins_g/total_g*100:.1f}%)")
    lines.append(f"  Draws:            {draws_g}  ({draws_g/total_g*100:.1f}%)  "
                 f"[realistic target: 18-25]")
    lines.append(f"  Away wins:        {away_wins_g}  ({away_wins_g/total_g*100:.1f}%)")
    lines.append(f"  Total goals:      {total_goals_g}")
    lines.append(f"  Avg goals/match:  {avg_goals_g:.2f}  [historical WC avg ~2.6]")
    lines.append(f"  0-0 scorelines:   {zero_zero_g}")
    lines.append(f"  1-0 or 0-1:       {one_zero_g}")

    # Knockout
    total_ko    = len(knockout_df)
    pens        = knockout_df["penalties"].sum() if "penalties" in knockout_df.columns else 0
    total_goals_ko = (knockout_df["predicted_home_goals"] +
                      knockout_df["predicted_away_goals"]).sum()
    avg_goals_ko   = total_goals_ko / total_ko if total_ko else 0

    lines.append("\nKNOCKOUT STAGE REALISM")
    lines.append(f"  Matches:               {total_ko}")
    lines.append(f"  Avg goals/match:       {avg_goals_ko:.2f}  [historical KO avg ~2.3]")
    lines.append(f"  Penalty matches:       {pens}")

    # R32 unique team count
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
    diag_lines             = realism_diagnostics(group_preds, knockout_preds, full_ko_preds)

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
    out_path.write_text("".join(report_lines))

    print(f"\n[quality_checks] Report written -> {out_path}")
    passed = len(all_errors) == 0
    print(f"\nOverall status: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60 + "\n")
    return passed
