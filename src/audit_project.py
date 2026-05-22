"""
Project audit script.

Usage:
    python -m src.audit_project

Checks:
  - Required files exist
  - Processed data is populated
  - Placeholder leakage in outputs
  - Top ratings look sane
  - Recent outputs exist
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK  " if ok else "FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{detail_str}")
    return ok


def audit_files() -> int:
    """Returns number of failures."""
    print("\n=== File Existence ===")
    failures = 0

    required = [
        ("data/group_fixtures.csv",          "Raw group fixtures"),
        ("data/knockout_slots.csv",           "Knockout slot definitions"),
        ("data/external/playoff_results.csv", "Playoff results mapping"),
        ("data/external/international_results.csv", "Historical match results"),
        ("data/processed/current_fifa_rankings.csv", "FIFA rankings (processed)"),
        ("data/processed/current_world_elo.csv",     "World Elo (processed)"),
        ("data/processed/tournament_pedigree_scores.csv", "Tournament pedigree (processed)"),
        ("data/processed/playoff_mapping.csv",       "Playoff mapping (processed)"),
        ("data/processed/group_fixtures_resolved.csv", "Resolved fixtures (processed)"),
        ("data/processed/team_form_features.csv",    "Form features (processed)"),
        ("data/processed/team_ratings_final.csv",    "Final ratings (processed)"),
        ("outputs/group_predictions.csv",    "Group predictions output"),
        ("outputs/knockout_predictions.csv", "Knockout predictions output"),
        ("outputs/group_tables.csv",         "Group tables output"),
        ("outputs/bracket_summary.csv",      "Bracket summary output"),
        ("src/data_sources/__init__.py",     "data_sources package"),
        ("src/data_sources/fifa_rankings.py","FIFA rankings module"),
        ("src/data_sources/world_elo.py",    "World Elo module"),
        ("src/data_sources/playoff_resolver.py", "Playoff resolver module"),
        ("src/data_sources/tournament_achievements.py", "Tournament achievements module"),
        ("src/data_sources/refresh_all.py",  "refresh_all module"),
        ("README.md",                        "README"),
        (".gitignore",                       ".gitignore"),
        (".env.example",                     ".env.example"),
    ]

    for rel_path, label in required:
        ok = (ROOT / rel_path).exists()
        if not _check(label, ok, rel_path):
            failures += 1

    return failures


def audit_placeholders() -> int:
    """Check no placeholder team names leaked into outputs."""
    print("\n=== Placeholder Leakage ===")
    import pandas as pd
    PLACEHOLDERS = {
        "UEFA Playoff A", "UEFA Playoff B", "UEFA Playoff C", "UEFA Playoff D",
        "FIFA Playoff 1", "FIFA Playoff 2",
    }
    failures = 0

    checks = [
        ("outputs/group_predictions.csv",    ["home_team", "away_team"]),
        ("outputs/group_tables.csv",         ["team"]),
        ("outputs/knockout_predictions.csv", ["predicted_home_team", "predicted_away_team"]),
    ]
    for rel_path, cols in checks:
        path = ROOT / rel_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in cols:
            if col not in df.columns:
                continue
            leaked = set(df[col].astype(str)).intersection(PLACEHOLDERS)
            ok = len(leaked) == 0
            if not _check(f"No placeholders in {rel_path}[{col}]", ok,
                          f"leaked: {leaked}" if not ok else ""):
                failures += 1

    return failures


def audit_ratings() -> int:
    """Check top ratings look realistic."""
    print("\n=== Rating Sanity ===")
    import pandas as pd
    failures = 0

    path = ROOT / "data/processed/team_ratings_final.csv"
    if not path.exists():
        _check("team_ratings_final.csv exists", False)
        return 1

    df = pd.read_csv(path)

    # Component columns present
    for col in ("fifa_component", "elo_component", "form_component",
                "market_component",
                "tournament_pedigree_component", "sources_used", "rating_explanation"):
        ok = col in df.columns
        if not _check(f"Column '{col}' present in team_ratings_final", ok):
            failures += 1

    # Top 5 teams are sensible
    top5 = df.head(5)["team"].tolist()
    expected_elite = {"France", "Argentina", "Brazil", "England", "Spain"}
    overlap = len(set(top5).intersection(expected_elite))
    ok = overlap >= 3
    if not _check(f"At least 3 elite teams in top 5 (got {top5[:5]})", ok):
        failures += 1

    # Spain not outside top 6
    if "team" in df.columns:
        df_sorted = df.sort_values("final_rating", ascending=False).reset_index(drop=True)
        spain_rank = df_sorted[df_sorted["team"] == "Spain"].index
        if len(spain_rank) > 0:
            ok = int(spain_rank[0]) < 6
            if not _check(f"Spain in top 6 (rank {int(spain_rank[0]) + 1})", ok):
                failures += 1

    # Playoff teams present
    PLAYOFF_TEAMS = {"Bosnia and Herzegovina", "Sweden", "Turkey", "Czechia", "DR Congo", "Iraq"}
    teams_in_file = set(df["team"].astype(str))
    for team in PLAYOFF_TEAMS:
        ok = team in teams_in_file
        if not _check(f"Playoff team '{team}' in ratings", ok):
            failures += 1

    return failures


def audit_outputs() -> int:
    """Check output files have expected row counts."""
    print("\n=== Output Row Counts ===")
    import pandas as pd
    failures = 0

    checks = [
        ("outputs/group_predictions.csv",    "group_predictions",    72),
        ("outputs/knockout_predictions.csv", "knockout_predictions", 32),
        ("outputs/group_tables.csv",         "group_tables",         48),
    ]
    for rel_path, label, expected in checks:
        path = ROOT / rel_path
        if not path.exists():
            _check(f"{label} exists", False)
            failures += 1
            continue
        df   = pd.read_csv(path)
        rows = len(df)
        ok   = rows == expected
        if not _check(f"{label}: {rows} rows (expected {expected})", ok):
            failures += 1

    return failures


def main() -> None:
    print("=" * 55)
    print("FIFA WC 2026 Model — Project Audit")
    print("=" * 55)

    total_failures = 0
    total_failures += audit_files()
    total_failures += audit_placeholders()
    total_failures += audit_ratings()
    total_failures += audit_outputs()

    print("\n" + "=" * 55)
    if total_failures == 0:
        print("Audit PASSED — all checks OK")
    else:
        print(f"Audit FAILED — {total_failures} check(s) failed")
    print("=" * 55 + "\n")
    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
