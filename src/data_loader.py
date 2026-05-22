"""
Loads and validates the competition fixture files.

For group fixtures the loader prefers the resolved file
(data/processed/group_fixtures_resolved.csv) when it exists, so that
playoff placeholders are substituted with actual team names.
"""

import sys
import pandas as pd
from src.config import GROUP_FIXTURES_PATH, KNOCKOUT_SLOTS_PATH, PROCESSED_DIR

_RESOLVED_FIXTURES_PATH = PROCESSED_DIR / "group_fixtures_resolved.csv"

# Required columns for each file
GROUP_REQUIRED = {"match_id", "group", "home_team", "away_team", "date_utc", "venue"}
KNOCKOUT_REQUIRED = {
    "match_id", "round", "multiplier", "date_utc",
    "venue", "slot_home", "slot_away",
}


def _validate_columns(df: pd.DataFrame, required: set, label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] {label} is missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)


def load_group_fixtures() -> pd.DataFrame:
    # Prefer resolved fixtures (playoff placeholders substituted)
    if _RESOLVED_FIXTURES_PATH.exists():
        path  = _RESOLVED_FIXTURES_PATH
        label = "group_fixtures_resolved.csv"
    elif GROUP_FIXTURES_PATH.exists():
        path  = GROUP_FIXTURES_PATH
        label = "group_fixtures.csv"
    else:
        print(f"[ERROR] Cannot find {GROUP_FIXTURES_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    _validate_columns(df, GROUP_REQUIRED, label)
    df["match_id"] = df["match_id"].astype(int)

    teams = set(df["home_team"]) | set(df["away_team"])
    placeholders = [t for t in teams
                    if "Playoff" in t or t.startswith("UEFA ") or t.startswith("FIFA ")]
    resolved_note = " [resolved]" if _RESOLVED_FIXTURES_PATH.exists() else " [unresolved]"
    print(f"[data_loader] group_fixtures{resolved_note}: {df.shape[0]} rows, "
          f"{df['group'].nunique()} groups, {len(teams)} unique team slots"
          + (f", {len(placeholders)} unresolved placeholders" if placeholders else ""))
    return df


def load_knockout_slots() -> pd.DataFrame:
    if not KNOCKOUT_SLOTS_PATH.exists():
        print(f"[ERROR] Cannot find {KNOCKOUT_SLOTS_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(KNOCKOUT_SLOTS_PATH)
    _validate_columns(df, KNOCKOUT_REQUIRED, "knockout_slots.csv")
    df["match_id"] = df["match_id"].astype(int)
    df["multiplier"] = df["multiplier"].astype(int)

    print(f"[data_loader] knockout_slots: {df.shape[0]} rows, "
          f"rounds: {df['round'].unique().tolist()}")
    return df


def all_group_teams(fixtures: pd.DataFrame) -> list[str]:
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    return teams


def group_map(fixtures: pd.DataFrame) -> dict[str, str]:
    """Returns {team_name: group_letter} for all teams."""
    mapping: dict[str, str] = {}
    for _, row in fixtures.iterrows():
        mapping[row["home_team"]] = row["group"]
        mapping[row["away_team"]] = row["group"]
    return mapping
