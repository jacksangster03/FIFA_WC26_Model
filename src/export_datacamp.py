"""
DataCamp competition export module.

Produces submission-ready files:
  outputs/datacamp_group_predictions.csv
  outputs/datacamp_knockout_predictions.csv
  outputs/datacamp_submission_notebook_cells.py
  outputs/datacamp_submission_checklist.txt

Run standalone:  python -m src.export_datacamp
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR, OUTPUT_FILES

# Required columns per competition spec
_GROUP_COLS = [
    "match_id",
    "group",
    "home_team",
    "away_team",
    "date_utc",
    "venue",
    "predicted_home_goals",
    "predicted_away_goals",
    "corners",
    "yellow_cards",
    "red_cards",
    "winning_team",
]

_KNOCKOUT_COLS = [
    "match_id",
    "round",
    "multiplier",
    "date_utc",
    "venue",
    "slot_home",
    "slot_away",
    "predicted_home_team",
    "predicted_away_team",
    "predicted_home_goals",
    "predicted_away_goals",
    "corners",
    "yellow_cards",
    "red_cards",
    "match_winner",
    "penalties",
]


def _normalise_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 'date' to 'date_utc' if needed; keep 'date_utc' if already present."""
    if "date_utc" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "date_utc"})
    return df


def _enforce_bool_penalties(df: pd.DataFrame) -> pd.DataFrame:
    """Convert penalties column to Python bool."""
    if "penalties" in df.columns:
        df["penalties"] = df["penalties"].astype(bool)
    return df


def _check_nulls(df: pd.DataFrame, label: str) -> list[str]:
    issues = []
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            issues.append(f"  {label}: column '{col}' has {n} null value(s)")
    return issues


def export_group_predictions(group_df: pd.DataFrame) -> pd.DataFrame:
    df = group_df.copy()
    df = _normalise_date_col(df)

    # Fill missing optional cols with sensible defaults
    for col in _GROUP_COLS:
        if col not in df.columns:
            if col in ("date_utc", "venue"):
                df[col] = ""
            else:
                df[col] = 0

    df = df[_GROUP_COLS]
    df["predicted_home_goals"] = df["predicted_home_goals"].astype(int)
    df["predicted_away_goals"] = df["predicted_away_goals"].astype(int)
    df["corners"]      = df["corners"].astype(int)
    df["yellow_cards"] = df["yellow_cards"].astype(int)
    df["red_cards"]    = df["red_cards"].astype(int)
    df["match_id"]     = df["match_id"].astype(int)
    return df


def export_knockout_predictions(knockout_df: pd.DataFrame) -> pd.DataFrame:
    df = knockout_df.copy()
    df = _normalise_date_col(df)
    df = _enforce_bool_penalties(df)

    for col in _KNOCKOUT_COLS:
        if col not in df.columns:
            if col in ("date_utc", "venue", "slot_home", "slot_away"):
                df[col] = ""
            elif col == "penalties":
                df[col] = False
            else:
                df[col] = 0

    # Drop internal helper columns (_home_xg, _away_xg, etc.)
    df = df[_KNOCKOUT_COLS]
    df["predicted_home_goals"] = df["predicted_home_goals"].astype(int)
    df["predicted_away_goals"] = df["predicted_away_goals"].astype(int)
    df["corners"]      = df["corners"].astype(int)
    df["yellow_cards"] = df["yellow_cards"].astype(int)
    df["red_cards"]    = df["red_cards"].astype(int)
    df["match_id"]     = df["match_id"].astype(int)
    df["multiplier"]   = df["multiplier"].astype(int)
    return df


def _notebook_cells(g_path: Path, k_path: Path) -> str:
    return f'''"""
DataCamp FIFA WC 2026 Prediction - Notebook Cells
Paste these cells into your DataCamp notebook to load predictions.
"""

# ── Cell 1: Load group predictions ────────────────────────────────────────────
import pandas as pd

group_predictions = pd.read_csv("{g_path.name}")
print(f"Group predictions: {{len(group_predictions)}} rows")
print(group_predictions.head())

# ── Cell 2: Load knockout predictions ─────────────────────────────────────────
knockout_predictions = pd.read_csv("{k_path.name}")
print(f"Knockout predictions: {{len(knockout_predictions)}} rows")
print(knockout_predictions.head())

# ── Cell 3: Quick validation ───────────────────────────────────────────────────
assert len(group_predictions)   == 72, f"Expected 72 group rows, got {{len(group_predictions)}}"
assert len(knockout_predictions) == 32, f"Expected 32 knockout rows, got {{len(knockout_predictions)}}"
assert group_predictions.isna().sum().sum()    == 0, "Nulls in group predictions"
assert knockout_predictions.isna().sum().sum() == 0, "Nulls in knockout predictions"
print("Validation passed.")

# ── Cell 4: Inspect champion prediction ────────────────────────────────────────
final = knockout_predictions[knockout_predictions["round"] == "Final"].iloc[0]
winner_side = final["match_winner"]
champion = final["predicted_home_team"] if winner_side == "home" else final["predicted_away_team"]
print(f"Predicted champion: {{champion}}")
print(f"Final: {{final['predicted_home_team']}} vs {{final['predicted_away_team']}}")
print(f"Score: {{final['predicted_home_goals']}}-{{final['predicted_away_goals']}}")
print(f"Penalties: {{final['penalties']}}")
'''


def _checklist(g_df: pd.DataFrame, k_df: pd.DataFrame, issues: list[str]) -> str:
    final = k_df[k_df["round"] == "Final"].iloc[0] if not k_df[k_df["round"] == "Final"].empty else None
    winner_side = final["match_winner"] if final is not None else "?"
    champion = (
        final["predicted_home_team"] if winner_side == "home"
        else final["predicted_away_team"]
    ) if final is not None else "?"

    checks = [
        ("Group rows == 72",             len(g_df) == 72),
        ("Knockout rows == 32",          len(k_df) == 32),
        ("No nulls in group",            g_df.isna().sum().sum() == 0),
        ("No nulls in knockout",         k_df.isna().sum().sum() == 0),
        ("Group has winning_team col",   "winning_team" in g_df.columns),
        ("KO has match_winner col",      "match_winner" in k_df.columns),
        ("KO penalties is bool",         k_df["penalties"].dtype == bool),
        ("No helper cols in group",      all(c in _GROUP_COLS for c in g_df.columns)),
        ("No helper cols in knockout",   all(c in _KNOCKOUT_COLS for c in k_df.columns)),
        ("Group goals non-negative",     (g_df["predicted_home_goals"] >= 0).all() and
                                         (g_df["predicted_away_goals"] >= 0).all()),
        ("KO goals non-negative",        (k_df["predicted_home_goals"] >= 0).all() and
                                         (k_df["predicted_away_goals"] >= 0).all()),
        ("Corners > 0",                  (g_df["corners"] > 0).all()),
        ("Yellow cards >= 1",            (g_df["yellow_cards"] >= 1).all()),
        ("Red cards >= 0",               (g_df["red_cards"] >= 0).all()),
    ]

    lines = [
        "DataCamp FIFA WC 2026 Submission Checklist",
        "=" * 45,
        "",
        f"Predicted champion: {champion}",
        "",
        "Pre-submission checks:",
    ]
    all_ok = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        lines.append(f"  [{status}] {label}")

    if issues:
        lines.append("")
        lines.append("Data issues found:")
        lines.extend(issues)

    lines += [
        "",
        f"Overall: {'READY TO SUBMIT' if all_ok and not issues else 'FIX ISSUES BEFORE SUBMITTING'}",
        "",
        "Required group columns:",
        "  " + ", ".join(_GROUP_COLS),
        "",
        "Required knockout columns:",
        "  " + ", ".join(_KNOCKOUT_COLS),
        "",
        "Scoring reminders:",
        "  Group:   40 winning_team | 25 exact score | 10 GD | 10 total goals",
        "           10 corners | 10 yellows | 5 reds",
        "  Knockout (x multiplier):",
        "           20 matchup both | 10 matchup one | 20 winner",
        "           5 penalties | 25 exact score | 10 GD | 10 total goals",
        "           10 corners | 10 yellows | 5 reds",
        "  Multipliers: R32 x1 | R16 x2 | QF x4 | SF x8 | 3P x8 | Final x16",
    ]
    return "\n".join(lines)


def run_export(
    group_df: Optional[pd.DataFrame] = None,
    knockout_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exports DataCamp-ready CSVs and notebook cells.
    Reads from outputs/ if not passed directly.
    """
    if group_df is None:
        group_df = pd.read_csv(OUTPUT_FILES["group_predictions"])
    if knockout_df is None:
        knockout_df = pd.read_csv(OUTPUT_FILES["knockout_predictions"])

    g_export = export_group_predictions(group_df)
    k_export = export_knockout_predictions(knockout_df)

    # Validate
    issues: list[str] = []
    issues.extend(_check_nulls(g_export, "group"))
    issues.extend(_check_nulls(k_export, "knockout"))
    if len(g_export) != 72:
        issues.append(f"  group: expected 72 rows, got {len(g_export)}")
    if len(k_export) != 32:
        issues.append(f"  knockout: expected 32 rows, got {len(k_export)}")

    # Save
    g_path = OUTPUTS_DIR / "datacamp_group_predictions.csv"
    k_path = OUTPUTS_DIR / "datacamp_knockout_predictions.csv"
    nb_path = OUTPUTS_DIR / "datacamp_submission_notebook_cells.py"
    cl_path = OUTPUTS_DIR / "datacamp_submission_checklist.txt"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    g_export.to_csv(g_path, index=False)
    k_export.to_csv(k_path, index=False)
    nb_path.write_text(_notebook_cells(g_path, k_path))
    cl_path.write_text(_checklist(g_export, k_export, issues))

    print(f"[export_datacamp] Group predictions  -> {g_path}  ({len(g_export)} rows)")
    print(f"[export_datacamp] Knockout predictions -> {k_path}  ({len(k_export)} rows)")
    print(f"[export_datacamp] Notebook cells     -> {nb_path}")
    print(f"[export_datacamp] Checklist          -> {cl_path}")

    if issues:
        print("[export_datacamp] ISSUES FOUND:")
        for iss in issues:
            print(iss)
    else:
        print("[export_datacamp] All checks PASSED - submission ready.")

    return g_export, k_export


# Allow typing hint to work when Optional not imported
from typing import Optional  # noqa: E402


if __name__ == "__main__":
    run_export()
