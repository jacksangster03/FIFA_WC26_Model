"""
Manual override layer.

Loads data/manual_overrides.csv (if it exists) and applies any overrides
to the group or knockout predictions DataFrames after model prediction
and before quality checks.

Override columns (all optional except match_id and phase):
  match_id, phase, predicted_home_team, predicted_away_team,
  predicted_home_goals, predicted_away_goals, corners, yellow_cards,
  red_cards, winning_team, match_winner, penalties, reason
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.config import MANUAL_OVERRIDES_PATH

_NUMERIC_COLS = {
    "predicted_home_goals", "predicted_away_goals",
    "corners", "yellow_cards", "red_cards",
}

_BOOL_COLS = {"penalties"}


def load_overrides() -> pd.DataFrame:
    """Returns the overrides DataFrame, or empty DataFrame if not found."""
    if not MANUAL_OVERRIDES_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(MANUAL_OVERRIDES_PATH)
        if df.empty:
            return df
        df["match_id"] = df["match_id"].astype(int)
        print(f"[overrides] Loaded {len(df)} override(s) from {MANUAL_OVERRIDES_PATH}")
        return df
    except Exception as exc:
        warnings.warn(f"[overrides] Failed to load overrides: {exc}")
        return pd.DataFrame()


def apply_overrides(
    predictions: pd.DataFrame,
    overrides: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:
    """
    Applies rows from `overrides` where phase matches to `predictions`.
    Returns modified predictions.
    """
    if overrides.empty:
        return predictions

    phase_overrides = overrides[overrides["phase"] == phase]
    if phase_overrides.empty:
        return predictions

    df = predictions.copy()

    for _, override in phase_overrides.iterrows():
        mid = override["match_id"]
        mask = df["match_id"] == mid

        if not mask.any():
            warnings.warn(
                f"[overrides] match_id {mid} not found in {phase} predictions"
            )
            continue

        reason = override.get("reason", "")
        applied = []

        for col in df.columns:
            if col in override.index and pd.notna(override[col]):
                val = override[col]
                if col in _NUMERIC_COLS:
                    val = int(val)
                elif col in _BOOL_COLS:
                    val = bool(val)
                df.loc[mask, col] = val
                applied.append(col)

        print(
            f"[overrides] Applied to match {mid} ({phase}): "
            f"{applied}"
            + (f" | reason: {reason}" if reason else "")
        )

    return df
