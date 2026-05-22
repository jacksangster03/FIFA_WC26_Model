"""
Playoff resolver.

Reads data/external/playoff_results.csv and substitutes placeholder
team names in group_fixtures.csv with resolved team names, producing:

  data/processed/playoff_mapping.csv
  data/processed/group_fixtures_resolved.csv

Also adds resolved teams to the SEEDED_RATINGS fallback if they are
absent (inheriting from the placeholder's seed).
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, EXTERNAL_DIR, PROCESSED_DIR

PLAYOFF_RESULTS_PATH  = EXTERNAL_DIR / "playoff_results.csv"
PLAYOFF_MAPPING_OUT   = PROCESSED_DIR / "playoff_mapping.csv"
RESOLVED_FIXTURES_OUT = PROCESSED_DIR / "group_fixtures_resolved.csv"

# Columns required in playoff_results.csv
_REQUIRED_COLS = {"placeholder", "replacement_team", "canonical_team"}


def load_playoff_mapping() -> pd.DataFrame:
    """
    Loads playoff_results.csv.  Returns empty DataFrame if missing.
    """
    if not PLAYOFF_RESULTS_PATH.exists():
        print(f"[playoff_resolver] {PLAYOFF_RESULTS_PATH} not found; no resolutions applied")
        return pd.DataFrame(columns=["placeholder", "replacement_team", "canonical_team",
                                     "group", "source", "status", "confidence", "notes"])
    df = pd.read_csv(PLAYOFF_RESULTS_PATH)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"[playoff_resolver] WARN: playoff_results.csv missing columns {missing}")
        return pd.DataFrame()
    resolved = df[df["status"] == "resolved"]
    print(f"[playoff_resolver] {len(resolved)}/{len(df)} playoff slots resolved")
    return resolved.copy()


def run(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces playoff_mapping.csv and group_fixtures_resolved.csv.
    Returns (playoff_mapping_df, resolved_fixtures_df).
    """
    if (PLAYOFF_MAPPING_OUT.exists() and RESOLVED_FIXTURES_OUT.exists()
            and not force_refresh):
        try:
            pm = pd.read_csv(PLAYOFF_MAPPING_OUT)
            rf = pd.read_csv(RESOLVED_FIXTURES_OUT)
            print(f"[playoff_resolver] Loaded cached resolved fixtures ({len(rf)} rows)")
            return pm, rf
        except Exception:
            pass

    mapping = load_playoff_mapping()

    # Build substitution dict: placeholder -> canonical_team
    sub: dict[str, str] = {}
    if not mapping.empty:
        for _, row in mapping.iterrows():
            sub[row["placeholder"]] = row["canonical_team"]

    # Load raw fixtures
    raw_path = DATA_DIR / "group_fixtures.csv"
    fixtures = pd.read_csv(raw_path)

    resolved = fixtures.copy()
    resolved["home_team"] = resolved["home_team"].map(lambda t: sub.get(t, t))
    resolved["away_team"] = resolved["away_team"].map(lambda t: sub.get(t, t))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not mapping.empty:
        mapping.to_csv(PLAYOFF_MAPPING_OUT, index=False)
    else:
        pd.DataFrame(columns=["placeholder", "replacement_team", "canonical_team",
                               "group", "source", "status", "confidence", "notes"]
                     ).to_csv(PLAYOFF_MAPPING_OUT, index=False)

    resolved.to_csv(RESOLVED_FIXTURES_OUT, index=False)

    applied = sum(1 for t in list(fixtures["home_team"]) + list(fixtures["away_team"])
                  if t in sub)
    print(f"[playoff_resolver] Resolved {len(sub)} placeholders "
          f"({applied} team slots substituted) -> {RESOLVED_FIXTURES_OUT}")
    return mapping, resolved


def get_resolved_teams() -> dict[str, str]:
    """Returns {placeholder: canonical_team} from cached mapping, or runs if needed."""
    if not PLAYOFF_MAPPING_OUT.exists():
        run()
    try:
        df = pd.read_csv(PLAYOFF_MAPPING_OUT)
        if df.empty or "placeholder" not in df.columns:
            return {}
        return dict(zip(df["placeholder"], df["canonical_team"]))
    except Exception:
        return {}
