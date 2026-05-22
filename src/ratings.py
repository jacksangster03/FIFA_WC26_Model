"""
Team strength rating system.

Source-driven blend (target weights):
  FIFA points (normalised)    35%
  World Elo (normalised)      25%
  Recency-weighted form       15%
  Tournament pedigree         10%
  Seeded/manual fallback      10%
  Host advantage               5%

When a source is unavailable for a team, its weight is redistributed
proportionally among available sources (excluding host, which is binary).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import HOST_COUNTRIES, HOST_BOOST, PROCESSED_DIR

# ---------------------------------------------------------------------------
# Seeded ratings (0-100 scale).  Used as the fallback regularisation component.
# ---------------------------------------------------------------------------
SEEDED_RATINGS: dict[str, float] = {
    # World elite
    "France":      92.0,
    "Brazil":      91.0,
    "Argentina":   90.0,
    "England":     88.0,
    "Spain":       87.0,

    # Strong contenders
    "Germany":     85.0,
    "Portugal":    84.0,
    "Netherlands": 83.0,
    "Belgium":     81.0,
    "Uruguay":     80.0,

    # Solid internationals
    "Colombia":    78.0,
    "Morocco":     76.0,
    "Mexico":      75.0,
    "Japan":       75.0,
    "USA":         73.0,
    "Senegal":     73.0,
    "Croatia":     74.0,
    "Austria":     73.0,
    "South Korea": 72.0,
    "Canada":      72.0,
    "Norway":      71.0,
    "Ecuador":     71.0,
    "Switzerland": 70.0,

    # Competitive qualifier-level
    "Sweden":       69.0,
    "Turkey":       68.0,
    "Scotland":     68.0,
    "Algeria":      66.0,
    "Australia":    64.0,
    "Saudi Arabia": 64.0,
    "Czechia":      63.5,
    "Iran":         63.0,
    "Bosnia and Herzegovina": 62.0,
    "Paraguay":     62.0,
    "Tunisia":      62.0,
    "Egypt":        61.0,
    "Ghana":        61.0,
    "Uzbekistan":   60.0,
    "Iraq":         59.0,

    # Weaker qualifiers
    "South Africa": 58.0,
    "Panama":       57.0,
    "Jordan":       55.0,
    "DR Congo":     54.0,
    "Cabo Verde":   52.0,
    "New Zealand":  51.0,
    "Qatar":        50.0,

    # Underdogs
    "Haiti":        44.0,
    "Curaçao":      40.0,

}

# Target weights
_W = {
    "fifa":     0.35,
    "elo":      0.25,
    "form":     0.15,
    "pedigree": 0.10,
    "seeded":   0.10,
    "host":     0.05,
}


def _load_csv_col(path, team_col: str, val_col: str) -> dict[str, float]:
    """Generic: load a processed CSV and return {team: value}."""
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if team_col in df.columns and val_col in df.columns:
            return dict(zip(df[team_col].astype(str), df[val_col].astype(float)))
    except Exception:
        pass
    return {}


def _blend(components: dict[str, Optional[float]],
           weights:    dict[str, float],
           seeded:     float,
           ) -> tuple[float, dict[str, float], list[str]]:
    """
    Weighted blend with proportional reweighting for missing sources.

    components: {source_name: value_or_None}  (host component excluded here)
    weights:    target weight per source
    seeded:     fallback regularisation value

    Returns (blended_score, used_components_dict, sources_used_list).
    """
    # Sources that contributed (have a value)
    available = {k: v for k, v in components.items() if v is not None}
    missing   = [k for k, v in components.items() if v is None]

    if not available:
        return seeded, {"seeded": seeded}, ["seeded_fallback"]

    # Redistribute missing weights proportionally among available sources
    missing_weight = sum(weights[k] for k in missing)
    avail_total    = sum(weights[k] for k in available)

    final_weights = {}
    for k in available:
        extra = weights[k] / avail_total * missing_weight if avail_total > 0 else 0.0
        final_weights[k] = weights[k] + extra

    blended = sum(final_weights[k] * available[k] for k in available)
    return blended, {k: round(available[k], 3) for k in available}, list(available.keys())


def compute_team_ratings(
    external_data: Optional[dict] = None,
) -> dict[str, float]:
    """
    Returns {team: composite_rating} using the source-driven blend.
    Called by main.py only when form_processor has not produced ratings yet.
    Normally form_processor.run_form_processing() does the full blend.
    """
    ratings = _load_form_adjusted_ratings()
    if ratings:
        merged = {t: ratings.get(t, v) for t, v in SEEDED_RATINGS.items()}
        merged.update({t: v for t, v in ratings.items() if t not in merged})
        print(f"[ratings] Using form-adjusted ratings from processed/ ({len(ratings)} teams)")
        return merged
    # No processed file: pure seeded + host boost
    print("[ratings] No processed ratings found; using seeded fallback")
    return {t: v + (HOST_BOOST if t in HOST_COUNTRIES else 0.0)
            for t, v in SEEDED_RATINGS.items()}


def blend_all_sources(
    teams:     list[str],
    form_lookup: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Full source-driven blend for all teams.
    Returns (ratings_dict, details_df).

    details_df has one row per team with all component columns.
    """
    from src.data_sources import fifa_rankings, world_elo, tournament_achievements

    fifa_scores     = fifa_rankings.load_normalised()
    elo_scores      = world_elo.load_normalised()
    pedigree_scores = tournament_achievements.load_normalised()

    rows = []
    ratings: dict[str, float] = {}

    for team in teams:
        seeded  = SEEDED_RATINGS.get(team, 55.0)
        is_host = team in HOST_COUNTRIES

        components: dict[str, Optional[float]] = {
            "fifa":     fifa_scores.get(team),
            "elo":      elo_scores.get(team),
            "form":     form_lookup.get(team),
            "pedigree": pedigree_scores.get(team),
            "seeded":   seeded,
        }

        blend_weights = {k: _W[k] for k in ["fifa", "elo", "form", "pedigree", "seeded"]}

        base_score, used_vals, sources_used = _blend(components, blend_weights, seeded)

        # Host component: additive flat boost, scaled by its weight
        host_val = (HOST_BOOST / 100.0 * 100.0) if is_host else 0.0

        # Final = base weighted on 0.95 of total weight + host on 0.05
        final = float(np.clip(base_score * 0.95 + host_val * _W["host"], 0.0, 100.0))

        # More intuitive: just add the host boost directly (same as before)
        final = float(np.clip(base_score + (HOST_BOOST if is_host else 0.0), 0.0, 100.0))

        ratings[team] = final

        explanation_parts = []
        if "fifa" in sources_used:
            explanation_parts.append(f"FIFA={used_vals['fifa']:.1f}")
        if "elo" in sources_used:
            explanation_parts.append(f"Elo={used_vals['elo']:.1f}")
        if "form" in sources_used:
            explanation_parts.append(f"form={used_vals['form']:.1f}")
        if "pedigree" in sources_used:
            explanation_parts.append(f"pedigree={used_vals['pedigree']:.1f}")
        if is_host:
            explanation_parts.append(f"host+{HOST_BOOST}")

        rows.append({
            "team":                        team,
            "final_rating":                round(final, 2),
            "fifa_component":              used_vals.get("fifa"),
            "elo_component":               used_vals.get("elo"),
            "form_component":              used_vals.get("form"),
            "tournament_pedigree_component": used_vals.get("pedigree"),
            "seeded_component":            seeded,
            "host_component":              HOST_BOOST if is_host else 0.0,
            "sources_used":                ",".join(sources_used),
            "rating_explanation":          " | ".join(explanation_parts),
        })

    details_df = pd.DataFrame(rows).sort_values("final_rating", ascending=False).reset_index(drop=True)
    return ratings, details_df


def _load_form_adjusted_ratings() -> Optional[dict[str, float]]:
    path = PROCESSED_DIR / "team_ratings_final.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "team" in df.columns and "final_rating" in df.columns:
            return dict(zip(df["team"], df["final_rating"].astype(float)))
    except Exception:
        pass
    return None


def get_rating(team: str, ratings: dict[str, float]) -> float:
    if team in ratings:
        return ratings[team]
    return 55.0
