"""
Expected-points diagnostics for DataCamp competition scoring.

Produces outputs/expected_points_diagnostics.csv and
outputs/expected_points_summary.md.

Group scoring:
  40  winning_team correct
  25  exact scoreline
  10  correct goal difference
  10  correct total goals
  10  corners exact
  10  yellow_cards exact
   5  red_cards exact

Knockout scoring (all values multiplied by round multiplier):
  20  matchup correct (both teams)
  10  matchup half-correct (one team)
  20  match_winner correct
   5  penalties correct
  25  exact scoreline
  10  correct GD
  10  correct total goals
  10  corners exact
  10  yellow_cards exact
   5  red_cards exact

Uses Poisson PMF, model win-probabilities, and simulation advancement
percentages where available.  This is an approximation, not exact EV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import OUTPUTS_DIR
from src.goal_model import expected_goals, scoreline_probability_matrix
from src.bracket_difficulty import knockout_win_probability
from src.ratings import get_rating


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_win_probs(
    home_xg: float,
    away_xg: float,
    max_g: int = 8,
) -> tuple[float, float, float]:
    """Returns (p_home_win, p_draw, p_away_win) from Poisson PMF."""
    mat = scoreline_probability_matrix(home_xg, away_xg, max_g)
    p_home = float(np.tril(mat, k=-1).sum())
    p_away = float(np.triu(mat, k=1).sum())
    p_draw = float(np.diag(mat).sum())
    return p_home, p_draw, p_away


def _exact_scoreline_prob(
    home_xg: float,
    away_xg: float,
    hg: int,
    ag: int,
) -> float:
    return float(poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg))


def _goal_diff_prob(
    home_xg: float,
    away_xg: float,
    target_gd: int,
    max_g: int = 8,
) -> float:
    """P(home_goals - away_goals == target_gd)."""
    mat = scoreline_probability_matrix(home_xg, away_xg, max_g)
    total = 0.0
    for h in range(max_g):
        for a in range(max_g):
            if h - a == target_gd:
                total += mat[h, a]
    return float(total)


def _total_goals_prob(
    home_xg: float,
    away_xg: float,
    target_total: int,
    max_g: int = 8,
) -> float:
    """P(home_goals + away_goals == target_total)."""
    mat = scoreline_probability_matrix(home_xg, away_xg, max_g)
    total = 0.0
    for h in range(max_g):
        for a in range(max_g):
            if h + a == target_total:
                total += mat[h, a]
    return float(total)


# ---------------------------------------------------------------------------
# Group match EV
# ---------------------------------------------------------------------------

def group_match_ev(
    row: pd.Series,
    ratings: dict[str, float],
) -> dict:
    home = str(row["home_team"])
    away = str(row["away_team"])
    stage = str(row.get("group", "group"))
    venue = str(row.get("venue", ""))
    hg = int(row["predicted_home_goals"])
    ag = int(row["predicted_away_goals"])
    pred_winner = str(row["winning_team"])
    pred_corners = int(row.get("corners", 9))
    pred_yellows = int(row.get("yellow_cards", 3))
    pred_reds = int(row.get("red_cards", 0))

    home_xg, away_xg = expected_goals(home, away, ratings, stage="group", venue=venue)
    p_home, p_draw, p_away = _poisson_win_probs(home_xg, away_xg)

    # Winner EV
    p_correct_winner = (
        p_home if pred_winner == "home" else
        p_away if pred_winner == "away" else
        p_draw
    )
    winner_ev = 40.0 * p_correct_winner

    # Scoreline EV
    scoreline_p = _exact_scoreline_prob(home_xg, away_xg, hg, ag)
    scoreline_ev = 25.0 * scoreline_p

    # Goal difference EV
    gd = hg - ag
    gd_p = _goal_diff_prob(home_xg, away_xg, gd)
    gd_ev = 10.0 * gd_p

    # Total goals EV
    tg = hg + ag
    tg_p = _total_goals_prob(home_xg, away_xg, tg)
    tg_ev = 10.0 * tg_p

    # Corners EV: use rough empirical Poisson approximation
    corners_p = float(poisson.pmf(pred_corners, 9.0)) * 2.5  # scale to ~15%
    corners_ev = 10.0 * min(corners_p, 0.30)

    # Yellows EV: similar approach
    yellows_p = float(poisson.pmf(pred_yellows, 3.5)) * 2.5
    yellows_ev = 10.0 * min(yellows_p, 0.30)

    # Reds EV: almost always 0; predicting 0 captures ~85%+ probability
    reds_p = 0.85 if pred_reds == 0 else 0.08
    reds_ev = 5.0 * reds_p

    total_ev = winner_ev + scoreline_ev + gd_ev + tg_ev + corners_ev + yellows_ev + reds_ev

    low_confidence = p_correct_winner < 0.40

    return {
        "match_id":        int(row["match_id"]),
        "phase":           "group",
        "multiplier":      1,
        "home_team":       home,
        "away_team":       away,
        "pred_winner":     pred_winner,
        "home_xg":         round(home_xg, 3),
        "away_xg":         round(away_xg, 3),
        "p_home_win":      round(p_home, 3),
        "p_draw":          round(p_draw, 3),
        "p_away_win":      round(p_away, 3),
        "p_pred_winner":   round(p_correct_winner, 3),
        "winner_ev":       round(winner_ev, 2),
        "scoreline_p":     round(scoreline_p, 4),
        "scoreline_ev":    round(scoreline_ev, 2),
        "gd_ev":           round(gd_ev, 2),
        "tg_ev":           round(tg_ev, 2),
        "corners_ev":      round(corners_ev, 2),
        "yellows_ev":      round(yellows_ev, 2),
        "reds_ev":         round(reds_ev, 2),
        "total_ev":        round(total_ev, 2),
        "low_confidence":  low_confidence,
        "flag":            "LOW CONFIDENCE WINNER" if low_confidence else "",
    }


# ---------------------------------------------------------------------------
# Knockout match EV
# ---------------------------------------------------------------------------

def knockout_match_ev(
    row: pd.Series,
    ratings: dict[str, float],
    advancement: Optional[dict[str, float]] = None,
) -> dict:
    home = str(row["predicted_home_team"])
    away = str(row["predicted_away_team"])
    rnd = str(row["round"])
    mult = int(row["multiplier"])
    hg = int(row["predicted_home_goals"])
    ag = int(row["predicted_away_goals"])
    pred_winner = str(row["match_winner"])
    penalties = bool(row["penalties"])
    pred_corners = int(row.get("corners", 9))
    pred_yellows = int(row.get("yellow_cards", 4))
    pred_reds = int(row.get("red_cards", 0))
    slot_h = str(row.get("slot_home", ""))
    slot_a = str(row.get("slot_away", ""))

    home_xg, away_xg = expected_goals(home, away, ratings, stage=rnd)

    # Win probability from logistic (for winner EV)
    home_r = get_rating(home, ratings)
    away_r = get_rating(away, ratings)
    p_home_win = knockout_win_probability(home_r, away_r, rnd)
    p_away_win = 1.0 - p_home_win

    p_correct_winner = p_home_win if pred_winner == "home" else p_away_win
    winner_ev = 20.0 * p_correct_winner

    # Matchup EV: use simulation advancement probabilities if available
    if advancement is not None:
        p_home_adv = advancement.get(home, 0.5) / 100.0
        p_away_adv = advancement.get(away, 0.5) / 100.0
        p_both = min(p_home_adv, p_away_adv)  # conservative: both reach this round
        matchup_ev_both = 20.0 * p_both
        matchup_ev_one = 10.0 * max(p_home_adv, p_away_adv) * (1.0 - min(p_home_adv, p_away_adv))
        matchup_ev = matchup_ev_both + matchup_ev_one
    else:
        matchup_ev = 15.0  # no information: assume partial credit likely

    # Scoreline EV
    scoreline_p = _exact_scoreline_prob(home_xg, away_xg, hg, ag)
    scoreline_ev = 25.0 * scoreline_p

    # GD and total goals EV
    gd = hg - ag
    gd_p = _goal_diff_prob(home_xg, away_xg, gd)
    gd_ev = 10.0 * gd_p
    tg = hg + ag
    tg_p = _total_goals_prob(home_xg, away_xg, tg)
    tg_ev = 10.0 * tg_p

    # Corners and yellows
    corners_p = float(poisson.pmf(pred_corners, 9.5)) * 2.5
    corners_ev = 10.0 * min(corners_p, 0.30)
    yellows_p = float(poisson.pmf(pred_yellows, 4.0)) * 2.5
    yellows_ev = 10.0 * min(yellows_p, 0.30)

    reds_p = 0.85 if pred_reds == 0 else 0.08
    reds_ev = 5.0 * reds_p

    # Penalty EV: P(our penalty call correct)
    # In a draw (det. prediction), we always predict penalties=True; ~50% of actual KO draws go to pens
    # In a decisive (det. prediction), penalties=False; ~80% of non-draw KOs have no pens
    penalties_p = 0.50 if penalties else 0.80
    penalty_ev = 5.0 * penalties_p

    # Winner vs prob consistency check
    flag = ""
    if (pred_winner == "home" and p_home_win < 0.45) or (pred_winner == "away" and p_away_win < 0.45):
        flag = "WINNER CONTRADICTS PROBABILITY"
    if p_correct_winner < 0.52:
        flag = flag + ("; " if flag else "") + "COIN-FLIP WINNER"

    base_ev = matchup_ev + winner_ev + scoreline_ev + gd_ev + tg_ev + corners_ev + yellows_ev + reds_ev + penalty_ev
    total_ev = base_ev * mult

    return {
        "match_id":            int(row["match_id"]),
        "phase":               "knockout",
        "multiplier":          mult,
        "home_team":           home,
        "away_team":           away,
        "round":               rnd,
        "pred_winner":         pred_winner,
        "penalties_predicted": penalties,
        "home_xg":             round(home_xg, 3),
        "away_xg":             round(away_xg, 3),
        "p_home_logistic":     round(p_home_win, 3),
        "p_away_logistic":     round(p_away_win, 3),
        "p_pred_winner":       round(p_correct_winner, 3),
        "matchup_ev":          round(matchup_ev, 2),
        "winner_ev":           round(winner_ev, 2),
        "scoreline_ev":        round(scoreline_ev, 2),
        "gd_ev":               round(gd_ev, 2),
        "tg_ev":               round(tg_ev, 2),
        "corners_ev":          round(corners_ev, 2),
        "yellows_ev":          round(yellows_ev, 2),
        "reds_ev":             round(reds_ev, 2),
        "penalty_ev":          round(penalty_ev, 2),
        "base_ev":             round(base_ev, 2),
        "total_ev_with_mult":  round(total_ev, 2),
        "flag":                flag,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_expected_points_diagnostics(
    group_df: pd.DataFrame,
    knockout_df: pd.DataFrame,
    ratings: dict[str, float],
    advancement_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Runs EV diagnostics for all matches.
    Returns (diagnostics_df, summary_lines).
    """
    # Build advancement lookup: team -> advancement% at each round
    # Use the advancement probabilities at the relevant KO round
    adv_lookup: dict[str, dict[str, float]] = {}
    if advancement_df is not None:
        round_cols = {
            "Round of 32":         "r32_pct",
            "Round of 16":         "r16_pct",
            "Quarter-final":       "qf_pct",
            "Semi-final":          "sf_pct",
            "Third-place playoff": "sf_pct",
            "Final":               "final_pct",
        }
        for _, r in advancement_df.iterrows():
            team = str(r["team"])
            adv_lookup[team] = {
                col: float(r[sim_col]) if sim_col in r.index else 50.0
                for col, sim_col in round_cols.items()
            }

    rows = []

    # Group matches
    for _, row in group_df.iterrows():
        rows.append(group_match_ev(row, ratings))

    # Knockout matches
    for _, row in knockout_df.iterrows():
        rnd = str(row["round"])
        adv_for_round: Optional[dict[str, float]] = None
        if adv_lookup:
            home = str(row["predicted_home_team"])
            away = str(row["predicted_away_team"])
            col = {
                "Round of 32":         "r32_pct",
                "Round of 16":         "r16_pct",
                "Quarter-final":       "qf_pct",
                "Semi-final":          "sf_pct",
                "Third-place playoff": "sf_pct",
                "Final":               "final_pct",
            }.get(rnd, "r32_pct")
            adv_for_round = {
                home: adv_lookup.get(home, {}).get(col, 50.0),
                away: adv_lookup.get(away, {}).get(col, 50.0),
            }
        rows.append(knockout_match_ev(row, ratings, adv_for_round))

    df = pd.DataFrame(rows)

    # Save CSV
    out_csv = OUTPUTS_DIR / "expected_points_diagnostics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[expected_points] Diagnostics saved -> {out_csv}")

    # Build summary
    group_rows = df[df["phase"] == "group"]
    ko_rows = df[df["phase"] == "knockout"]
    flags = df[df["flag"].str.len() > 0]

    total_group_ev = float(group_rows["total_ev"].sum())
    total_ko_ev = float(ko_rows["total_ev_with_mult"].sum())
    total_ev = total_group_ev + total_ko_ev

    lines = [
        "# Expected-Points Diagnostics",
        "",
        "## Summary",
        f"| Stage       | Matches | Est. Total EV |",
        f"|-------------|---------|---------------|",
        f"| Group stage | {len(group_rows):>7} | {total_group_ev:>13.1f} |",
        f"| Knockout    | {len(ko_rows):>7} | {total_ko_ev:>13.1f} |",
        f"| **Total**   | {len(df):>7} | **{total_ev:.1f}** |",
        "",
        "## Flagged Matches",
        "",
    ]

    if flags.empty:
        lines.append("No flagged matches.")
    else:
        lines.append("| Match | Phase | Round/Group | Home | Away | Flag |")
        lines.append("|-------|-------|-------------|------|------|------|")
        for _, r in flags.iterrows():
            rnd = r.get("round", r.get("phase", ""))
            lines.append(
                f"| {int(r['match_id'])} | {r['phase']} | {rnd} "
                f"| {r['home_team']} | {r['away_team']} | {r['flag']} |"
            )

    lines += [
        "",
        "## High-EV Knockout Matches (by total_ev_with_mult)",
        "",
    ]
    if not ko_rows.empty:
        top_ko = ko_rows.nlargest(10, "total_ev_with_mult")[
            ["match_id", "round", "multiplier", "home_team", "away_team",
             "p_pred_winner", "base_ev", "total_ev_with_mult", "flag"]
        ]
        lines.append(top_ko.to_string(index=False))

    lines += [
        "",
        "## Low-Confidence Group Picks",
        "",
    ]
    low_conf = group_rows[group_rows["low_confidence"]]
    if low_conf.empty:
        lines.append("None (all group winner picks have P >= 0.40).")
    else:
        lines.append(low_conf[["match_id", "home_team", "away_team",
                                "pred_winner", "p_pred_winner", "total_ev"]].to_string(index=False))

    lines += [
        "",
        "## Notes",
        "- EV approximations use Poisson PMF for scoreline probability.",
        "- Corners/yellows EV uses Poisson(mu=9.0/3.5) scaled to empirical hit rate ~15%.",
        "- Knockout matchup EV uses simulation advancement percentages.",
        "- `COIN-FLIP WINNER` = predicted winner has P < 0.52.",
        "- `WINNER CONTRADICTS PROBABILITY` = predicted winner has P < 0.45.",
    ]

    out_md = OUTPUTS_DIR / "expected_points_summary.md"
    out_md.write_text("\n".join(lines))
    print(f"[expected_points] Summary written -> {out_md}")

    return df, lines
