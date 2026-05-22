"""
Model report generator.

Writes outputs/model_report.md with:
  - Pipeline summary
  - Team ratings used
  - Group tables
  - Bracket summary
  - Scoring opportunities breakdown
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import OUTPUT_FILES, RATING_WEIGHTS, HOST_BOOST, HOST_COUNTRIES


def generate_model_report(
    group_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
    group_tables: pd.DataFrame,
    ratings: dict[str, float],
    has_external_data: bool,
) -> None:
    out_path = OUTPUT_FILES["model_report"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def h1(title: str) -> None:
        lines.append(f"# {title}\n\n")

    def h2(title: str) -> None:
        lines.append(f"## {title}\n\n")

    def h3(title: str) -> None:
        lines.append(f"### {title}\n\n")

    def para(text: str) -> None:
        lines.append(text + "\n\n")

    h1("FIFA World Cup 2026 — Prediction Model Report")
    para(f"Generated: {date.today().isoformat()}")

    # ---
    h2("Pipeline Overview")
    para(
        "This model uses a hybrid approach combining:\n"
        "- Seeded team strength ratings calibrated to current FIFA/Elo data\n"
        "- Poisson expected-goals model with World Cup conservatism\n"
        "- Stage-specific xG multipliers (group, R32, R16, QF, SF, Final)\n"
        "- Host advantage for USA, Mexico, Canada\n"
        "- Tournament pedigree weighting\n"
        "- Deterministic penalty shoot-out resolution\n"
        "- Side models for corners and cards\n"
        "- Manual override layer for late-breaking intelligence"
    )

    data_status = "External data loaded and blended" if has_external_data else "Seeded fallback ratings only"
    para(f"**Data status:** {data_status}")

    # ---
    h2("Rating Weights")
    for component, weight in RATING_WEIGHTS.items():
        lines.append(f"- `{component}`: {weight:.0%}\n")
    lines.append(f"\nHost boost: +{HOST_BOOST} rating points for {', '.join(sorted(HOST_COUNTRIES))}\n\n")

    # ---
    h2("Team Ratings Used")
    rated = sorted(ratings.items(), key=lambda x: -x[1])
    lines.append("| Rank | Team | Rating |\n")
    lines.append("|------|------|--------|\n")
    for i, (team, rating) in enumerate(rated, 1):
        lines.append(f"| {i} | {team} | {rating:.1f} |\n")
    lines.append("\n")

    # ---
    h2("Group Stage Results")
    for grp in sorted(group_tables["group"].unique()):
        grp_df = group_tables[group_tables["group"] == grp].sort_values("position")
        h3(f"Group {grp}")
        lines.append("| Pos | Team | Pts | W | D | L | GF | GA | GD |\n")
        lines.append("|-----|------|-----|---|---|---|----|----|----|\n")
        for _, row in grp_df.iterrows():
            lines.append(
                f"| {int(row['position'])} | {row['team']} | {int(row['pts'])} | "
                f"{int(row['wins'])} | {int(row['draws'])} | {int(row['losses'])} | "
                f"{int(row['gf'])} | {int(row['ga'])} | {int(row['gd'])} |\n"
            )
        lines.append("\n")

    # ---
    h2("Knockout Bracket")
    for rnd in ["Round of 32", "Round of 16", "Quarter-final", "Semi-final",
                "Third-place playoff", "Final"]:
        rnd_df = knockout_predictions[knockout_predictions["round"] == rnd]
        if rnd_df.empty:
            continue
        h3(rnd)
        for _, row in rnd_df.iterrows():
            hg, ag = row["predicted_home_goals"], row["predicted_away_goals"]
            winner = (
                row["predicted_home_team"] if row["match_winner"] == "home"
                else row["predicted_away_team"]
            )
            pen_str = " (pen)" if row["penalties"] else ""
            lines.append(
                f"- Match {row['match_id']}: **{row['predicted_home_team']}** {hg}–{ag} "
                f"**{row['predicted_away_team']}** → Winner: **{winner}**{pen_str}\n"
            )
        lines.append("\n")

    # ---
    h2("Scoring Opportunity Summary")
    total_matches = len(group_predictions) + len(knockout_predictions)
    ko = knockout_predictions
    penalties_count = ko["penalties"].sum() if "penalties" in ko.columns else 0
    para(
        f"- Total matches predicted: {total_matches}\n"
        f"- Group stage matches: {len(group_predictions)}\n"
        f"- Knockout matches: {len(knockout_predictions)}\n"
        f"- Knockout matches going to penalties: {int(penalties_count)}\n"
        f"- Draws in group stage: {(group_predictions['winning_team'] == 'draw').sum()}"
    )

    out_path.write_text("".join(lines))
    print(f"[report] Model report written -> {out_path}")
