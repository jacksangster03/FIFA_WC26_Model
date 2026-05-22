# External Data

Place optional external datasets here. The pipeline works without them
(it uses seeded ratings as fallback). With these files it blends the
data for more accurate predictions.

## Supported files

| File | Description | Key columns |
|------|-------------|-------------|
| `international_results.csv` | Historical international match results | team, opponent, score, date, tournament |
| `fifa_rankings.csv` | FIFA World Ranking points | team/country, points/total_points |
| `world_football_elo.csv` | World Football Elo ratings | team/country, elo/rating |
| `world_cup_matches.csv` | All historical World Cup matches | home_team, away_team, home_score, away_score, year |
| `squad_values.csv` | Transfermarkt squad market values (EUR) | team/country, value_eur/squad_value |
| `cards_corners_history.csv` | Historical cards and corners by team | team, avg_corners, avg_yellows, avg_reds |

## Sources

- **international_results.csv**: [martj42/international-football-results](https://github.com/martj42/international-football-results)
- **fifa_rankings.csv**: [FIFA Ranking](https://www.fifa.com/fifa-world-ranking)
- **world_football_elo.csv**: [eloratings.net](https://www.eloratings.net/)
- **squad_values.csv**: [Transfermarkt](https://www.transfermarkt.com/)

## Column name flexibility

The pipeline tries several common column name variants so that data from
different sources works without renaming. If loading fails, check
`src/external_data.py` for the exact names it looks for.
