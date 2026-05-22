"""
Match feature engineering.

Currently produces a lightweight feature dict from team ratings.
Extend this module when external historical data becomes available.
"""

from __future__ import annotations

from src.config import HOST_COUNTRIES
from src.ratings import get_rating


def build_match_features(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    stage: str = "group",
    venue: str = "",
) -> dict:
    """
    Returns a dict of features for a single match.
    """
    home_rating = get_rating(home_team, ratings)
    away_rating = get_rating(away_team, ratings)
    diff = home_rating - away_rating

    # Detect whether the home team is actually playing in their home country.
    venue_lower = venue.lower()
    home_is_host = home_team in HOST_COUNTRIES and any(
        city in venue_lower
        for city in _host_cities(home_team)
    )

    return {
        "home_team":      home_team,
        "away_team":      away_team,
        "home_rating":    home_rating,
        "away_rating":    away_rating,
        "rating_diff":    diff,
        "abs_diff":       abs(diff),
        "stage":          stage,
        "home_is_host":   home_is_host,
        "is_knockout":    stage != "group",
        "is_final":       stage in ("Final", "Semi-final"),
    }


def _host_cities(team: str) -> list[str]:
    cities = {
        "USA": [
            "sofi stadium", "at&t stadium", "metlife", "hard rock",
            "levi's", "lumen field", "nrg stadium", "gillette",
            "lincoln financial", "geha field", "arrowhead", "mercedes-benz",
        ],
        "Mexico": ["estadio azteca", "estadio akron", "estadio bbva"],
        "Canada": ["bmo field", "bc place"],
    }
    return cities.get(team, [])
