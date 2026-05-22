"""
Refresh all external data sources.

Run via:
    python -m src.main --refresh-data

Or call programmatically:
    from src.data_sources.refresh_all import run_all
    run_all(force_refresh=True)
"""

from __future__ import annotations

from src.data_sources import fifa_rankings, world_elo, tournament_achievements, playoff_resolver


def run_all(force_refresh: bool = True) -> None:
    """Refresh all processed data files."""
    print("\n[refresh_all] Refreshing all data sources...")

    print("\n[refresh_all] 1/4  FIFA rankings")
    fifa_rankings.run(force_refresh=force_refresh)

    print("\n[refresh_all] 2/4  World Elo ratings")
    world_elo.run(force_refresh=force_refresh)

    print("\n[refresh_all] 3/4  Tournament pedigree")
    tournament_achievements.run(force_refresh=force_refresh)

    print("\n[refresh_all] 4/4  Playoff resolution")
    playoff_resolver.run(force_refresh=force_refresh)

    print("\n[refresh_all] All sources refreshed.\n")
