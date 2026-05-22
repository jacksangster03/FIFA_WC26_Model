"""
data_sources — ingestion modules that produce processed CSVs.

Each module exposes a run() function that writes to data/processed/.
Run all at once via refresh_all.run_all() or via:
    python -m src.main --refresh-data
"""
