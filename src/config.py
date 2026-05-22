"""
Central configuration for the FIFA WC 2026 prediction model.
Edit this file to tune weights, paths, and model behaviour.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
SRC_DIR = ROOT / "src"

GROUP_FIXTURES_PATH = DATA_DIR / "group_fixtures.csv"
KNOCKOUT_SLOTS_PATH = DATA_DIR / "knockout_slots.csv"
MANUAL_OVERRIDES_PATH = DATA_DIR / "manual_overrides.csv"

EXTERNAL_FILES = {
    "international_results": EXTERNAL_DIR / "international_results.csv",
    "fifa_rankings":         EXTERNAL_DIR / "fifa_rankings.csv",
    "world_football_elo":    EXTERNAL_DIR / "world_football_elo.csv",
    "world_cup_matches":     EXTERNAL_DIR / "world_cup_matches.csv",
    "squad_values":          EXTERNAL_DIR / "squad_values.csv",
    "cards_corners_history": EXTERNAL_DIR / "cards_corners_history.csv",
}

OUTPUT_FILES = {
    "group_predictions":    OUTPUTS_DIR / "group_predictions.csv",
    "knockout_predictions": OUTPUTS_DIR / "knockout_predictions.csv",
    "group_tables":         OUTPUTS_DIR / "group_tables.csv",
    "bracket_summary":      OUTPUTS_DIR / "bracket_summary.csv",
    "model_report":         OUTPUTS_DIR / "model_report.md",
    "quality_checks":       OUTPUTS_DIR / "submission_quality_checks.txt",
    "simulation_summary":   OUTPUTS_DIR / "simulation_summary.csv",
}

# ---------------------------------------------------------------------------
# Rating component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
RATING_WEIGHTS = {
    "elo":               0.40,
    "fifa_points":       0.25,
    "recent_form":       0.15,
    "squad_value":       0.10,
    "host":              0.05,
    "tournament_pedigree": 0.05,
}

# ---------------------------------------------------------------------------
# Host countries and boost
# ---------------------------------------------------------------------------
HOST_COUNTRIES = {"USA", "Mexico", "Canada"}
HOST_BOOST = 4.0          # rating points added to host countries

# ---------------------------------------------------------------------------
# Goal model parameters
# ---------------------------------------------------------------------------
BASE_XG = 1.35            # expected goals per team at equal strength, neutral
HOME_XG_BONUS = 0.10      # for the designated home team at neutral venues
ACTUAL_HOME_BONUS = 0.22  # for host-country matches played in their own country
STRENGTH_SCALE = 25.0     # rating points per xG unit adjustment
STRENGTH_XG_FACTOR = 0.42 # xG shift per STRENGTH_SCALE rating points

MIN_XG = 0.30
MAX_XG = 3.50

# Rating-point gap that separates "decisive" matches from draws in
# deterministic mode.  Teams within this many points predict a draw;
# beyond it, the higher-rated team is expected to win.
DRAW_RATING_THRESHOLD_GROUP = 12.0
DRAW_RATING_THRESHOLD_KNOCKOUT = 8.0

# Deterministic scoreline formula biases.
# WINNER_BIAS is added to the winner's xG before rounding to goals,
# pushing moderate-strength favourites from 1 goal to 2.
# LOSER_XG_THRESHOLD: if the loser's xG falls below this value the
# loser is predicted to score 0 (strongly dominated sides).
WINNER_BIAS          = 0.40
LOSER_XG_THRESHOLD   = 0.80

# Poisson scoreline grid
MAX_GOALS_GRID = 7        # 0..6 goals per team in probability matrix

# World Cup conservatism multiplier (applied on top of all stage factors)
WC_CONSERVATISM = 0.92

# Knockout stage xG multipliers (relative to group stage)
KNOCKOUT_XG_FACTORS = {
    "group":              1.00,
    "Round of 32":        0.90,
    "Round of 16":        0.88,
    "Quarter-final":      0.85,
    "Semi-final":         0.83,
    "Third-place playoff": 0.88,
    "Final":              0.82,
}

# ---------------------------------------------------------------------------
# Penalty shoot-out rules
# ---------------------------------------------------------------------------
# Minimum rating advantage to deterministically win on penalties.
# Below this threshold the higher-rated team is still favoured but close.
PENALTY_DETERMINISTIC_THRESHOLD = 5.0

# Knockout win-probability scale by stage (used by diagnostics and
# deterministic penalty tie-breaks).
KNOCKOUT_PROBABILITY_SCALES = {
    "Round of 32":         15.5,
    "Round of 16":         16.5,
    "Quarter-final":       18.0,
    "Semi-final":          19.0,
    "Third-place playoff": 18.0,
    "Final":               20.0,
}

# ---------------------------------------------------------------------------
# Cards & corners model parameters
# ---------------------------------------------------------------------------
BASE_CORNERS = 9.5
BASE_YELLOW  = 3.5
BASE_RED     = 0.0

KNOCKOUT_YELLOW_BOOST = 0.8
KNOCKOUT_RED_BOOST    = 0.05   # very small

CLOSE_GAME_YELLOW_BOOST  = 0.6  # added when |xg_diff| < 0.3
MISMATCH_CORNER_PENALTY  = -1.0 # mismatch games often have fewer corners

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
DEFAULT_SIM_N = 1_000     # default number of simulated tournaments
MONTE_CARLO_N = 1_000     # kept for backwards compatibility

# ---------------------------------------------------------------------------
# Validation expectations
# ---------------------------------------------------------------------------
EXPECTED_GROUP_ROWS    = 72
EXPECTED_KNOCKOUT_ROWS = 32
VALID_WINNER_VALUES    = {"home", "away", "draw"}
VALID_KO_WINNER_VALUES = {"home", "away"}
