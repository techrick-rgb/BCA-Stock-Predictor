"""
Configuration settings for the BBCA Stock Predictor pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

RAW_DATA_PATH = DATA_DIR / "bbca_raw.csv"
MODEL_PATH = MODELS_DIR / "random_forest_bbca_model.pkl"
MODEL_CHECKSUM_PATH = MODELS_DIR / "model_checksum.sha256"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Stock parameters
TICKER = "BBCA.JK"
DEFAULT_START_DATE = "2015-01-01"

# Feature definitions
FEATURE_COLS = [
    "MA7",
    "MA30",
    "MA90",
    "Volatility30",
    "RSI14",
    "Close_lag1",
    "Close_lag2",
    "Close_lag3",
    "Close_lag5",
]

# Model validation split
TRAIN_TEST_SPLIT_RATIO = 0.85
RANDOM_STATE = 42
