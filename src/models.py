"""
Machine learning training, model serialization, evaluation, and recursive inference.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import (
    FEATURE_COLS,
    MODEL_CHECKSUM_PATH,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from src.features import compute_rsi

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def split_time_series(
    data: pd.DataFrame,
    split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series strictly sequentially (no random shuffling)
    to prevent temporal leakage.
    """
    trainable = data.dropna(subset=["Target"]).reset_index(drop=True)
    X = trainable[feature_cols].values
    y = trainable["Target"].values
    close_today = trainable["Close"].values
    target_dates = trainable["Target_Date"].values

    split_idx = int(len(trainable) * split_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    close_test = close_today[split_idx:]
    dates_test = target_dates[split_idx:]

    return X_train, X_test, y_train, y_test, close_test, dates_test


def evaluate_naive_baseline(y_test: np.ndarray, close_test: np.ndarray) -> dict[str, float]:
    """
    Persistence naive baseline: predict tomorrow's price = today's price (y_pred = close_t).
    Essential benchmark for financial random walk series.
    """
    return compute_metrics(y_test, close_test)


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[LinearRegression, dict[str, float]]:
    """Train and evaluate an Ordinary Least Squares baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return model, metrics


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    close_test: np.ndarray,
    tune: bool = False,
) -> tuple[RandomForestRegressor, np.ndarray, dict[str, float]]:
    """
    Train Random Forest regressor with optional TimeSeriesSplit hyperparameter tuning.
    Calculates directional accuracy against today's closing price.
    """
    if tune:
        logger.info("Executing GridSearchCV with TimeSeriesSplit...")
        param_grid = {
            "n_estimators": [150, 250],
            "max_depth": [6, 10],
            "min_samples_leaf": [2, 5],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        search = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Directional accuracy: did the model predict the correct direction of change vs today?
    pred_direction = np.sign(y_pred - close_test)
    actual_direction = np.sign(y_test - close_test)
    metrics["Directional Acc (%)"] = float(np.mean(pred_direction == actual_direction) * 100)

    return model, y_pred, metrics


def predict_future_recursive(
    model: Any, history_df: pd.DataFrame, n_days: int = 7, feature_cols: list[str] = FEATURE_COLS
) -> pd.DataFrame:
    """
    Perform multi-step recursive price forecasting into the future.
    Each predicted step t+1 is appended to update lag and rolling metrics for step t+2.
    """
    df = history_df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["Close"].astype(float)

    predictions = []
    current_df = df.copy()

    for _ in range(n_days):
        tmp = current_df.copy()
        tmp["MA7"] = tmp["Close"].rolling(window=7).mean()
        tmp["MA30"] = tmp["Close"].rolling(window=30).mean()
        tmp["MA90"] = tmp["Close"].rolling(window=90).mean()
        tmp["Return"] = tmp["Close"].pct_change()
        tmp["Volatility30"] = tmp["Return"].rolling(window=30).std()
        tmp["RSI14"] = compute_rsi(tmp["Close"])
        for lag in [1, 2, 3, 5]:
            tmp[f"Close_lag{lag}"] = tmp["Close"].shift(lag)

        last_features = tmp[feature_cols].iloc[[-1]].values
        predicted_price = float(model.predict(last_features)[0])
        next_date = pd.bdate_range(
            start=current_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=1
        )[0]

        predictions.append({"Date": next_date, "Predicted_Close": predicted_price})

        new_row = pd.DataFrame({"Date": [next_date], "Close": [predicted_price]})
        current_df = pd.concat([current_df, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


def calculate_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file for cryptographic integrity verification."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_model(
    model: Any, path: Path, checksum_path: Path | None = MODEL_CHECKSUM_PATH
) -> str:
    """
    Serialize model artifact to disk and generate a SHA-256 checksum
    to safeguard against insecure deserialization and artifact tampering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    checksum = calculate_sha256(path)
    logger.info("Model saved to %s (SHA-256: %s)", path, checksum)

    if checksum_path:
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checksum_path, "w", encoding="utf-8") as f:
            f.write(checksum)
        logger.info("Model checksum persisted to %s", checksum_path)

    return checksum


def load_model(
    path: Path,
    checksum_path: Path | None = MODEL_CHECKSUM_PATH,
    verify_checksum: bool = True,
) -> Any:
    """
    Load serialized model artifact with cryptographic checksum validation.
    Prevents arbitrary code execution via tampered pickle/joblib payloads.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    if verify_checksum and checksum_path and checksum_path.exists():
        expected_checksum = checksum_path.read_text(encoding="utf-8").strip()
        actual_checksum = calculate_sha256(path)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Security Alert: Model artifact checksum mismatch! "
                f"Expected {expected_checksum}, found {actual_checksum}. "
                f"Aborting deserialization to prevent arbitrary code execution."
            )
        logger.info("Model integrity verified (SHA-256 match).")

    return joblib.load(path)
