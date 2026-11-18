"""
Standalone training and evaluation pipeline for BCA Stock Predictor.
Runs offline/batch model training, benchmark comparisons, and artifact serialization.
"""

import json
import logging
import sys

from src.config import (
    DEFAULT_START_DATE,
    METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    TICKER,
)
from src.data import fetch_stock_data
from src.features import build_features
from src.models import (
    evaluate_naive_baseline,
    save_model,
    split_time_series,
    train_linear_regression,
    train_random_forest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_pipeline")


def run_pipeline(tune: bool = True):
    logger.info("=== Starting Data Engineering & Model Training Pipeline ===")
    
    # 1. Ingestion
    raw_df = fetch_stock_data(ticker=TICKER, start_date=DEFAULT_START_DATE, fallback_path=RAW_DATA_PATH)
    logger.info("Dataset loaded: %d rows from %s to %s", len(raw_df), raw_df["Date"].min(), raw_df["Date"].max())

    # Save fresh raw snapshot if updated
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Raw snapshot persisted to %s", RAW_DATA_PATH)

    # 2. Feature Engineering
    features_df = build_features(raw_df)
    logger.info("Feature matrix generated: %d usable rows with target labels", len(features_df))

    # 3. Time Series Train/Test Split
    X_train, X_test, y_train, y_test, close_test, dates_test = split_time_series(features_df)
    logger.info("Split details: Train set = %d, Test set = %d", len(X_train), len(X_test))

    # 4. Benchmarking: Naive Baseline
    naive_metrics = evaluate_naive_baseline(y_test, close_test)
    logger.info("Naive Baseline (t-1 persistence): RMSE=%.2f, MAE=%.2f, MAPE=%.2f%%, R2=%.4f",
                naive_metrics["RMSE"], naive_metrics["MAE"], naive_metrics["MAPE (%)"], naive_metrics["R2"])

    # 5. Benchmarking: Linear Regression
    lr_model, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    logger.info("Linear Regression: RMSE=%.2f, MAE=%.2f, MAPE=%.2f%%, R2=%.4f",
                lr_metrics["RMSE"], lr_metrics["MAE"], lr_metrics["MAPE (%)"], lr_metrics["R2"])

    # 6. Main Model: Random Forest (Tuned)
    rf_model, rf_preds, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test, close_test, tune=tune
    )
    logger.info("Random Forest (Tuned): RMSE=%.2f, MAE=%.2f, MAPE=%.2f%%, R2=%.4f, Directional Acc=%.2f%%",
                rf_metrics["RMSE"], rf_metrics["MAE"], rf_metrics["MAPE (%)"], rf_metrics["R2"],
                rf_metrics.get("Directional Acc (%)", 0.0))

    # 7. Model Serialization
    save_model(rf_model, MODEL_PATH)

    # 8. Record Metrics Artifact
    metrics_summary = {
        "ticker": TICKER,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "models": {
            "naive_baseline": naive_metrics,
            "linear_regression": lr_metrics,
            "random_forest_tuned": rf_metrics,
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Evaluation metrics recorded to %s", METRICS_PATH)
    logger.info("=== Pipeline Completed Successfully ===")

    return metrics_summary


if __name__ == "__main__":
    run_pipeline(tune=True)
