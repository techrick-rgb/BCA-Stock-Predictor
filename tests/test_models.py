import unittest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.models import (
    calculate_sha256,
    compute_metrics,
    evaluate_naive_baseline,
    load_model,
    predict_future_recursive,
    save_model,
    split_time_series,
)
from src.features import build_features
import tempfile
from pathlib import Path


class TestModels(unittest.TestCase):
    def test_compute_metrics_perfect(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        metrics = compute_metrics(y, y)
        self.assertAlmostEqual(metrics["RMSE"], 0.0)
        self.assertAlmostEqual(metrics["MAE"], 0.0)
        self.assertAlmostEqual(metrics["R2"], 1.0)
        self.assertAlmostEqual(metrics["MAPE (%)"], 0.0)

    def test_split_time_series_chronology(self):
        dates = pd.date_range("2023-01-01", periods=150, freq="B")
        prices = 5000.0 + np.cumsum(np.random.normal(0, 10, size=150))
        df = pd.DataFrame({"Date": dates, "Close": prices})
        data = build_features(df)

        X_train, X_test, y_train, y_test, close_test, dates_test = split_time_series(data, split_ratio=0.8)
        self.assertEqual(len(X_train) + len(X_test), len(data.dropna(subset=["Target"])))

        # Train dates must strictly precede test dates
        train_dates = data.dropna(subset=["Target"])["Target_Date"].values[:len(X_train)]
        self.assertTrue((train_dates.max() < dates_test.min()))

    def test_predict_future_recursive_shape(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        prices = 1000.0 + np.cumsum(np.random.normal(0, 5, size=120))
        df = pd.DataFrame({"Date": dates, "Close": prices})

        dummy_model = DummyRegressor(strategy="constant", constant=1050.0)
        # Dummy fit on dummy 2D array
        dummy_model.fit(np.zeros((10, 9)), np.zeros(10))

        future_df = predict_future_recursive(dummy_model, df, n_days=5)
        self.assertEqual(len(future_df), 5)
        self.assertIn("Date", future_df.columns)
        self.assertIn("Predicted_Close", future_df.columns)
        # All predictions should match dummy constant
        self.assertTrue((future_df["Predicted_Close"] == 1050.0).all())
        # Dates should be monotonic
        self.assertTrue(future_df["Date"].is_monotonic_increasing)

    def test_model_checksum_verification_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_file = tmp_path / "model.pkl"
            checksum_file = tmp_path / "model.sha256"

            dummy = DummyRegressor(strategy="constant", constant=42.0)
            dummy.fit(np.zeros((5, 2)), np.zeros(5))

            save_model(dummy, model_file, checksum_path=checksum_file)
            self.assertTrue(checksum_file.exists())

            loaded = load_model(model_file, checksum_path=checksum_file, verify_checksum=True)
            self.assertIsNotNone(loaded)

    def test_model_checksum_tampered_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_file = tmp_path / "model.pkl"
            checksum_file = tmp_path / "model.sha256"

            dummy = DummyRegressor(strategy="constant", constant=42.0)
            dummy.fit(np.zeros((5, 2)), np.zeros(5))

            save_model(dummy, model_file, checksum_path=checksum_file)

            # Corrupt the model file
            with open(model_file, "ab") as f:
                f.write(b"\x00TAMPERED")

            # Checksum verification must abort deserialization
            with self.assertRaises(ValueError) as ctx:
                load_model(model_file, checksum_path=checksum_file, verify_checksum=True)
            self.assertIn("Security Alert", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
