import unittest

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS
from src.features import build_features, compute_rsi


class TestFeatureEngineering(unittest.TestCase):
    def test_compute_rsi_bounds(self):
        prices = pd.Series([100, 102, 101, 105, 107, 106, 110, 112, 111, 115, 118, 120, 119, 122, 125, 128])
        rsi = compute_rsi(prices, period=14)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    def test_compute_rsi_constant_series_no_nan_or_div_zero(self):
        prices = pd.Series([100.0] * 30)
        rsi = compute_rsi(prices, period=14)
        # Should not raise exception; gain and loss are 0, rs=0/(0+1e-9)=0 -> rsi=100-100=0
        self.assertFalse(np.isinf(rsi.iloc[-1]))

    def test_build_features_schema(self):
        dates = pd.date_range("2024-01-01", periods=150, freq="B")
        prices = 1000.0 + np.cumsum(np.random.normal(0, 5, size=150))
        df = pd.DataFrame({"Date": dates, "Close": prices})

        feats = build_features(df)
        for col in FEATURE_COLS:
            self.assertIn(col, feats.columns)
        self.assertIn("Target", feats.columns)
        self.assertIn("Target_Date", feats.columns)

    def test_zero_lookahead_leakage(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        prices = [float(100 + i) for i in range(120)]
        df = pd.DataFrame({"Date": dates, "Close": prices})

        feats = build_features(df)
        # For any row i, Target must be the actual Close of the next day
        for i in range(len(feats) - 1):
            curr_date = feats["Date"].iloc[i]
            target_date = feats["Target_Date"].iloc[i]
            self.assertGreater(target_date, curr_date)
            # Target price must be the close of target date
            expected_target_price = df[df["Date"] == target_date]["Close"].values[0]
            self.assertEqual(feats["Target"].iloc[i], expected_target_price)


if __name__ == "__main__":
    unittest.main()
