import unittest

import pandas as pd

from src.config import RAW_DATA_PATH
from src.data import fetch_stock_data, validate_stock_data


class TestDataIngestion(unittest.TestCase):
    def test_validate_stock_data_success(self):
        df = pd.DataFrame({
            "Date": ["2026-01-02", "2026-01-01"],
            "Close": [5100.0, 5000.0],
            "Open": [5050.0, 4950.0],
        })
        validated = validate_stock_data(df)
        self.assertEqual(len(validated), 2)
        # Should be sorted chronologically
        self.assertEqual(validated["Date"].iloc[0], pd.Timestamp("2026-01-01"))
        self.assertEqual(validated["Close"].iloc[0], 5000.0)

    def test_validate_stock_data_empty(self):
        with self.assertRaises(ValueError):
            validate_stock_data(pd.DataFrame())

    def test_validate_stock_data_missing_column(self):
        df = pd.DataFrame({"Date": ["2026-01-01"], "Open": [5000.0]})
        with self.assertRaises(ValueError):
            validate_stock_data(df)

    def test_validate_stock_data_non_positive_price(self):
        df = pd.DataFrame({"Date": ["2026-01-01"], "Close": [-100.0]})
        with self.assertRaises(ValueError):
            validate_stock_data(df)

    def test_fetch_stock_data_fallback(self):
        if RAW_DATA_PATH.exists():
            df = fetch_stock_data(ticker="INVALID.TICKER", start_date="2020-01-01", fallback_path=RAW_DATA_PATH)
            self.assertFalse(df.empty)
            self.assertIn("Close", df.columns)
            self.assertIn("Date", df.columns)


if __name__ == "__main__":
    unittest.main()
