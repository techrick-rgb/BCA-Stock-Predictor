"""
Data ingestion, validation, and storage layer.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import RAW_DATA_PATH, TICKER

logger = logging.getLogger(__name__)


def validate_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, handle datetime parsing, and ensure monotonic ordering."""
    if df.empty:
        raise ValueError("Dataframe is empty.")

    required_columns = {"Date", "Close"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    if (df["Close"] <= 0).any():
        raise ValueError("Invalid non-positive stock prices detected.")

    return df


def fetch_stock_data(
    ticker: str = TICKER,
    start_date: str = "2015-01-01",
    fallback_path: Path | None = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Fetch historical daily stock candles with offline fallback resilience.
    Flattens MultiIndex columns if present.
    """
    try:
        logger.info("Attempting to fetch %s from Yahoo Finance...", ticker)
        df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            return validate_stock_data(df)
        logger.warning("Empty response from yfinance. Checking fallback...")
    except Exception as exc:
        logger.warning("Failed to fetch online data (%s). Using fallback...", exc)

    if fallback_path and Path(fallback_path).exists():
        logger.info("Loading fallback data from %s", fallback_path)
        df_local = pd.read_csv(fallback_path)
        return validate_stock_data(df_local)

    raise RuntimeError(
        f"Unable to retrieve stock data for {ticker} from remote API or local fallback."
    )
