"""
Feature engineering pipeline for financial time-series.
"""

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using rolling average gains and losses."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """
    Construct moving averages, volatility, RSI, and lag features.
    
    Target is set to the NEXT trading day close (t+1).
    All features for row t strictly depend on data up to day t (zero lookahead leakage).
    """
    d = df[["Date", "Close"]].copy()
    d["MA7"] = d["Close"].rolling(window=7).mean()
    d["MA30"] = d["Close"].rolling(window=30).mean()
    d["MA90"] = d["Close"].rolling(window=90).mean()
    d["Return"] = d["Close"].pct_change()
    d["Volatility30"] = d["Return"].rolling(window=30).std()
    d["RSI14"] = compute_rsi(d["Close"])

    for lag in [1, 2, 3, 5]:
        d[f"Close_lag{lag}"] = d["Close"].shift(lag)

    # Target definition: next trading day's close
    d["Target"] = d["Close"].shift(-1)
    d["Target_Date"] = d["Date"].shift(-1)

    # Filter out initial warm-up rows where moving averages / lags are NaN
    valid_rows = d[feature_cols].notna().all(axis=1)
    return d[valid_rows].reset_index(drop=True)
