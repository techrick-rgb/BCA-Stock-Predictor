"""
Dashboard Interaktif — Prediksi Harga Saham Bank BCA (BBCA.JK)
================================================================
Jalankan dengan:
    streamlit run app.py

Dashboard ini menggunakan pipeline yang sama dengan notebook
`notebooks/BCA_Stock_Prediction.ipynb`, dibungkus jadi aplikasi
interaktif: pilih rentang tanggal, jumlah hari prediksi, lalu
lihat hasilnya langsung tanpa perlu jalankan notebook manual.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import asyncio


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# --------------------------------------------------------------------------
# Konfigurasi halaman
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="BCA Stock Predictor",
    page_icon="📈",
    layout="wide",
)

TICKER = "BBCA.JK"
FEATURE_COLS = [
    "MA7", "MA30", "MA90", "Volatility30", "RSI14",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5",
]


# --------------------------------------------------------------------------
# Fungsi bantu
# --------------------------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[["Date", "Close"]].copy()
    d["MA7"] = d["Close"].rolling(window=7).mean()
    d["MA30"] = d["Close"].rolling(window=30).mean()
    d["MA90"] = d["Close"].rolling(window=90).mean()
    d["Return"] = d["Close"].pct_change()
    d["Volatility30"] = d["Return"].rolling(window=30).std()
    d["RSI14"] = compute_rsi(d["Close"])
    for lag in [1, 2, 3, 5]:
        d[f"Close_lag{lag}"] = d["Close"].shift(lag)
    return d.dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(ticker: str, start_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, auto_adjust=True)
    # yfinance versi terbaru bisa mengembalikan kolom MultiIndex -> ratakan
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Close"] = df["Close"].astype(float)
    return df


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame, tune: bool = True):
    X = data[FEATURE_COLS].values
    y = data["Close"].values
    split_idx = int(len(data) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = data["Date"].values[split_idx:]

    if tune:
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [8, 12, None],
            "min_samples_leaf": [1, 2, 5],
        }
        tscv = TimeSeriesSplit(n_splits=5)
        search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MAPE (%)": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return model, dates_test, y_test, y_pred, metrics


def predict_future(model, data: pd.DataFrame, n_days: int) -> pd.DataFrame:
    history_df = data[["Date", "Close"]].copy()
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    history_df["Close"] = history_df["Close"].astype(float)

    rows = []
    for _ in range(n_days):
        tmp = history_df.copy()
        tmp["MA7"] = tmp["Close"].rolling(window=7).mean()
        tmp["MA30"] = tmp["Close"].rolling(window=30).mean()
        tmp["MA90"] = tmp["Close"].rolling(window=90).mean()
        tmp["Return"] = tmp["Close"].pct_change()
        tmp["Volatility30"] = tmp["Return"].rolling(window=30).std()
        tmp["RSI14"] = compute_rsi(tmp["Close"])
        for lag in [1, 2, 3, 5]:
            tmp[f"Close_lag{lag}"] = tmp["Close"].shift(lag)

        last_features = tmp[FEATURE_COLS].iloc[[-1]]
        next_price = float(model.predict(last_features)[0])
        next_date = pd.bdate_range(
            start=history_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=1
        )[0]
        rows.append({"Tanggal": next_date, "Prediksi_Close": next_price})

        new_row = pd.DataFrame({"Date": [next_date], "Close": [next_price]})
        history_df = pd.concat([history_df, new_row], ignore_index=True)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")
start_date = st.sidebar.date_input(
    "Data historis mulai dari", value=date(2015, 1, 1), max_value=date.today() - timedelta(days=30)
)
n_days_future = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 7)
tune_model = st.sidebar.checkbox("Tuning hyperparameter (lebih akurat, lebih lambat)", value=True)
run_button = st.sidebar.button("🔄 Jalankan / Perbarui Analisis", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber data: Yahoo Finance (`BBCA.JK`). "
    "Model: Random Forest Regressor. "
    "Dibuat untuk tujuan edukasi/portofolio, bukan rekomendasi investasi."
)

# --------------------------------------------------------------------------
# Halaman utama
# --------------------------------------------------------------------------
st.title("📈 BCA Stock Predictor")
st.caption("Prediksi Harga Saham BBCA dengan Machine Learning • Data historis real-time dari Yahoo Finance")

if "has_run" not in st.session_state:
    st.session_state.has_run = False

if run_button:
    st.session_state.has_run = True

if not st.session_state.has_run:
    st.info("⬅️ Atur parameter di sidebar, lalu klik **Jalankan / Perbarui Analisis** untuk memulai.")
    st.stop()

with st.spinner("Mengambil data dari Yahoo Finance..."):
    raw_df = fetch_data(TICKER, start_date.isoformat())

if raw_df.empty:
    st.error("Data tidak ditemukan. Coba ubah tanggal mulai atau cek koneksi internet.")
    st.stop()

data = build_features(raw_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Terakhir", f"Rp {data['Close'].iloc[-1]:,.0f}")
col2.metric(
    "Perubahan Harian Terakhir",
    f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):+,.0f}",
    f"{((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100):+.2f}%",
)
col3.metric("Jumlah Data", f"{len(data):,} hari")
col4.metric("Periode Data", f"{data['Date'].min().date()} → {data['Date'].max().date()}")

# --- Grafik harga & moving average ---
st.subheader("Harga & Moving Average")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="#0d47a1")))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA7"], name="MA7", line=dict(width=1)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA30"], name="MA30", line=dict(width=1)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA90"], name="MA90", line=dict(width=1)))
fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_price, use_container_width=True)

# --- Model & evaluasi ---
with st.spinner("Melatih model Random Forest..."):
    model, dates_test, y_test, y_pred, metrics = train_model(data, tune=tune_model)

st.subheader("Evaluasi Model (Data Uji)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("RMSE", f"{metrics['RMSE']:,.2f}")
m2.metric("MAE", f"{metrics['MAE']:,.2f}")
m3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
m4.metric("R²", f"{metrics['R2']:.4f}")

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=dates_test, y=y_test, name="Aktual", line=dict(color="#0d47a1")))
fig_pred.add_trace(go.Scatter(x=dates_test, y=y_pred, name="Prediksi", line=dict(color="#e65100", dash="dash")))
fig_pred.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_pred, use_container_width=True)

# --- Feature importance ---
with st.expander("🔍 Lihat Feature Importance"):
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color="#2e7d32"))
    fig_imp.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

# --- Prediksi masa depan ---
st.subheader(f"Prediksi {n_days_future} Hari ke Depan")
with st.spinner("Menghitung prediksi masa depan..."):
    future_df = predict_future(model, data, n_days_future)

fig_future = go.Figure()
recent = data.tail(60)
fig_future.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"], name="Historis (60 hari terakhir)", line=dict(color="#0d47a1")))
fig_future.add_trace(go.Scatter(x=future_df["Tanggal"], y=future_df["Prediksi_Close"], name="Prediksi", line=dict(color="#c62828", dash="dash"), mode="lines+markers"))
fig_future.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_future, use_container_width=True)

st.dataframe(
    future_df.rename(columns={"Prediksi_Close": "Prediksi Harga (Rp)"}).style.format({"Prediksi Harga (Rp)": "{:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download Prediksi (CSV)",
    data=future_df.to_csv(index=False).encode("utf-8"),
    file_name=f"prediksi_bbca_{n_days_future}hari.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Dashboard ini dibuat untuk tujuan edukasi & portofolio. "
    "Prediksi model bukan rekomendasi investasi/trading. Selalu lakukan riset mandiri (DYOR)."
)
