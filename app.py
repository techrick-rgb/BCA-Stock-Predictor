"""
Interactive Dashboard: Bank BCA (BBCA.JK) Stock Price Forecasting
=================================================================


Run with:
    streamlit run app.py

This dashboard uses the same pipeline as the notebook
`notebooks/BCA_Stock_Prediction.ipynb`, wrapped as an interactive
app: pick a date range and the number of forecast days, then see
the results right away without running the notebook manually.
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


TRANSLATIONS = {
    "en": {
        "settings": "⚙️ Settings",
        "start_date": "Historical data start date",
        "n_days": "How many days ahead to forecast?",
        "tune": "Hyperparameter tuning (more accurate, slower & heavier)",
        "run": "🔄 Run / Refresh Analysis",
        "sidebar_caption": (
            "Data source: Yahoo Finance (`BBCA.JK`). "
            "Model: Random Forest Regressor. "
            "Built for educational/portfolio purposes, not investment advice."
        ),
        "title": "📈 BCA Stock Predictor",
        "subtitle": "BBCA Stock Price Forecasting with Machine Learning • Real-time historical data from Yahoo Finance",
        "info_start": "⬅️ Set the parameters in the sidebar, then click **Run / Refresh Analysis** to start.",
        "spinner_fetch": "Fetching data from Yahoo Finance...",
        "err_nodata": "No data found. Try changing the start date or check your internet connection.",
        "m_latest_price": "Latest Price",
        "m_daily_change": "Latest Daily Change",
        "m_data_points": "Data Points",
        "m_period": "Data Period",
        "days_unit": "{n} days",
        "sec_price": "Price & Moving Averages",
        "spinner_train": "Training Random Forest model...",
        "sec_eval": "Model Evaluation (Test Set)",
        "m_dir_acc": "Directional Accuracy",
        "sec_baseline": "Model vs. naive baseline",
        "row_model": "Random Forest",
        "row_naive": "Naive baseline (tomorrow = today)",
        "beats_yes": "✅ The model's RMSE ({m}) is lower than the naive baseline ({b}).",
        "beats_no": "⚠️ The model does not beat the naive baseline (RMSE {m} vs {b}). Treat the forecast with extra caution.",
        "forecast_note": "Recursive forecast: each day is predicted from the previous predicted day, so errors accumulate over time.",
        "trace_actual": "Actual",
        "trace_pred": "Predicted",
        "expander_fi": "🔍 View Feature Importance",
        "sec_future": "{n}-Day Forecast",
        "spinner_future": "Calculating future forecast...",
        "trace_hist": "Historical (last 60 days)",
        "trace_forecast": "Forecast",
        "col_date": "Date",
        "col_price": "Predicted Price (Rp)",
        "download": "⬇️ Download Forecast (CSV)",
        "disclaimer": (
            "⚠️ **Disclaimer:** This dashboard was built for educational and portfolio purposes. "
            "Model predictions are not investment or trading advice. Always do your own research (DYOR)."
        ),
    },
    "id": {
        "settings": "⚙️ Pengaturan",
        "start_date": "Data historis mulai dari",
        "n_days": "Prediksi berapa hari ke depan?",
        "tune": "Tuning hyperparameter (lebih akurat, lebih lambat & berat)",
        "run": "🔄 Jalankan / Perbarui Analisis",
        "sidebar_caption": (
            "Sumber data: Yahoo Finance (`BBCA.JK`). "
            "Model: Random Forest Regressor. "
            "Dibuat untuk tujuan edukasi/portofolio, bukan rekomendasi investasi."
        ),
        "title": "📈 BCA Stock Predictor",
        "subtitle": "Prediksi Harga Saham BBCA dengan Machine Learning • Data historis real-time dari Yahoo Finance",
        "info_start": "⬅️ Atur parameter di sidebar, lalu klik **Jalankan / Perbarui Analisis** untuk memulai.",
        "spinner_fetch": "Mengambil data dari Yahoo Finance...",
        "err_nodata": "Data tidak ditemukan. Coba ubah tanggal mulai atau cek koneksi internet.",
        "m_latest_price": "Harga Terakhir",
        "m_daily_change": "Perubahan Harian Terakhir",
        "m_data_points": "Jumlah Data",
        "m_period": "Periode Data",
        "days_unit": "{n} hari",
        "sec_price": "Harga & Moving Average",
        "spinner_train": "Melatih model Random Forest...",
        "sec_eval": "Evaluasi Model (Data Uji)",
        "m_dir_acc": "Akurasi Arah",
        "sec_baseline": "Model vs. baseline naif",
        "row_model": "Random Forest",
        "row_naive": "Baseline naif (besok = hari ini)",
        "beats_yes": "✅ RMSE model ({m}) lebih rendah daripada baseline naif ({b}).",
        "beats_no": "⚠️ Model tidak mengalahkan baseline naif (RMSE {m} vs {b}). Gunakan prediksi dengan ekstra hati-hati.",
        "forecast_note": "Prediksi rekursif: setiap hari diprediksi dari hari prediksi sebelumnya, sehingga error menumpuk seiring waktu.",
        "trace_actual": "Aktual",
        "trace_pred": "Prediksi",
        "expander_fi": "🔍 Lihat Feature Importance",
        "sec_future": "Prediksi {n} Hari ke Depan",
        "spinner_future": "Menghitung prediksi masa depan...",
        "trace_hist": "Historis (60 hari terakhir)",
        "trace_forecast": "Prediksi",
        "col_date": "Tanggal",
        "col_price": "Prediksi Harga (Rp)",
        "download": "⬇️ Download Prediksi (CSV)",
        "disclaimer": (
            "⚠️ **Disclaimer:** Dashboard ini dibuat untuk tujuan edukasi & portofolio. "
            "Prediksi model bukan rekomendasi investasi/trading. Selalu lakukan riset mandiri (DYOR)."
        ),
    },
}

LANGUAGES = {"English": "en", "Bahasa Indonesia": "id"}


_lang_choice = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()))
LANG = LANGUAGES[_lang_choice]


def t(key: str, **kwargs) -> str:
    """Return the text for `key` in the selected language (falls back to English)."""
    text = TRANSLATIONS[LANG].get(key) or TRANSLATIONS["en"][key]
    return text.format(**kwargs) if kwargs else text



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

    # Target = NEXT trading day's close. Features of day t only use data up to
    # day t, so there is no look-ahead (data leakage).
    d["Target"] = d["Close"].shift(-1)
    d["Target_Date"] = d["Date"].shift(-1)

    # Drop only the warm-up rows (incomplete features). The last row has no
    # target yet, but it is kept: its features are used to forecast tomorrow.
    return d[d[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(ticker: str, start_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, auto_adjust=True)
    # Newer yfinance versions may return MultiIndex columns -> flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Close"] = df["Close"].astype(float)
    return df


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame, tune: bool = True):
    # Rows without a target (the most recent day) cannot be used for training
    train_data = data.dropna(subset=["Target"]).reset_index(drop=True)

    X = train_data[FEATURE_COLS].values
    y = train_data["Target"].values              # next-day close
    close_today = train_data["Close"].values     # reference for the naive baseline
    split_idx = int(len(train_data) * 0.85)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    close_test = close_today[split_idx:]
    dates_test = train_data["Target_Date"].values[split_idx:]

    if tune:
        param_grid = {
            "n_estimators": [150, 250],
            "max_depth": [8, 12],
            "min_samples_leaf": [1, 3],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = _regression_metrics(y_test, y_pred)
    # Share of days where the predicted direction (up/down vs today) was right
    metrics["Directional Acc (%)"] = float(
        np.mean(np.sign(y_pred - close_test) == np.sign(y_test - close_test)) * 100
    )
    # Naive baseline: "tomorrow's price = today's price"
    baseline_metrics = _regression_metrics(y_test, close_test)

    return model, dates_test, y_test, y_pred, metrics, baseline_metrics


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

        last_features = tmp[FEATURE_COLS].iloc[[-1]].values
        next_price = float(model.predict(last_features)[0])
        next_date = pd.bdate_range(
            start=history_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=1
        )[0]
        rows.append({"Date": next_date, "Predicted_Close": next_price})

        new_row = pd.DataFrame({"Date": [next_date], "Close": [next_price]})
        history_df = pd.concat([history_df, new_row], ignore_index=True)

    return pd.DataFrame(rows)



st.sidebar.header(t("settings"))
start_date = st.sidebar.date_input(
    t("start_date"), value=date(2015, 1, 1), max_value=date.today() - timedelta(days=30)
)
n_days_future = st.sidebar.slider(t("n_days"), 1, 30, 7)
tune_model = st.sidebar.checkbox(t("tune"), value=False)
run_button = st.sidebar.button(t("run"), type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(t("sidebar_caption"))


st.title(t("title"))
st.caption(t("subtitle"))

if "has_run" not in st.session_state:
    st.session_state.has_run = False

if run_button:
    st.session_state.has_run = True

if not st.session_state.has_run:
    st.info(t("info_start"))
    st.stop()

with st.spinner(t("spinner_fetch")):
    raw_df = fetch_data(TICKER, start_date.isoformat())

if raw_df.empty:
    st.error(t("err_nodata"))
    st.stop()

data = build_features(raw_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric(t("m_latest_price"), f"Rp {data['Close'].iloc[-1]:,.0f}")
col2.metric(
    t("m_daily_change"),
    f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):+,.0f}",
    f"{((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100):+.2f}%",
)
col3.metric(t("m_data_points"), t("days_unit", n=f"{len(data):,}"))
col4.metric(t("m_period"), f"{data['Date'].min().date()} → {data['Date'].max().date()}")


st.subheader(t("sec_price"))
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="#0d47a1")))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA7"], name="MA7", line=dict(width=1)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA30"], name="MA30", line=dict(width=1)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA90"], name="MA90", line=dict(width=1)))
fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_price, use_container_width=True)


with st.spinner(t("spinner_train")):
    model, dates_test, y_test, y_pred, metrics, baseline = train_model(data, tune=tune_model)

st.subheader(t("sec_eval"))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("RMSE", f"{metrics['RMSE']:,.2f}")
m2.metric("MAE", f"{metrics['MAE']:,.2f}")
m3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
m4.metric("R²", f"{metrics['R2']:.4f}")
m5.metric(t("m_dir_acc"), f"{metrics['Directional Acc (%)']:.1f}%")

st.markdown(f"**{t('sec_baseline')}**")
cmp_df = pd.DataFrame(
    {
        "RMSE": [metrics["RMSE"], baseline["RMSE"]],
        "MAE": [metrics["MAE"], baseline["MAE"]],
        "MAPE (%)": [metrics["MAPE (%)"], baseline["MAPE (%)"]],
        "R²": [metrics["R2"], baseline["R2"]],
    },
    index=[t("row_model"), t("row_naive")],
)
st.dataframe(cmp_df.style.format("{:,.2f}"), use_container_width=True)
if metrics["RMSE"] < baseline["RMSE"]:
    st.success(t("beats_yes", m=f"{metrics['RMSE']:,.2f}", b=f"{baseline['RMSE']:,.2f}"))
else:
    st.warning(t("beats_no", m=f"{metrics['RMSE']:,.2f}", b=f"{baseline['RMSE']:,.2f}"))

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=dates_test, y=y_test, name=t("trace_actual"), line=dict(color="#0d47a1")))
fig_pred.add_trace(go.Scatter(x=dates_test, y=y_pred, name=t("trace_pred"), line=dict(color="#e65100", dash="dash")))
fig_pred.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_pred, use_container_width=True)


with st.expander(t("expander_fi")):
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color="#2e7d32"))
    fig_imp.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)


st.subheader(t("sec_future", n=n_days_future))
st.caption(t("forecast_note"))
with st.spinner(t("spinner_future")):
    future_df = predict_future(model, data, n_days_future)

fig_future = go.Figure()
recent = data.tail(60)
fig_future.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"], name=t("trace_hist"), line=dict(color="#0d47a1")))
fig_future.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted_Close"], name=t("trace_forecast"), line=dict(color="#c62828", dash="dash"), mode="lines+markers"))
fig_future.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_future, use_container_width=True)

st.dataframe(
    future_df.rename(columns={"Date": t("col_date"), "Predicted_Close": t("col_price")})
    .style.format({t("col_price"): "{:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)


st.download_button(
    t("download"),
    data=future_df.to_csv(index=False).encode("utf-8"),
    file_name=f"bbca_forecast_{n_days_future}days.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(t("disclaimer"))