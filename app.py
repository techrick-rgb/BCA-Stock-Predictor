"""
BCA Stock Predictor - Production Serving Dashboard (Streamlit)
==============================================================
Decoupled serving application utilizing pre-trained artifacts and
modular pipeline components from `src/`.
"""

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    FEATURE_COLS,
    MODEL_PATH,
    RAW_DATA_PATH,
    TICKER,
)
from src.data import fetch_stock_data
from src.features import build_features
from src.models import (
    compute_metrics,
    evaluate_naive_baseline,
    load_model,
    predict_future_recursive,
    save_model,
    split_time_series,
    train_random_forest,
)

st.set_page_config(
    page_title="BCA Stock Predictor",
    page_icon="📈",
    layout="wide",
)

TRANSLATIONS = {
    "en": {
        "settings": "Settings",
        "start_date": "Historical data start date",
        "n_days": "Forecast horizon (trading days ahead)",
        "model_mode": "Model Mode",
        "mode_serving": "Fast Serving (Pre-trained Artifact)",
        "mode_retrain": "Retrain Model (GridSearchCV)",
        "run": "Run Analysis",
        "sidebar_caption": (
            "Data source: Yahoo Finance (`BBCA.JK`) with local fallback.\n"
            "Architecture: Decoupled batch pipeline & serving dashboard.\n"
            "Disclaimer: Educational and portfolio evaluation only."
        ),
        "title": "BCA Stock Price Forecasting",
        "subtitle": "PT Bank Central Asia Tbk (BBCA.JK) • Time-Series Modeling & Persistence Benchmark",
        "info_start": "Select parameters in the sidebar and click **Run Analysis** to inspect predictions.",
        "spinner_fetch": "Ingesting stock candles...",
        "err_nodata": "No data available. Check your internet connection or local dataset.",
        "m_latest_price": "Latest Close Price",
        "m_daily_change": "Daily Change",
        "m_data_points": "Available Observations",
        "m_period": "Date Range",
        "days_unit": "{n} trading days",
        "sec_price": "Historical Price & Technical Trends",
        "spinner_eval": "Evaluating model against test set...",
        "sec_eval": "Model Performance (Hold-Out Test Set)",
        "m_dir_acc": "Directional Hit Rate",
        "sec_baseline": "Benchmark: Machine Learning vs. Naive Random Walk",
        "row_model": "Random Forest Regressor",
        "row_naive": "Naive Persistence Baseline (t-1)",
        "beats_yes": "Model RMSE ({m}) outperformed the naive persistence baseline ({b}).",
        "beats_no": "The naive persistence baseline (RMSE {b}) matches or exceeds Random Forest ({m}). Typical for daily financial random walks without macro/orderbook features.",
        "forecast_note": "Recursive multi-step forecast: prediction at step t is iteratively fed back as lag for step t+1.",
        "trace_actual": "Actual Close",
        "trace_pred": "Predicted Close",
        "expander_fi": "Feature Importance Breakdown",
        "sec_future": "{n}-Trading-Day Forecast",
        "spinner_future": "Generating future forecast...",
        "trace_hist": "Historical (Last 60 Days)",
        "trace_forecast": "Recursive Forecast",
        "col_date": "Date",
        "col_price": "Predicted Close (IDR)",
        "download": "Export Forecast (CSV)",
        "disclaimer": (
            "Disclaimer: This dashboard is developed for engineering evaluation and portfolio presentation. "
            "Stock market movements are governed by macroeconomic factors and order flow not captured in price-derived technical features."
        ),
    },
    "id": {
        "settings": "Pengaturan",
        "start_date": "Data historis mulai dari",
        "n_days": "Cakupan prediksi (hari bursa ke depan)",
        "model_mode": "Mode Model",
        "mode_serving": "Fast Serving (Artefak Pre-trained)",
        "mode_retrain": "Latih Ulang (GridSearchCV)",
        "run": "Jalankan Analisis",
        "sidebar_caption": (
            "Sumber data: Yahoo Finance (`BBCA.JK`) dengan fallback lokal.\n"
            "Arsitektur: Pipeline batch & dashboard inferensi terpisah.\n"
            "Disclaimer: Khusus keperluan evaluasi teknis & portofolio."
        ),
        "title": "Prediksi Harga Saham BCA (BBCA.JK)",
        "subtitle": "PT Bank Central Asia Tbk • Pemodelan Time-Series & Benchmark Baseline Naif",
        "info_start": "Atur parameter di sidebar lalu klik **Jalankan Analisis** untuk melihat hasil.",
        "spinner_fetch": "Mengambil data candle saham...",
        "err_nodata": "Data tidak ditemukan. Cek koneksi internet atau dataset lokal.",
        "m_latest_price": "Harga Penutupan Terakhir",
        "m_daily_change": "Perubahan Harian",
        "m_data_points": "Jumlah Observasi",
        "m_period": "Rentang Periode",
        "days_unit": "{n} hari bursa",
        "sec_price": "Harga Historis & Indikator Tren",
        "spinner_eval": "Mengevaluasi model pada data uji...",
        "sec_eval": "Evaluasi Model (Data Uji Hold-Out)",
        "m_dir_acc": "Akurasi Arah (Hit Rate)",
        "sec_baseline": "Benchmark: Machine Learning vs. Baseline Naif",
        "row_model": "Random Forest Regressor",
        "row_naive": "Baseline Naif (Persistensi t-1)",
        "beats_yes": "RMSE Model ({m}) mengungguli baseline naif ({b}).",
        "beats_no": "Baseline naif (RMSE {b}) setara atau mengungguli Random Forest ({m}). Karakteristik umum pada time-series harga saham harian tanpa data order flow/fundamental.",
        "forecast_note": "Prediksi rekursif multi-step: estimasi hari t diumpankan kembali sebagai lag hari t+1.",
        "trace_actual": "Harga Aktual",
        "trace_pred": "Harga Prediksi",
        "expander_fi": "Tingkat Kepentingan Fitur (Feature Importance)",
        "sec_future": "Prediksi {n} Hari Bursa ke Depan",
        "spinner_future": "Menghitung proyeksi harga masa depan...",
        "trace_hist": "Historis (60 Hari Terakhir)",
        "trace_forecast": "Proyeksi Rekursif",
        "col_date": "Tanggal",
        "col_price": "Prediksi Harga (Rp)",
        "download": "Unduh Hasil Prediksi (CSV)",
        "disclaimer": (
            "Disclaimer: Dashboard ini dibangun untuk evaluasi data engineering dan portofolio. "
            "Pergerakan harga saham dipengaruhi faktor makroekonomi dan sentimen pasar yang tidak sepenuhnya tercermin pada indikator teknikal."
        ),
    },
}

LANGUAGES = {"English": "en", "Bahasa Indonesia": "id"}
_lang_choice = st.sidebar.selectbox("Language / Bahasa", list(LANGUAGES.keys()))
LANG = LANGUAGES[_lang_choice]


def t(key: str, **kwargs) -> str:
    """Localization lookup."""
    text = TRANSLATIONS[LANG].get(key) or TRANSLATIONS["en"][key]
    return text.format(**kwargs) if kwargs else text


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
    """Fetch stock data with local caching and offline fallback."""
    return fetch_stock_data(ticker=ticker, start_date=start_date, fallback_path=RAW_DATA_PATH)


@st.cache_resource(show_spinner=False)
def get_model_artifact(model_path: Path):
    """Load serialized model artifact from storage."""
    if model_path.exists():
        return load_model(model_path)
    return None


# Sidebar controls
st.sidebar.header(t("settings"))
start_date = st.sidebar.date_input(
    t("start_date"), value=date(2015, 1, 1), max_value=date.today() - timedelta(days=30)
)
n_days_future = st.sidebar.slider(t("n_days"), min_value=1, max_value=30, value=7)
mode = st.sidebar.radio(
    t("model_mode"),
    [t("mode_serving"), t("mode_retrain")],
    index=0,
)
run_button = st.sidebar.button(t("run"), type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(t("sidebar_caption"))

# Main view
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
    raw_df = get_cached_stock_data(TICKER, start_date.isoformat())

if raw_df.empty:
    st.error(t("err_nodata"))
    st.stop()

data = build_features(raw_df)

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
latest_price = data["Close"].iloc[-1]
prev_price = data["Close"].iloc[-2]
daily_diff = latest_price - prev_price
daily_pct = (daily_diff / prev_price) * 100

col1.metric(t("m_latest_price"), f"Rp {latest_price:,.0f}")
col2.metric(t("m_daily_change"), f"{daily_diff:+,.0f}", f"{daily_pct:+.2f}%")
col3.metric(t("m_data_points"), t("days_unit", n=f"{len(data):,}"))
col4.metric(t("m_period"), f"{data['Date'].min().strftime('%Y-%m-%d')} → {data['Date'].max().strftime('%Y-%m-%d')}")

# Price Chart
st.subheader(t("sec_price"))
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="#1f77b4", width=1.5)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA7"], name="MA7", line=dict(color="#ff7f0e", width=1)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA30"], name="MA30", line=dict(color="#2ca02c", width=1)))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data["MA90"], name="MA90", line=dict(color="#9467bd", width=1)))
fig_price.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=25, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
)
st.plotly_chart(fig_price, use_container_width=True)

# Model Serving & Evaluation
with st.spinner(t("spinner_eval")):
    X_train, X_test, y_train, y_test, close_test, dates_test = split_time_series(data)

    model = None
    if mode == t("mode_serving"):
        model = get_model_artifact(MODEL_PATH)

    if model is None:
        # Fallback or Retrain
        model, y_pred, metrics = train_random_forest(
            X_train, y_train, X_test, y_test, close_test, tune=(mode == t("mode_retrain"))
        )
        save_model(model, MODEL_PATH)
    else:
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        pred_direction = np.sign(y_pred - close_test)
        actual_direction = np.sign(y_test - close_test)
        metrics["Directional Acc (%)"] = float(np.mean(pred_direction == actual_direction) * 100)

    baseline = evaluate_naive_baseline(y_test, close_test)

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
    st.info(t("beats_no", m=f"{metrics['RMSE']:,.2f}", b=f"{baseline['RMSE']:,.2f}"))

# Predicted vs Actual Chart
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=dates_test, y=y_test, name=t("trace_actual"), line=dict(color="#1f77b4", width=1.5)))
fig_pred.add_trace(go.Scatter(x=dates_test, y=y_pred, name=t("trace_pred"), line=dict(color="#d62728", width=1.5, dash="dash")))
fig_pred.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=25, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
)
st.plotly_chart(fig_pred, use_container_width=True)

# Feature Importance
with st.expander(t("expander_fi")):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
        fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color="#2ca02c"))
        fig_imp.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Relative Importance",
            template="plotly_white",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# Recursive Future Forecast
st.subheader(t("sec_future", n=n_days_future))
st.caption(t("forecast_note"))

with st.spinner(t("spinner_future")):
    future_df = predict_future_recursive(model, data, n_days=n_days_future)

fig_future = go.Figure()
recent = data.tail(60)
fig_future.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"], name=t("trace_hist"), line=dict(color="#1f77b4", width=1.5)))
fig_future.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted_Close"], name=t("trace_forecast"), line=dict(color="#d62728", width=2, dash="dash"), mode="lines+markers"))
fig_future.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=25, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
)
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
