"""
dashboard/application.py

Unified Alchemist Dashboard - Corrected Live Inference Version.
Interfaces with MLflow Model Registry and LangGraph Agent.

Run: streamlit run dashboard/application.py
"""

import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking.mlflow_utils import AlchemistTracker
from src.features.store import FeatureStore
from src.ingestion.market import MarketIngestion
from src.explainability.shap_engine import SHAPExplainer
from src.agent.graph import run_query

st.set_page_config(
    page_title="⚗️ Alchemist",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"st_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cached Helpers 
@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

@st.cache_resource
def get_tracker(_cfg, symbol):
    return AlchemistTracker(_cfg, symbol=symbol)

@st.cache_resource
def load_prod_model(_tracker):
    """Load the model from the MLflow Registry."""
    return _tracker.load_production_model()

@st.cache_data(ttl=1800)
def load_price(_cfg, symbol):
    """Load raw OHLCV prices for a symbol."""
    return MarketIngestion(_cfg).load(symbol)

@st.cache_data(ttl=1800)
def load_inference_features(_cfg, symbol):
    """
    Build inference features (unlabelled, latest date included)
    using the dedicated build_inference() method.
    """
    store = FeatureStore(_cfg)
    return store.build_inference(symbol)

# Sidebar
def render_sidebar(cfg):
    st.sidebar.title("⚗️ Alchemist")
    st.sidebar.caption("Quantamental Intelligence Engine")

    symbol = st.sidebar.selectbox(
        "Asset", [cfg["assets"]["gold"], cfg["assets"]["silver"]], index=0
    )
    threshold = st.sidebar.slider(
        "Execution Threshold", 0.50, 0.80, 0.55, 0.01
    )

    if st.sidebar.button("🔄 Clear Cache & Refresh", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.chat_history = []
        st.rerun()

    return symbol, threshold

# Tabs
def tab_live_signal(cfg, symbol, threshold, model, inference_df, price_df):
    if model is None:
        st.error(f"No Production model found for {symbol} in MLflow. Run `main.py` first.")
        return

    feature_cols = [c for c in model.feature_cols if c in inference_df.columns]
    if not feature_cols:
        st.error("No matching features between model and feature store. Re-train the model.")
        return

    latest = inference_df[feature_cols].iloc[[-1]]
    latest_date = inference_df.index[-1]

    prob = float(model.predict_proba(latest)[0, 1])
    signal = "🟢 BUY SIGNAL" if prob >= threshold else "🟡 FLAT / NO SIGNAL"

    regime = "UNKNOWN"
    if "regime_code" in inference_df.columns:
        code = inference_df["regime_code"].iloc[-1]
        regime_map = {
            0: "LOW_VOL (Bull)",
            1: "MID_VOL (Transition)",
            2: "HIGH_VOL (Crisis/Bear)",
        }
        regime = regime_map.get(code, "UNKNOWN")

    current_px = price_df["close"].iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Date (Live)", str(latest_date.date()))
    c2.metric("Latest Close", f"${current_px:.2f}")
    c3.metric("ML Probability", f"{prob:.1%}")
    c4.metric("Signal", signal)
    c5.metric("Regime", str(regime))

    st.divider()
    st.subheader("Live Mathematical Drivers (SHAP)")

    with st.spinner("Calculating live SHAP Log-Odds…"):
        background = inference_df[feature_cols].tail(100).fillna(0)
        explainer = SHAPExplainer(model.model, feature_cols)
        explainer.fit(background)
        explainer.compute(latest)

        local_df = explainer.local_explanation(0).head(10)
        colors = ["#00CC96" if v > 0 else "#EF553B" for v in local_df["shap_value"]]
        fig = go.Figure(
            go.Bar(
                x=local_df["shap_value"],
                y=local_df["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}" for v in local_df["feature_value"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Local SHAP — Log-Odds Contribution",
            xaxis_title="Push to Probability (Log-Odds)",
            yaxis=dict(autorange="reversed"),
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Green = Pushing probability UP. Red = DOWN. "
            "Text inside bars = Feature Value."
        )

def tab_backtest(cfg, symbol, model, tracker):
    if model is None:
        st.info("No model found. Run training pipeline.")
        return

    client = mlflow.MlflowClient()
    model_name = tracker.model_name

    try:
        versions = client.get_latest_versions(model_name, stages=["Production", "Staging"])
        versions = list(versions)
        if not versions:
            st.warning(f"No tracked runs or metrics found in MLflow Registry for {model_name}.")
            return

        run_id = versions[0].run_id
        metrics = client.get_run(run_id).data.metrics

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("OOS Sharpe", f"{metrics.get('bt_sharpe_ratio', 0):.2f}")
        c2.metric("OOS Sortino", f"{metrics.get('bt_sortino_ratio', 0):.2f}")
        c3.metric("Max Drawdown", f"{metrics.get('bt_max_drawdown', 0):.1%}")
        c4.metric("Win Rate", f"{metrics.get('bt_win_rate', 0):.1%}")
        c5.metric("Total Trades", str(int(metrics.get("bt_total_trades", 0))))
    except Exception as e:
        st.warning(f"Could not retrieve metrics: {e}")

    st.divider()
    st.info(
        "Full trade log and equity curve available in the MLflow Dashboard. "
        "Metrics shown are strict Purged Walk-Forward Out-Of-Sample performance."
    )


def tab_regime(inference_df, price_df):
    if "regime_code" not in inference_df.columns:
        st.info("No regime data available.")
        return

    regime_series = inference_df["regime_code"].tail(1000)
    price_series = price_df["close"].reindex(regime_series.index).ffill()

    cmap = {
        0: "rgba(0,204,150,0.15)",   # LOW_VOL (Green/Bull)
        1: "rgba(100,100,100,0.15)", # MID_VOL (Gray/Transition)
        2: "rgba(239,85,59,0.15)",   # HIGH_VOL (Red/Bear/Crisis)
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            name="Price",
            line=dict(color="white", width=1.5),
        )
    )

    prev, start_b = None, None
    for date, reg in regime_series.items():
        if reg != prev:
            if prev is not None:
                fig.add_vrect(
                    x0=start_b,
                    x1=date,
                    fillcolor=cmap.get(prev, "gray"),
                    opacity=1.0,
                    layer="below",
                    line_width=0,
                )
            start_b, prev = date, reg

    fig.update_layout(
        title="Asset Price + HMM Volatility Regimes",
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_agent(symbol):
    st.subheader(f"💬 Ask Alchemist ({symbol})")
    st.caption("Agent remembers previous questions in this session.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(
        "Ask about the macro outlook, SHAP drivers, or WGC documents…"
    ):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing Quant Data & RAG Documents…"):
                try:
                    result = run_query(
                        question=prompt,
                        symbol=symbol,
                        thread_id=st.session_state.thread_id,
                    )
                    report = result["final_report"]
                    st.markdown(report)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": report}
                    )
                    with st.expander("View Agent Tool Routing"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Agent Engine Failed: {e}")

#Main
def main():
    from pathlib import Path
    import joblib

    cfg = load_config()
    symbol, threshold = render_sidebar(cfg)
    st.title(f"⚗️ Alchemist — {symbol} Intelligence Engine")

    # ── Load model: MLflow first, then local file 
    tracker = get_tracker(cfg, symbol)
    model = load_prod_model(tracker)

    if model is None:
        # Asset‑specific fallback (e.g. alchemist_model_GLD.joblib)
        local_model_path = Path(f"data/models/alchemist_model_{symbol}.joblib")
        if not local_model_path.exists():
            local_model_path = Path("data/models/alchemist_model.joblib")

        if local_model_path.exists():
            try:
                model = joblib.load(local_model_path)
                st.success(f"✅ Loaded model from local backup for {symbol}.")
            except Exception as e:
                st.warning(f"Local backup recovery failed: {e}")

    # ── Load data ─────────────────────────────────────────────────
    with st.spinner("Synchronizing Engine Systems…"):
        try:
            price_df = load_price(cfg, symbol)
            inference_df = load_inference_features(cfg, symbol)
        except Exception as e:
            st.error(f"❌ Critical data sync failure: {e}")
            return

    # ── Centralised model‑missing check
    if model is None:
        st.warning(
            f"⚠️ **No Production Model Found for {symbol} in the MLflow Registry or Local Storage.**\n\n"
            f"Real‑time signals, backtests, and SHAP drivers are unavailable "
            f"until a model is trained and promoted to **Production**.\n\n"
            f"👉 **Run the training pipeline from your terminal:**"
        )
        st.code(f"python main.py --mode train --symbol {symbol}", language="bash")
        st.divider()
        tab_agent(symbol)
        return

    # Full dashboard layout
    t1, t2, t3, t4 = st.tabs([
        "📡 Live Signal", "📈 Backtest OOS", "🌊 Regime Monitor", "🤖 Research Agent"
    ])
    with t1:
        tab_live_signal(cfg, symbol, threshold, model, inference_df, price_df)
    with t2:
        tab_backtest(cfg, symbol, model, tracker)
    with t3:
        tab_regime(inference_df, price_df)
    with t4:
        tab_agent(symbol)

if __name__ == "__main__":
    main()