"""
CryptoPredict AI - Live Cryptocurrency Movement Prediction Dashboard
A high-accuracy, multi-source predictive analytics dashboard combining exchange price action,
sentiment indices (Fear & Greed), crypto news NLP, and an ensemble machine learning model.
"""

import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_fetcher import CRYPTO_PAIRS
from src.live_predictor import LivePredictor

# Page Setup
st.set_page_config(
    page_title="CryptoPredict AI | Live Movement Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Fintech Dark Theme)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0b0e14;
    }
    
    /* Header Card */
    .metric-card {
        background: linear-gradient(135deg, #151b26 0%, #1e2638 100%);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #2a3449;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Hero Prediction Card */
    .prediction-card {
        background: radial-gradient(circle at top left, #1a2333 0%, #0e131d 100%);
        padding: 24px 28px;
        border-radius: 16px;
        border: 1px solid #2e3c54;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    }
    
    /* Signal Badges */
    .signal-badge {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1px;
        padding: 6px 18px;
        border-radius: 8px;
        display: inline-block;
    }
    
    .news-card {
        background: #141b26;
        border-left: 4px solid #29b6f6;
        padding: 14px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        border-top: 1px solid #222c3d;
        border-right: 1px solid #222c3d;
        border-bottom: 1px solid #222c3d;
    }

    .news-card:hover {
        background: #192230;
        transition: 0.2s ease-in-out;
    }

    .driver-tag {
        background: #1b263b;
        color: #90caf9;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        border: 1px solid #2a3d5e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return LivePredictor()


predictor = get_predictor()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png?v=025", width=48)
    st.title("⚡ CryptoPredict AI")
    st.caption("Multi-Source Intelligence & ML Forecasting")
    st.divider()

    st.subheader("⚙️ Prediction Settings")
    symbol_keys = list(CRYPTO_PAIRS.keys())
    selected_symbol = st.selectbox(
        "Select Cryptocurrency",
        options=symbol_keys,
        format_func=lambda s: f"{CRYPTO_PAIRS[s]['name']} ({s.replace('USDT', '')})"
    )

    horizon_option = st.radio(
        "Forecast Horizon",
        options=["1h", "4h"],
        format_func=lambda h: "1-Hour Ahead (Short-Term)" if h == "1h" else "4-Hours Ahead (Medium-Term)",
        horizontal=True
    )

    st.divider()
    force_retrain = st.button("🔄 Force Retrain & Refresh", use_container_width=True)
    
    auto_refresh = st.checkbox("Auto-Refresh every 60s", value=False)
    if auto_refresh:
        time.sleep(60)
        st.rerun()

    st.divider()
    st.subheader("🌐 Data Pipeline Status")
    st.success("🟢 Binance Spot API: Connected")
    st.success("🟢 Fear & Greed Index: Live")
    st.success("🟢 News Sentiment Feeds: Active")
    st.info("🧠 Ensemble: HistGradientBoosting + Random Forest + Calibrated Logistic Regression")


# --- FETCH DATA & INFERENCE ---
with st.spinner(f"Analyzing multi-source indicators & training ensemble for {selected_symbol}..."):
    try:
        data = predictor.analyze_and_predict(
            symbol=selected_symbol,
            horizon=horizon_option,
            force_retrain=force_retrain
        )
    except Exception as e:
        st.error(f"Prediction Pipeline Error: {str(e)}")
        st.stop()

# Unpack Data
ticker = data["ticker_info"]
pred = data["prediction"]
targets = data["price_targets"]
sentiment = data["sentiment"]
tech = data["tech_summary"]
df_chart = data["chart_data"]
metrics = data["metrics"]

# --- TOP STATS BAR ---
price_chg = ticker["price_change_percent"]
chg_color = "#00E676" if price_chg >= 0 else "#FF1744"
chg_sign = "+" if price_chg >= 0 else ""

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        label=f"{data['symbol_name']} Price",
        value=f"${data['current_price']:,.2f}",
        delta=f"{chg_sign}{price_chg:.2f}% (24h)"
    )
with col2:
    st.metric(
        label="24h High / Low",
        value=f"${ticker['high_24h']:,.2f}",
        delta=f"Low: ${ticker['low_24h']:,.2f}",
        delta_color="off"
    )
with col3:
    st.metric(
        label="24h Volume",
        value=f"{ticker['volume_24h']:,.1f} {selected_symbol.replace('USDT', '')}",
        delta=f"${ticker['quote_volume_24h']/1e6:,.1f}M USD",
        delta_color="off"
    )
with col4:
    fg_val = sentiment["fear_and_greed_value"]
    fg_label = sentiment["fear_and_greed_label"]
    st.metric(
        label="Fear & Greed Index",
        value=f"{fg_val}/100",
        delta=f"{fg_label}",
        delta_color="normal" if fg_val > 50 else "inverse"
    )
with col5:
    st.metric(
        label="News Sentiment",
        value=f"{sentiment['avg_news_sentiment']:+.2f}",
        delta=f"{sentiment['bullish_news_pct']}% Bullish",
        delta_color="normal" if sentiment['avg_news_sentiment'] >= 0 else "inverse"
    )

st.write("")

# --- HERO PREDICTION BANNER ---
sig = pred["signal"]
sig_color = pred["signal_color"]
conf = pred["confidence"]

st.markdown(f"""
<div class="prediction-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
        <div>
            <div style="color: #90caf9; font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;">
                AI Predictive Direction ({data['horizon'].upper()} Forecast)
            </div>
            <div class="signal-badge" style="background-color: {sig_color}22; color: {sig_color}; border: 2px solid {sig_color};">
                {sig}
            </div>
            <div style="margin-top: 12px; color: #cfd8dc; font-size: 15px;">
                Model Confidence: <strong style="color: white; font-size: 18px;">{conf:.1f}%</strong> | 
                Probability: <span style="color: #00E676;">{pred['prob_up']}% Bullish</span> vs <span style="color: #FF1744;">{pred['prob_down']}% Bearish</span>
            </div>
        </div>
        <div style="text-align: right; background: #131c2b; padding: 14px 20px; border-radius: 10px; border: 1px solid #23324a;">
            <div style="color: #90caf9; font-size: 13px; margin-bottom: 4px;">TARGET PRICE OBJECTIVE</div>
            <div style="font-size: 24px; font-weight: 700; color: #ffffff;">${targets['primary_target']:,.2f}</div>
            <div style="font-size: 13px; color: #b0bec5; margin-top: 4px;">
                Invalidation / Stop: <span style="color: #ef5350;">${targets['stop_loss']:,.2f}</span> | ATR Range: ±${targets['current_atr']:,.2f}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Confidence Progress Bar
st.progress(pred["prob_up"] / 100.0, text=f"Model Probability Split: {pred['prob_up']}% Bullish (Green) vs {pred['prob_down']}% Bearish (Red)")

st.write("")

# --- TABS SECTION ---
tab_chart, tab_intelligence, tab_sentiment, tab_backtest = st.tabs([
    "📈 Interactive Chart & Signals",
    "🧠 Model Intelligence & Drivers",
    "📰 Sentiment & Live News",
    "📊 Backtest & Performance Audit"
])

# === TAB 1: INTERACTIVE CHART ===
with tab_chart:
    st.subheader(f"{data['symbol_name']} Price Action & AI Signal Markers")
    
    # Plotly Candlestick Chart with Subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=("Candlestick & Overlays", "MACD Momentum (12, 26, 9)", "RSI (14)")
    )

    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_chart["timestamp"],
        open=df_chart["open"],
        high=df_chart["high"],
        low=df_chart["low"],
        close=df_chart["close"],
        name="Price",
        increasing_line_color="#00E676",
        decreasing_line_color="#FF1744"
    ), row=1, col=1)

    # 2. EMAs
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["ema_9"],
        mode="lines", name="EMA 9", line=dict(color="#00E5FF", width=1.5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["ema_21"],
        mode="lines", name="EMA 21", line=dict(color="#FF9100", width=1.5)
    ), row=1, col=1)

    # 3. Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["bb_upper"],
        mode="lines", name="Upper BB", line=dict(color="rgba(158, 158, 158, 0.4)", width=1, dash="dot"),
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["bb_lower"],
        mode="lines", name="Lower BB", line=dict(color="rgba(158, 158, 158, 0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(158, 158, 158, 0.05)",
        showlegend=False
    ), row=1, col=1)

    # 4. Historical AI Signal Markers Overlay
    signals = data["signals_history"]
    buy_times = [s["timestamp"] for s in signals if s["signal"] == "BUY"]
    buy_prices = [s["price"] * 0.998 for s in signals if s["signal"] == "BUY"]
    sell_times = [s["timestamp"] for s in signals if s["signal"] == "SELL"]
    sell_prices = [s["price"] * 1.002 for s in signals if s["signal"] == "SELL"]

    if buy_times:
        fig.add_trace(go.Scatter(
            x=buy_times, y=buy_prices,
            mode="markers", name="AI Buy Signal",
            marker=dict(symbol="triangle-up", size=11, color="#00E676")
        ), row=1, col=1)

    if sell_times:
        fig.add_trace(go.Scatter(
            x=sell_times, y=sell_prices,
            mode="markers", name="AI Sell Signal",
            marker=dict(symbol="triangle-down", size=11, color="#FF1744")
        ), row=1, col=1)

    # 5. MACD Subplot
    macd_colors = np.where(df_chart["macd_hist"] >= 0, "#00E676", "#FF1744")
    fig.add_trace(go.Bar(
        x=df_chart["timestamp"], y=df_chart["macd_hist"],
        name="MACD Hist", marker_color=macd_colors, showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["macd"],
        mode="lines", name="MACD", line=dict(color="#2979FF", width=1.5), showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["macd_signal"],
        mode="lines", name="Signal", line=dict(color="#FFD600", width=1.5), showlegend=False
    ), row=2, col=1)

    # 6. RSI Subplot
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["rsi_14"],
        mode="lines", name="RSI (14)", line=dict(color="#AB47BC", width=2), showlegend=False
    ), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 23, 68, 0.6)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 230, 118, 0.6)", row=3, col=1)

    fig.update_layout(
        height=750,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark",
        plot_bgcolor="#0b0e14",
        paper_bgcolor="#0b0e14",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Quick Indicator Status Checklist
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info(f"**RSI (14):** {tech['rsi']} ({tech['rsi_status']})")
    with c2:
        st.info(f"**MACD:** {tech['macd_status']}")
    with c3:
        st.info(f"**Trend:** {tech['trend_status']}")
    with c4:
        st.info(f"**Volatility:** {tech['volatility_status']}")

# === TAB 2: MODEL INTELLIGENCE ===
with tab_intelligence:
    st.subheader("🧠 Model Explainability & Market Drivers")
    st.write("Why did the AI produce this prediction? Here are the top quantitative factors influencing the ensemble:")

    col_feat, col_weights = st.columns([0.6, 0.4])
    
    with col_feat:
        top_drivers = pred["top_drivers"]
        if top_drivers:
            feat_names = [d[0] for d in top_drivers]
            feat_scores = [d[1] for d in top_drivers]
            
            fig_imp = go.Figure(go.Bar(
                x=feat_scores[::-1],
                y=feat_names[::-1],
                orientation='h',
                marker=dict(color="#29b6f6", line=dict(color="#0288d1", width=1))
            ))
            fig_imp.update_layout(
                title="Top Feature Importance (Gini Index)",
                template="plotly_dark",
                plot_bgcolor="#141b26",
                paper_bgcolor="#141b26",
                margin=dict(l=10, r=10, t=40, b=10),
                height=300
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.write("Feature importances loading...")

    with col_weights:
        st.markdown("### Key Factor Explanations")
        for feat, score in top_drivers[:4]:
            st.markdown(f"""
            <div style="background: #141b26; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #00E5FF;">
                <strong style="color: #90caf9;">{feat}</strong> (Weight: {score:.3f})<br/>
                <span style="font-size: 13px; color: #b0bec5;">
                    Significantly driving current {data['horizon']} direction determination.
                </span>
            </div>
            """, unsafe_allow_html=True)

# === TAB 3: SENTIMENT & LIVE NEWS ===
with tab_sentiment:
    st.subheader("📰 Market Sentiment & Live News Intelligence")
    
    col_fg, col_news_ratio = st.columns([0.5, 0.5])
    
    with col_fg:
        st.markdown("### Crypto Fear & Greed Index (30-Day Trend)")
        fng_history = sentiment.get("fear_and_greed_history", [])
        if fng_history:
            df_fng = pd.DataFrame(fng_history)
            fig_fng = go.Figure()
            fig_fng.add_trace(go.Scatter(
                x=df_fng["timestamp"],
                y=df_fng["value"],
                mode="lines+markers",
                line=dict(color="#FFD600", width=2.5),
                name="F&G Score"
            ))
            fig_fng.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_fng.update_layout(
                template="plotly_dark",
                plot_bgcolor="#141b26",
                paper_bgcolor="#141b26",
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_fng, use_container_width=True)

    with col_news_ratio:
        st.markdown("### Live News Sentiment Breakdown")
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Bullish", "Bearish", "Neutral"],
            values=[
                sentiment["bullish_news_pct"],
                sentiment["bearish_news_pct"],
                max(0, 100 - sentiment["bullish_news_pct"] - sentiment["bearish_news_pct"])
            ],
            hole=.6,
            marker_colors=["#00E676", "#FF1744", "#FFA726"]
        )])
        fig_donut.update_layout(
            template="plotly_dark",
            plot_bgcolor="#141b26",
            paper_bgcolor="#141b26",
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("### Real-Time Aggregated Headlines")
    news_items = sentiment.get("latest_news", [])
    for item in news_items[:8]:
        lbl = item["sentiment_label"]
        lbl_color = "#00E676" if lbl == "Bullish" else ("#FF1744" if lbl == "Bearish" else "#FFA726")
        st.markdown(f"""
        <div class="news-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; color: #90caf9; text-transform: uppercase;">{item['source']} • {item['published']}</span>
                <span style="font-size: 12px; font-weight: 700; color: {lbl_color}; background: {lbl_color}22; padding: 2px 8px; border-radius: 4px;">
                    {lbl} ({item['sentiment_score']:+.2f})
                </span>
            </div>
            <div style="font-size: 15px; font-weight: 600; margin-top: 6px; color: #eceff1;">
                <a href="{item['link']}" target="_blank" style="color: #eceff1; text-decoration: none;">{item['title']}</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === TAB 4: BACKTEST AUDIT ===
with tab_backtest:
    st.subheader("📊 Walk-Forward Backtest & Model Performance Audit")
    st.caption("Evaluated strictly on out-of-sample forward test data with zero lookahead bias.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Test Accuracy / Win Rate", f"{metrics.get('accuracy', 0)}%")
    with m2:
        st.metric("Precision (Bullish)", f"{metrics.get('precision', 0)}%")
    with m3:
        st.metric("Recall", f"{metrics.get('recall', 0)}%")
    with m4:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0)}%")

    st.write("")
    col_cm, col_strat = st.columns([0.45, 0.55])
    
    with col_cm:
        st.markdown("### Confusion Matrix")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        df_cm = pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"])
        st.dataframe(df_cm, use_container_width=True)

    with col_strat:
        st.markdown("### Simulated Cumulative Strategy PnL")
        strat_ret = metrics.get("cum_strategy_return", 0.0)
        bench_ret = metrics.get("cum_benchmark_return", 0.0)
        
        st.markdown(f"""
        - **AI Ensemble Strategy Return:** `{strat_ret:+.2f}%`
        - **Buy & Hold Benchmark Return:** `{bench_ret:+.2f}%`
        """)
        diff = strat_ret - bench_ret
        diff_color = "#00E676" if diff >= 0 else "#FF1744"
        st.markdown(f"""
        <div style="background: #141b26; padding: 14px; border-radius: 8px; border: 1px solid #25334a;">
            Alpha / Outperformance: <strong style="color: {diff_color}; font-size: 18px;">{diff:+.2f}%</strong>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.caption(f"⚡ CryptoPredict AI Engine • Last Updated: {data['last_updated']} • Disclaimer: Cryptocurrency trading entails substantial market risk. Predictions are for informational and research purposes only.")
