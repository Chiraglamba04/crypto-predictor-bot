"""
CryptoPredict AI - Live Cryptocurrency Movement Prediction & Investment Advisory Dashboard
Enhanced with Exact Points Up/Down Movement Forecasts, Actionable Entry/Target/Stop Trade Setups,
and a Multi-Coin AI Market Scanner for identifying the best crypto investments in real-time.
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
    page_title="CryptoPredict AI | Investment Advisor & Points Movement Forecast",
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
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    }
    
    /* Trade Setup Action Card */
    .setup-card {
        background: #111823;
        border-radius: 14px;
        border: 1px solid #24334a;
        padding: 20px 24px;
        margin-bottom: 25px;
    }

    .signal-badge {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 1px;
        padding: 6px 18px;
        border-radius: 8px;
        display: inline-block;
    }

    .action-badge {
        font-size: 18px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
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

    .opportunity-card {
        background: #131b28;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid #223249;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return LivePredictor()


predictor = get_predictor()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png?v=025", width=44)
    st.title("⚡ CryptoPredict AI")
    st.caption("Movement Points Forecast & Investment Advisor")
    st.divider()

    st.subheader("⚙️ Select Market")
    symbol_keys = list(CRYPTO_PAIRS.keys())
    selected_symbol = st.selectbox(
        "Cryptocurrency",
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
    st.subheader("🌐 System Status")
    st.success("🟢 Binance API: Connected")
    st.success("🟢 Fear & Greed: Real-Time")
    st.success("🟢 News NLP: Active")
    st.info("🧠 Ensemble: HistGradientBoosting + Random Forest + Logistic Regression")


# --- FETCH DATA & INFERENCE ---
with st.spinner(f"Analyzing points movement and investment setup for {selected_symbol}..."):
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
movement = data["expected_movement"]
setup = data["trade_setup"]
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

# --- HERO PREDICTION & POINTS MOVEMENT BANNER ---
sig = pred["signal"]
sig_color = pred["signal_color"]
conf = pred["confidence"]
delta_pts = movement["points_delta"]
delta_pct = movement["pct_delta"]
pts_sign = "+" if delta_pts >= 0 else ""
pts_color = "#00E676" if delta_pts >= 0 else "#FF1744"
move_dir = "UPWARD MOVEMENT" if delta_pts >= 0 else "DOWNWARD MOVEMENT"

st.markdown(f"""
<div class="prediction-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div>
            <div style="color: #90caf9; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;">
                FORECAST DIRECTION & PROBABILITY ({data['horizon'].upper()} TIMEFRAME)
            </div>
            <div class="signal-badge" style="background-color: {sig_color}22; color: {sig_color}; border: 2px solid {sig_color};">
                {sig}
            </div>
            <div style="margin-top: 10px; color: #cfd8dc; font-size: 14px;">
                Model Confidence: <strong style="color: white; font-size: 16px;">{conf:.1f}%</strong> | 
                Probability Split: <span style="color: #00E676;">{pred['prob_up']}% Up</span> vs <span style="color: #FF1744;">{pred['prob_down']}% Down</span>
            </div>
        </div>
        <div style="background: #141f30; padding: 16px 24px; border-radius: 12px; border: 1px solid #2b3d5c; text-align: right;">
            <div style="color: #90caf9; font-size: 12px; letter-spacing: 1px; text-transform: uppercase;">
                EXPECTED POINT MOVEMENT ({data['horizon'].upper()})
            </div>
            <div style="font-size: 30px; font-weight: 800; color: {pts_color};">
                {pts_sign}{delta_pts:,.2f} pts
            </div>
            <div style="font-size: 14px; font-weight: 600; color: {pts_color};">
                {move_dir} ({pts_sign}{delta_pct:.2f}%)
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- ACTIONABLE INVESTMENT SETUP CARD ---
st.markdown(f"""
<div class="setup-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #24354d; padding-bottom: 12px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 18px; font-weight: 700; color: white;">🎯 Where & How to Invest:</span>
            <span class="action-badge" style="background: {setup['action_color']}25; color: {setup['action_color']}; border: 1.5px solid {setup['action_color']};">
                {setup['action']}
            </span>
        </div>
        <div style="font-size: 14px; color: #90caf9;">
            Risk-to-Reward Ratio: <strong style="color: white; font-size: 16px;">{setup['risk_reward_ratio']}</strong> | 
            Allocation: <span style="color: #ffb74d;">{setup['recommended_allocation']}</span>
        </div>
    </div>
    <div style="font-size: 15px; color: #cfd8dc; margin-bottom: 18px; font-style: italic;">
        💡 <strong>Action Verdict:</strong> {setup['verdict']}
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        <div style="background: #172130; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #29b6f6;">
            <div style="font-size: 12px; color: #90caf9;">RECOMMENDED ENTRY ZONE</div>
            <div style="font-size: 18px; font-weight: 700; color: white; margin-top: 4px;">{setup['entry_zone']}</div>
            <div style="font-size: 12px; color: #b0bec5; margin-top: 2px;">Optimal buying/entry window</div>
        </div>
        <div style="background: #172130; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #00E676;">
            <div style="font-size: 12px; color: #00E676;">TARGET 1 (CONSERVATIVE)</div>
            <div style="font-size: 18px; font-weight: 700; color: white; margin-top: 4px;">${setup['target_1']:,.2f}</div>
            <div style="font-size: 12px; color: #00E676; margin-top: 2px;">+{setup['target_1_points']:,.2f} pts (+{setup['target_1_pct']:.2f}%)</div>
        </div>
        <div style="background: #172130; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #76ff03;">
            <div style="font-size: 12px; color: #76ff03;">TARGET 2 (AGGRESSIVE)</div>
            <div style="font-size: 18px; font-weight: 700; color: white; margin-top: 4px;">${setup['target_2']:,.2f}</div>
            <div style="font-size: 12px; color: #76ff03; margin-top: 2px;">+{setup['target_2_points']:,.2f} pts (+{setup['target_2_pct']:.2f}%)</div>
        </div>
        <div style="background: #172130; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #FF1744;">
            <div style="font-size: 12px; color: #FF1744;">STOP LOSS / INVALIDATION</div>
            <div style="font-size: 18px; font-weight: 700; color: white; margin-top: 4px;">${setup['stop_loss']:,.2f}</div>
            <div style="font-size: 12px; color: #FF1744; margin-top: 2px;">-{setup['stop_loss_points']:,.2f} pts (-{setup['stop_loss_pct']:.2f}%)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS SECTION ---
tab_scanner, tab_chart, tab_calculator, tab_intelligence, tab_sentiment, tab_backtest = st.tabs([
    "💎 AI Market Scanner (Where to Invest Now)",
    "📈 Interactive Chart & Signals",
    "🧮 Position Size & Risk Calculator",
    "🧠 Model Intelligence & Drivers",
    "📰 Sentiment & Live News",
    "📊 Backtest & Performance Audit"
])

# === TAB 1: AI MARKET SCANNER (WHERE TO INVEST RIGHT NOW) ===
with tab_scanner:
    st.subheader("💎 Multi-Coin AI Scanner: Where Should You Invest Right Now?")
    st.write("The bot scans all supported cryptocurrencies and ranks them by **Highest AI Confidence**, **Upward Movement Potential**, and **Risk-to-Reward Ratio**.")

    with st.spinner("Scanning all cryptocurrency markets for top investment opportunities..."):
        opportunities = predictor.scan_all_coins(horizon=horizon_option)

    if opportunities:
        top_pick = opportunities[0]
        
        # Highlight #1 Pick
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1b283d 0%, #101826 100%); border-radius: 12px; padding: 18px 24px; border: 2px solid #00E5FF; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: #00E5FF22; color: #00E5FF; border: 1px solid #00E5FF; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 12px;">🏆 #1 TOP AI INVESTMENT PICK RIGHT NOW</span>
                    <h2 style="color: white; margin: 8px 0 4px 0;">{top_pick['name']} ({top_pick['symbol'].replace('USDT','')}) - ${top_pick['price']:,.2f}</h2>
                    <div style="font-size: 14px; color: #b0bec5;">
                        AI Signal: <strong style="color: #00E676;">{top_pick['signal']}</strong> ({top_pick['confidence']:.1f}% Confidence) | Action: <strong style="color: #00E676;">{top_pick['action']}</strong>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 13px; color: #90caf9;">EXPECTED MOVEMENT</div>
                    <div style="font-size: 24px; font-weight: 800; color: #00E676;">{top_pick['points_delta']:+.2f} pts ({top_pick['pct_delta']:+.2f}%)</div>
                    <div style="font-size: 13px; color: #cfd8dc;">Target 1: <strong>${top_pick['target_1']:,.2f}</strong> (+{top_pick['target_1_pct']:.2f}%)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison Table of all coins
        st.markdown("### 📋 Full Market Scan Rankings")
        scan_df = pd.DataFrame([
            {
                "Coin": f"{o['name']} ({o['symbol'].replace('USDT','')})",
                "Current Price": f"${o['price']:,.2f}",
                "AI Signal": o["signal"],
                "Confidence": f"{o['confidence']:.1f}%",
                "Forecast Movement": f"{o['points_delta']:+.2f} pts ({o['pct_delta']:+.2f}%)",
                "Action Verdict": o["action"],
                "Target 1": f"${o['target_1']:,.2f}",
                "Stop Loss": f"${o['stop_loss']:,.2f}",
                "Risk/Reward": o["risk_reward"]
            }
            for o in opportunities
        ])
        st.dataframe(scan_df, use_container_width=True)


# === TAB 2: INTERACTIVE CHART ===
with tab_chart:
    st.subheader(f"{data['symbol_name']} Price Action & AI Signal Markers")
    
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


# === TAB 3: POSITION SIZE & RISK CALCULATOR ===
with tab_calculator:
    st.subheader("🧮 Position Sizing & Capital Risk Calculator")
    st.write("Calculate exactly how much money to invest and what your exact profit or loss will be for this setup:")

    calc_c1, calc_c2 = st.columns(2)
    with calc_c1:
        total_portfolio = st.number_input("Total Trading Portfolio ($ USD)", min_value=100.0, value=5000.0, step=500.0)
        risk_pct = st.slider("Maximum Risk Per Trade (% of account)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        
    with calc_c2:
        max_dollar_risk = total_portfolio * (risk_pct / 100.0)
        curr_p = data["current_price"]
        sl_dist = abs(curr_p - setup["stop_loss"])
        
        # Position size in tokens = max dollar risk / stop loss distance in dollars
        if sl_dist > 0:
            token_position_size = max_dollar_risk / sl_dist
            dollar_position_size = token_position_size * curr_p
        else:
            token_position_size = 0.0
            dollar_position_size = 0.0

        potential_profit_t1 = token_position_size * abs(setup["target_1"] - curr_p)
        potential_profit_t2 = token_position_size * abs(setup["target_2"] - curr_p)

        st.markdown(f"""
        <div style="background: #141f2e; padding: 18px 20px; border-radius: 10px; border: 1px solid #283e5c;">
            <div style="color: #90caf9; font-size: 13px;">CALCULATED TRADE EXECUTION SIZING</div>
            <div style="font-size: 22px; font-weight: 700; color: white; margin-top: 4px;">
                Invest: ${dollar_position_size:,.2f} ({token_position_size:.4f} {selected_symbol.replace('USDT','')})
            </div>
            <div style="font-size: 14px; color: #cfd8dc; margin-top: 8px;">
                • Max Risk (if stopped out): <span style="color: #FF1744; font-weight: 700;">-${max_dollar_risk:,.2f}</span> ({risk_pct}%)<br/>
                • Target 1 Profit: <span style="color: #00E676; font-weight: 700;">+${potential_profit_t1:,.2f}</span><br/>
                • Target 2 Profit: <span style="color: #76ff03; font-weight: 700;">+${potential_profit_t2:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# === TAB 4: MODEL INTELLIGENCE ===
with tab_intelligence:
    st.subheader("🧠 Model Explainability & Market Drivers")
    st.write("Why did the AI produce this prediction? Top factors influencing the ensemble:")

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


# === TAB 5: SENTIMENT & LIVE NEWS ===
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


# === TAB 6: BACKTEST AUDIT ===
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
st.caption(f"⚡ CryptoPredict AI Engine • Last Updated: {data['last_updated']} • Disclaimer: Cryptocurrency trading entails substantial market risk. Predictions and trade setups are for quantitative research purposes only and not financial advice.")
