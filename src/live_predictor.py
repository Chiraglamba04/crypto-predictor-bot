"""
Live Predictor Pipeline
High-level orchestrator that connects data fetching, feature engineering, model inference,
actionable trade setups (entry, targets, stop-loss, expected points & pips), and multi-market scanning
across both Cryptocurrency and Forex/Commodities.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.data_fetcher import DataFetcher, CRYPTO_PAIRS, FOREX_PAIRS, ALL_MARKETS
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.model_engine import CryptoModelEngine


class LivePredictor:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 60  # seconds

    def analyze_and_predict(
        self,
        symbol: str = "BTCUSDT",
        horizon: str = "1h",
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Runs the full predictive pipeline for Crypto OR Forex:
        1. Multi-source data ingestion (Candles, F&G/Macro, News RSS)
        2. Technical indicator math & Feature synthesis
        3. Walk-forward ensemble model training or inference
        4. Expected Points Movement & Pips Movement calculation
        5. Actionable Investment Setup (Where to enter, Targets, Invalidation, Risk/Reward)
        6. Historical signals generation for chart overlay
        """
        meta = ALL_MARKETS.get(symbol, {"name": symbol, "pip_size": 0.0001, "digits": 4, "type": "crypto"})
        is_forex = meta.get("type") == "forex"
        pip_size = meta.get("pip_size", 0.0001)
        digits = meta.get("digits", 4)

        # 1. Fetch market data using unified router
        raw_df = self.fetcher.fetch_klines(symbol=symbol, interval="1h", limit=500)
        ticker_info = self.fetcher.fetch_ticker(symbol=symbol)
        sentiment_summary = self.fetcher.get_market_sentiment_summary(asset_type="forex" if is_forex else "crypto")

        # 2. Extract sentiment metrics
        fg_score = sentiment_summary["fear_and_greed_value"]
        news_sentiment = sentiment_summary["avg_news_sentiment"]

        # 3. Engineer features
        fe = FeatureEngineer(fg_score=fg_score, news_sentiment=news_sentiment)
        df_featured = fe.create_features(raw_df, fg_score=fg_score, news_sentiment=news_sentiment)
        
        # 4. Prepare dataset
        X, y, latest_features = fe.prepare_training_dataset(df_featured, horizon=horizon)

        # 5. Model Engine
        engine = CryptoModelEngine(symbol=symbol.replace("=", "_"), horizon=horizon)
        
        model_exists = os.path.exists(engine.model_path)
        if force_retrain or not model_exists:
            metrics = engine.train_and_evaluate(X, y)
        else:
            try:
                engine.load_model()
                metrics = engine.metrics
            except Exception:
                metrics = engine.train_and_evaluate(X, y)

        # 6. Live Prediction
        prediction = engine.predict_live(latest_features)

        # 7. Exact Points & Pips Movement Calculation
        current_price = float(df_featured["close"].iloc[-1])
        current_atr = float(df_featured["atr_14"].iloc[-1])
        multiplier = 1.0 if horizon == "1h" else 2.2
        
        prob_up = prediction["prob_up"] / 100.0
        prob_down = prediction["prob_down"] / 100.0

        # Directional bias scaled by ATR
        directional_bias = prob_up - 0.50
        expected_points_delta = float(directional_bias * 2.0 * current_atr * multiplier)
        expected_pct_delta = float((expected_points_delta / current_price) * 100.0)
        expected_pips_delta = float(expected_points_delta / pip_size)

        # 8. Actionable Investment Setup
        is_bullish = "BULLISH" in prediction["signal"]
        is_bearish = "BEARISH" in prediction["signal"]

        fmt_price = lambda p: f"${p:,.{digits}f}" if not is_forex or "GC=F" in symbol else f"{p:.{digits}f}"

        if is_bullish:
            action = "STRONG BUY / LONG" if "STRONG" in prediction["signal"] else "BUY / ACCUMULATE"
            action_color = "#00E676"
            entry_low = current_price * (0.9995 if is_forex else 0.9985)
            entry_high = current_price * (1.0002 if is_forex else 1.0005)
            
            target_1 = current_price + (current_atr * multiplier * 0.9)
            target_2 = current_price + (current_atr * multiplier * 1.8)
            stop_loss = current_price - (current_atr * multiplier * 0.75)
            
            risk_points = abs(current_price - stop_loss)
            reward_points = abs(target_1 - current_price)
            rr_ratio = reward_points / (risk_points + 1e-6)
            
            t1_points = target_1 - current_price
            t1_pct = (t1_points / current_price) * 100.0
            t2_points = target_2 - current_price
            t2_pct = (t2_points / current_price) * 100.0
            sl_points = current_price - stop_loss
            sl_pct = (sl_points / current_price) * 100.0
            
            verdict = f"High probability upside move. Enter near {fmt_price(entry_low)} - {fmt_price(entry_high)}."
            
        elif is_bearish:
            action = "STRONG SELL / SHORT" if "STRONG" in prediction["signal"] else "SELL / TAKE PROFIT"
            action_color = "#FF1744"
            entry_low = current_price * (0.9998 if is_forex else 0.9995)
            entry_high = current_price * (1.0005 if is_forex else 1.0015)
            
            target_1 = current_price - (current_atr * multiplier * 0.9)
            target_2 = current_price - (current_atr * multiplier * 1.8)
            stop_loss = current_price + (current_atr * multiplier * 0.75)
            
            risk_points = abs(stop_loss - current_price)
            reward_points = abs(current_price - target_1)
            rr_ratio = reward_points / (risk_points + 1e-6)
            
            t1_points = target_1 - current_price
            t1_pct = (t1_points / current_price) * 100.0
            t2_points = target_2 - current_price
            t2_pct = (t2_points / current_price) * 100.0
            sl_points = stop_loss - current_price
            sl_pct = (sl_points / current_price) * 100.0
            
            verdict = f"High probability downward pressure. Consider short position or taking profit."
        else:
            action = "WAIT / NO CLEAR SETUP"
            action_color = "#FFA726"
            entry_low = current_price * 0.998
            entry_high = current_price * 1.002
            target_1 = current_price + (current_atr * 0.5)
            target_2 = current_price + (current_atr * 1.0)
            stop_loss = current_price - (current_atr * 0.5)
            rr_ratio = 1.0
            t1_points = target_1 - current_price
            t1_pct = (t1_points / current_price) * 100.0
            t2_points = target_2 - current_price
            t2_pct = (t2_points / current_price) * 100.0
            sl_points = current_price - stop_loss
            sl_pct = (sl_points / current_price) * 100.0
            verdict = "Market in consolidation range. Wait for clear directional breakout before trading."

        trade_setup = {
            "action": action,
            "action_color": action_color,
            "verdict": verdict,
            "entry_zone": f"{fmt_price(entry_low)} - {fmt_price(entry_high)}",
            "target_1": round(target_1, digits),
            "target_1_points": round(t1_points, digits),
            "target_1_pips": round(t1_points / pip_size, 1),
            "target_1_pct": round(t1_pct, 2),
            "target_2": round(target_2, digits),
            "target_2_points": round(t2_points, digits),
            "target_2_pips": round(t2_points / pip_size, 1),
            "target_2_pct": round(t2_pct, 2),
            "stop_loss": round(stop_loss, digits),
            "stop_loss_points": round(sl_points, digits),
            "stop_loss_pips": round(sl_points / pip_size, 1),
            "stop_loss_pct": round(sl_pct, 2),
            "risk_reward_ratio": f"1 : {rr_ratio:.2f}",
            "recommended_allocation": "1% - 2% of capital" if is_forex else "1% - 3% of capital"
        }

        # 9. Historical signals on the last 50 candles for charting
        signals_history = []
        recent_featured = df_featured.tail(60).copy()
        
        for idx in range(20, len(recent_featured)):
            row = recent_featured.iloc[[idx]][FEATURE_COLUMNS]
            try:
                prob = float(engine.pipeline.predict_proba(row)[:, 1][0])
                if prob >= 0.58:
                    sig = "BUY"
                elif prob <= 0.42:
                    sig = "SELL"
                else:
                    sig = "HOLD"
            except Exception:
                sig = "HOLD"
            signals_history.append({
                "timestamp": recent_featured["timestamp"].iloc[idx],
                "price": recent_featured["close"].iloc[idx],
                "signal": sig
            })

        # 10. Technical Summary Dashboard
        rsi_val = float(df_featured["rsi_14"].iloc[-1])
        macd_val = float(df_featured["macd"].iloc[-1])
        macd_sig_val = float(df_featured["macd_signal"].iloc[-1])
        ema_9_val = float(df_featured["ema_9"].iloc[-1])
        ema_21_val = float(df_featured["ema_21"].iloc[-1])

        tech_summary = {
            "rsi": round(rsi_val, 1),
            "rsi_status": "Oversold (Bullish Reversal Zone)" if rsi_val < 32 else ("Overbought (Bearish Pullback Zone)" if rsi_val > 68 else "Neutral Momentum"),
            "macd_status": "Bullish Crossover" if macd_val > macd_sig_val else "Bearish Momentum",
            "trend_status": "Short-Term Uptrend (EMA 9 > EMA 21)" if ema_9_val > ema_21_val else "Short-Term Downtrend (EMA 9 < EMA 21)",
            "volatility_status": "High Volatility" if float(df_featured["bb_bandwidth"].iloc[-1]) > 5.0 else "Consolidating / Rangebound",
        }

        result = {
            "symbol": symbol,
            "symbol_name": meta.get("name", symbol),
            "display_name": meta.get("display", symbol),
            "is_forex": is_forex,
            "pip_size": pip_size,
            "digits": digits,
            "horizon": horizon,
            "current_price": current_price,
            "ticker_info": ticker_info,
            "prediction": prediction,
            "expected_movement": {
                "points_delta": round(expected_points_delta, digits),
                "pips_delta": round(expected_pips_delta, 1),
                "pct_delta": round(expected_pct_delta, 2),
                "direction": "UP" if expected_points_delta >= 0 else "DOWN",
                "volatility_atr": round(current_atr, digits),
            },
            "trade_setup": trade_setup,
            "sentiment": sentiment_summary,
            "tech_summary": tech_summary,
            "price_targets": {
                "expected_high": round(current_price + (current_atr * multiplier), digits),
                "expected_low": round(current_price - (current_atr * multiplier), digits),
                "primary_target": round(target_1, digits),
                "stop_loss": round(stop_loss, digits),
                "current_atr": round(current_atr, digits),
            },
            "metrics": metrics,
            "signals_history": signals_history,
            "chart_data": df_featured.tail(120),
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        return result

    def scan_all_markets(self, market_type: str = "crypto", horizon: str = "1h") -> List[Dict[str, Any]]:
        """
        Scans all supported pairs for either Crypto or Forex.
        """
        target_dict = CRYPTO_PAIRS if market_type == "crypto" else FOREX_PAIRS
        opportunities = []
        for sym, meta in target_dict.items():
            try:
                res = self.analyze_and_predict(symbol=sym, horizon=horizon, force_retrain=False)
                opportunities.append({
                    "symbol": sym,
                    "name": meta["name"],
                    "display": meta.get("display", meta["name"]),
                    "price": res["current_price"],
                    "digits": res.get("digits", 2),
                    "is_forex": res.get("is_forex", False),
                    "signal": res["prediction"]["signal"],
                    "confidence": res["prediction"]["confidence"],
                    "prob_up": res["prediction"]["prob_up"],
                    "points_delta": res["expected_movement"]["points_delta"],
                    "pips_delta": res["expected_movement"]["pips_delta"],
                    "pct_delta": res["expected_movement"]["pct_delta"],
                    "action": res["trade_setup"]["action"],
                    "entry_zone": res["trade_setup"]["entry_zone"],
                    "target_1": res["trade_setup"]["target_1"],
                    "target_1_pct": res["trade_setup"]["target_1_pct"],
                    "target_1_pips": res["trade_setup"].get("target_1_pips", 0),
                    "stop_loss": res["trade_setup"]["stop_loss"],
                    "risk_reward": res["trade_setup"]["risk_reward_ratio"],
                    "score": res["prediction"]["confidence"] * (1.2 if "BULLISH" in res["prediction"]["signal"] else 0.8)
                })
            except Exception:
                continue

        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities
