"""
Live Predictor Pipeline
High-level orchestrator that connects data fetching, feature engineering, model inference,
and visual indicator preparation for real-time cryptocurrency forecasts.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.data_fetcher import DataFetcher, CRYPTO_PAIRS
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
        Runs the full predictive pipeline:
        1. Multi-source data ingestion (Candles, F&G, News RSS)
        2. Technical indicator math & Feature synthesis
        3. Walk-forward ensemble model training or inference
        4. Target price range estimation based on volatility
        5. Historical signals generation for chart overlay
        """
        cache_key = f"{symbol}_{horizon}"
        now = time.time()
        
        # 1. Fetch market data
        interval = "1h" if horizon == "1h" else "4h"
        raw_df = self.fetcher.fetch_binance_klines(symbol=symbol, interval="1h", limit=500)
        ticker_info = self.fetcher.fetch_24h_ticker(symbol=symbol)
        sentiment_summary = self.fetcher.get_market_sentiment_summary()

        # 2. Extract sentiment metrics
        fg_score = sentiment_summary["fear_and_greed_value"]
        news_sentiment = sentiment_summary["avg_news_sentiment"]

        # 3. Engineer features
        fe = FeatureEngineer(fg_score=fg_score, news_sentiment=news_sentiment)
        df_featured = fe.create_features(raw_df, fg_score=fg_score, news_sentiment=news_sentiment)
        
        # 4. Prepare dataset
        X, y, latest_features = fe.prepare_training_dataset(df_featured, horizon=horizon)

        # 5. Model Engine
        engine = CryptoModelEngine(symbol=symbol, horizon=horizon)
        
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

        # 7. Price Target Estimation (based on ATR)
        current_price = float(df_featured["close"].iloc[-1])
        current_atr = float(df_featured["atr_14"].iloc[-1])
        multiplier = 1.0 if horizon == "1h" else 2.2
        
        expected_high = current_price + (current_atr * multiplier)
        expected_low = current_price - (current_atr * multiplier)

        if "BULLISH" in prediction["signal"]:
            primary_target = current_price + (current_atr * multiplier * 0.8)
            stop_loss = current_price - (current_atr * 1.2)
        elif "BEARISH" in prediction["signal"]:
            primary_target = current_price - (current_atr * multiplier * 0.8)
            stop_loss = current_price + (current_atr * 1.2)
        else:
            primary_target = current_price
            stop_loss = current_price - (current_atr * 1.0)

        # 8. Historical signals on the last 50 candles for charting
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

        # 9. Technical Summary Dashboard
        rsi_val = float(df_featured["rsi_14"].iloc[-1])
        macd_val = float(df_featured["macd"].iloc[-1])
        macd_sig_val = float(df_featured["macd_signal"].iloc[-1])
        ema_9_val = float(df_featured["ema_9"].iloc[-1])
        ema_21_val = float(df_featured["ema_21"].iloc[-1])

        tech_summary = {
            "rsi": round(rsi_val, 1),
            "rsi_status": "Oversold (Bullish Reversal)" if rsi_val < 30 else ("Overbought (Bearish Pullback)" if rsi_val > 70 else "Neutral Momentum"),
            "macd_status": "Bullish Crossover" if macd_val > macd_sig_val else "Bearish Momentum",
            "trend_status": "Short-Term Uptrend (EMA 9 > EMA 21)" if ema_9_val > ema_21_val else "Short-Term Downtrend (EMA 9 < EMA 21)",
            "volatility_status": "High Volatility" if float(df_featured["bb_bandwidth"].iloc[-1]) > 5.0 else "Consolidating / Low Volatility",
        }

        result = {
            "symbol": symbol,
            "symbol_name": CRYPTO_PAIRS.get(symbol, {}).get("name", symbol),
            "horizon": horizon,
            "current_price": current_price,
            "ticker_info": ticker_info,
            "prediction": prediction,
            "sentiment": sentiment_summary,
            "tech_summary": tech_summary,
            "price_targets": {
                "expected_high": round(expected_high, 2),
                "expected_low": round(expected_low, 2),
                "primary_target": round(primary_target, 2),
                "stop_loss": round(stop_loss, 2),
                "current_atr": round(current_atr, 2),
            },
            "metrics": metrics,
            "signals_history": signals_history,
            "chart_data": df_featured.tail(120),
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        return result
