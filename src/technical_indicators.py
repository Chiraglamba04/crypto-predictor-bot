"""
Technical Indicators Calculation Engine
Provides vectorized implementations of essential technical indicators for crypto price action.
All calculations are pure pandas/numpy with no external C-dependencies.
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate complete suite of technical indicators on an OHLCV dataframe.
        Expects columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        data = df.copy()
        data = data.sort_values("timestamp").reset_index(drop=True)

        # 1. Moving Averages (EMA & SMA)
        data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
        data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
        data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
        data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
        
        # Price distance from EMAs (%)
        data["dist_ema_9"] = (data["close"] - data["ema_9"]) / data["ema_9"] * 100
        data["dist_ema_21"] = (data["close"] - data["ema_21"]) / data["ema_21"] * 100
        data["dist_ema_50"] = (data["close"] - data["ema_50"]) / data["ema_50"] * 100
        
        # EMA crossover signal (9 over 21)
        data["ema_cross"] = np.where(data["ema_9"] > data["ema_21"], 1, -1)

        # 2. RSI (Relative Strength Index, 14)
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        data["rsi_14"] = 100 - (100 / (1 + rs))

        # RSI Momentum (Slope of RSI over 3 periods)
        data["rsi_slope"] = data["rsi_14"].diff(3)

        # 3. MACD (Moving Average Convergence Divergence: 12, 26, 9)
        ema_12 = data["close"].ewm(span=12, adjust=False).mean()
        ema_26 = data["close"].ewm(span=26, adjust=False).mean()
        data["macd"] = ema_12 - ema_26
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        data["macd_cross"] = np.where(data["macd"] > data["macd_signal"], 1, -1)

        # 4. Bollinger Bands (20 periods, 2 std dev)
        data["bb_mid"] = data["close"].rolling(window=20).mean()
        data["bb_std"] = data["close"].rolling(window=20).std()
        data["bb_upper"] = data["bb_mid"] + (data["bb_std"] * 2)
        data["bb_lower"] = data["bb_mid"] - (data["bb_std"] * 2)
        
        # Bandwidth & %B
        data["bb_bandwidth"] = (data["bb_upper"] - data["bb_lower"]) / (data["bb_mid"] + 1e-10) * 100
        data["bb_pct_b"] = (data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"] + 1e-10)

        # 5. Average True Range (ATR 14) & Normalized ATR
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift(1)).abs()
        low_close = (data["low"] - data["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr_14"] = true_range.rolling(window=14).mean()
        data["natr_14"] = (data["atr_14"] / data["close"]) * 100

        # 6. Volume Indicators
        data["vol_sma_20"] = data["volume"].rolling(window=20).mean()
        data["vol_ratio"] = data["volume"] / (data["vol_sma_20"] + 1e-10)
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(data)):
            if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                obv.append(obv[-1] + data["volume"].iloc[i])
            elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
                obv.append(obv[-1] - data["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        data["obv"] = obv
        data["obv_ema"] = pd.Series(obv, index=data.index).ewm(span=20, adjust=False).mean()
        data["obv_trend"] = np.where(data["obv"] > data["obv_ema"], 1, -1)

        # 7. Stochastic Oscillator (%K, %D)
        low_14 = data["low"].rolling(window=14).min()
        high_14 = data["high"].rolling(window=14).max()
        data["stoch_k"] = ((data["close"] - low_14) / (high_14 - low_14 + 1e-10)) * 100
        data["stoch_d"] = data["stoch_k"].rolling(window=3).mean()

        # 8. Returns & Price Changes
        data["return_1h"] = data["close"].pct_change(1) * 100
        data["return_4h"] = data["close"].pct_change(4) * 100
        data["return_24h"] = data["close"].pct_change(24) * 100

        # Drop warmup NaNs
        return data.dropna().reset_index(drop=True)
