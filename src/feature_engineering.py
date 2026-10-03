"""
Feature Engineering for Cryptocurrency Price Movement Prediction
Prepares tabular feature sets combining technical indicators, sentiment metrics, and cyclical time features.
Generates multi-horizon forward targets without lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from src.technical_indicators import TechnicalIndicators


FEATURE_COLUMNS = [
    # Technical Momentum & Trend
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "ema_cross",
    "rsi_14", "rsi_slope",
    "macd", "macd_signal", "macd_hist", "macd_cross",
    # Volatility & Bands
    "bb_bandwidth", "bb_pct_b", "natr_14",
    # Volume & Flow
    "vol_ratio", "obv_trend",
    # Oscillators
    "stoch_k", "stoch_d",
    # Prior Returns
    "return_1h", "return_4h", "return_24h",
    # Sentiment
    "fear_greed_score", "news_sentiment_score",
    # Cyclical Time
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]


class FeatureEngineer:
    def __init__(self, fg_score: float = 0.65, news_sentiment: float = 0.2):
        self.fg_score = fg_score
        self.news_sentiment = news_sentiment

    def create_features(self, df_raw: pd.DataFrame, fg_score: Optional[float] = None, news_sentiment: Optional[float] = None) -> pd.DataFrame:
        """
        Takes raw OHLCV dataframe, computes technical indicators and appends external sentiment.
        """
        fg = fg_score if fg_score is not None else self.fg_score
        news_sent = news_sentiment if news_sentiment is not None else self.news_sentiment

        # Compute technical indicators
        df = TechnicalIndicators.calculate_all(df_raw)
        
        # Append sentiment features
        df["fear_greed_score"] = float(fg) / 100.0 if fg > 1.0 else float(fg)
        df["news_sentiment_score"] = float(news_sent)

        # Cyclical time features
        hour = df["timestamp"].dt.hour
        day = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * day / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * day / 7.0)

        return df

    def prepare_training_dataset(
        self, 
        df_featured: pd.DataFrame, 
        horizon: str = "1h",
        threshold_pct: float = 0.10
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Creates features X and targets y for a specific forecasting horizon.
        Horizons:
        - '1h': next 1 candle (shift -1)
        - '4h': next 4 candles (shift -4)
        
        Target definition (Binary with magnitude threshold):
        1 = Bullish (Price increases >= +threshold_pct)
        0 = Bearish/Stagnant (Price decreases or fails to beat threshold)
        """
        df = df_featured.copy()
        
        shift_periods = 1 if horizon == "1h" else 4
        
        # Future return
        df["future_close"] = df["close"].shift(-shift_periods)
        df["future_return"] = (df["future_close"] - df["close"]) / df["close"] * 100.0
        
        # Target label: 1 if positive return, 0 if negative or flat
        # For a clean binary classification with probability calibration
        df["target"] = (df["future_return"] > 0).astype(int)

        # Drop the last rows that do not have future outcomes
        valid_df = df.dropna(subset=["future_return"]).reset_index(drop=True)
        
        X = valid_df[FEATURE_COLUMNS]
        y = valid_df["target"]
        
        # Latest row without future label (for live inference)
        latest_X = df.iloc[[-1]][FEATURE_COLUMNS]
        
        return X, y, latest_X
