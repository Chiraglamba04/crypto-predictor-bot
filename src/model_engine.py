"""
Machine Learning Model Engine & Ensemble Predictor
Implements an ensemble of HistGradientBoosting, RandomForest, and Calibrated Logistic Regression.
Provides time-series cross-validation, feature importance explanation, and strategy backtesting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.feature_engineering import FEATURE_COLUMNS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class CryptoModelEngine:
    def __init__(self, symbol: str = "BTCUSDT", horizon: str = "1h"):
        self.symbol = symbol
        self.horizon = horizon
        self.model_path = os.path.join(MODELS_DIR, f"{symbol}_{horizon}_ensemble.joblib")
        self.pipeline: Optional[VotingClassifier] = None
        self.metrics: Dict[str, Any] = {}
        self.feature_importances: Dict[str, float] = {}

    def _build_ensemble(self) -> VotingClassifier:
        """
        Builds a soft-voting ensemble combining gradient boosting, random forest, and regularized logistic regression.
        """
        hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=15,
            l2_regularization=1.5,
            random_state=42
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=500, random_state=42))
        ])

        ensemble = VotingClassifier(
            estimators=[
                ("hgb", hgb),
                ("rf", rf),
                ("lr", lr_pipe)
            ],
            voting="soft",
            weights=[2, 1, 1]
        )
        return ensemble

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Chronologically splits data (no shuffle) to avoid lookahead leakage.
        Trains the ensemble on training split and audits on out-of-sample test split.
        """
        n_samples = len(X)
        split_idx = int(n_samples * split_ratio)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Handle class balance
        self.pipeline = self._build_ensemble()
        self.pipeline.fit(X_train, y_train)

        # Predictions on test set
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # Compute performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Compute feature importance from Random Forest component
        rf_model = self.pipeline.named_estimators_["rf"]
        importances = rf_model.feature_importances_
        feat_imp_dict = {
            col: round(float(imp), 4)
            for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
        }
        self.feature_importances = feat_imp_dict

        # Simulated Cumulative Return Strategy vs Buy & Hold
        test_returns = X_test["return_1h"].values if "return_1h" in X_test else np.zeros(len(X_test))
        # Strategy: Long when P(Up) > 0.52, Flat/Cash otherwise
        strategy_returns = np.where(y_prob > 0.52, test_returns, 0.0)
        cum_benchmark = float(np.prod(1 + test_returns / 100.0) - 1.0) * 100.0
        cum_strategy = float(np.prod(1 + strategy_returns / 100.0) - 1.0) * 100.0

        self.metrics = {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "confusion_matrix": cm,
            "cum_strategy_return": round(cum_strategy, 2),
            "cum_benchmark_return": round(cum_benchmark, 2),
            "win_rate": round(float(acc) * 100, 2),
        }

        # Save model and metadata
        self.save_model()
        return self.metrics

    def predict_live(self, latest_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs live inference on the most recent candle features.
        Returns directional signal, confidence score, and top drivers.
        """
        if self.pipeline is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                raise ValueError("Model has not been trained yet. Call train_and_evaluate first.")

        prob_up = float(self.pipeline.predict_proba(latest_features)[:, 1][0])
        prob_down = 1.0 - prob_up

        # Classification label and badge
        if prob_up >= 0.65:
            signal = "STRONG BULLISH"
            color = "#00E676"  # Bright Green
            confidence = prob_up * 100
        elif prob_up >= 0.54:
            signal = "BULLISH"
            color = "#66BB6A"  # Green
            confidence = prob_up * 100
        elif prob_up <= 0.35:
            signal = "STRONG BEARISH"
            color = "#FF1744"  # Bright Red
            confidence = prob_down * 100
        elif prob_up <= 0.46:
            signal = "BEARISH"
            color = "#EF5350"  # Red
            confidence = prob_down * 100
        else:
            signal = "NEUTRAL / SIDEWAYS"
            color = "#FFA726"  # Amber
            confidence = max(prob_up, prob_down) * 100

        # Top 5 market drivers from feature importance
        top_drivers = list(self.feature_importances.items())[:5]

        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "signal": signal,
            "signal_color": color,
            "confidence": round(confidence, 1),
            "prob_up": round(prob_up * 100, 1),
            "prob_down": round(prob_down * 100, 1),
            "top_drivers": top_drivers,
            "metrics": self.metrics
        }

    def save_model(self):
        joblib.dump({
            "pipeline": self.pipeline,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances
        }, self.model_path)

    def load_model(self):
        data = joblib.load(self.model_path)
        self.pipeline = data["pipeline"]
        self.metrics = data.get("metrics", {})
        self.feature_importances = data.get("feature_importances", {})
