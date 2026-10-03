"""
Background Auto-Updater Daemon
Periodically fetches latest market candles, updates sentiment metrics,
retrains or infers new forecasts, and saves live states to a local JSON cache.
"""

import time
import json
import logging
import os
from datetime import datetime
from src.live_predictor import LivePredictor, CRYPTO_PAIRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CryptoUpdater")

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "live_cache.json")


def run_update_cycle(symbols=None, horizons=None):
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    if horizons is None:
        horizons = ["1h", "4h"]

    predictor = LivePredictor()
    state = {}

    for sym in symbols:
        state[sym] = {}
        for hz in horizons:
            try:
                logger.info(f"Updating predictions for {sym} ({hz})...")
                res = predictor.analyze_and_predict(symbol=sym, horizon=hz, force_retrain=False)
                # Keep lightweight serializable summary
                state[sym][hz] = {
                    "symbol": sym,
                    "symbol_name": res["symbol_name"],
                    "horizon": hz,
                    "current_price": res["current_price"],
                    "signal": res["prediction"]["signal"],
                    "signal_color": res["prediction"]["signal_color"],
                    "confidence": res["prediction"]["confidence"],
                    "prob_up": res["prediction"]["prob_up"],
                    "prob_down": res["prediction"]["prob_down"],
                    "primary_target": res["price_targets"]["primary_target"],
                    "stop_loss": res["price_targets"]["stop_loss"],
                    "tech_summary": res["tech_summary"],
                    "fear_and_greed": res["sentiment"]["fear_and_greed_value"],
                    "fear_and_greed_label": res["sentiment"]["fear_and_greed_label"],
                    "last_updated": res["last_updated"]
                }
            except Exception as e:
                logger.error(f"Error updating {sym} {hz}: {e}")

    # Write atomic JSON update
    tmp_file = CACHE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_file, CACHE_FILE)
    logger.info(f"Successfully cached predictions for {len(symbols)} pairs.")


def start_loop(interval_seconds: int = 300):
    logger.info(f"Starting continuous updater loop every {interval_seconds} seconds...")
    while True:
        try:
            run_update_cycle()
        except Exception as e:
            logger.error(f"Cycle exception: {e}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    start_loop(interval_seconds=300)
