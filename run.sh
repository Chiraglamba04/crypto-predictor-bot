#!/usr/bin/env bash
# Quick launcher for CryptoPredict AI Dashboard
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting CryptoPredict AI Dashboard on http://localhost:8501 ..."
streamlit run app.py --server.port=8501 --server.headless=true
