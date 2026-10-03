#!/usr/bin/env bash
# Quick launcher for CryptoPredict AI Dashboard
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Check if Streamlit is already running on port 8501
if lsof -i :8501 >/dev/null 2>&1; then
    echo "⚡ CryptoPredict AI is already running! Opening your browser..."
    open "http://localhost:8501"
    exit 0
fi

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting CryptoPredict AI Dashboard on http://localhost:8501 ..."
# Open browser after a brief delay
(sleep 2 && open "http://localhost:8501") &
streamlit run app.py --server.port=8501 --server.headless=true
