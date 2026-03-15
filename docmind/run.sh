#!/usr/bin/env bash
set -e

echo "Starting DocMind..."

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting FastAPI backend on port 8000..."
python -m uvicorn docmind.app.main:app \
    --host 0.0.0.0 \
    --port 8000 &
API_PID=$!

sleep 5

echo "Starting Streamlit frontend on port $PORT..."
streamlit run docmind/streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0

kill $API_PID 2>/dev/null || true