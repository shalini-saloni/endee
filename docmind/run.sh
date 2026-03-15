#!/usr/bin/env bash
set -e

echo "Starting DocMind..."

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting FastAPI backend on port 8000..."
python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 &
API_PID=$!

# Give FastAPI time to start before Streamlit loads
sleep 5

echo "Starting Streamlit frontend on port $PORT..."
streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0

kill $API_PID 2>/dev/null || true