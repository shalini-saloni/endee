#!/usr/bin/env bash
set -e

echo "Starting DocMind services..."

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting FastAPI backend..."

uvicorn docmind.app.main:app \
--host 0.0.0.0 \
--port 8000 &

API_PID=$!

sleep 3

echo "Starting Streamlit frontend..."

streamlit run docmind/streamlit_app.py \
--server.port=$PORT \
--server.address=0.0.0.0

kill $API_PID