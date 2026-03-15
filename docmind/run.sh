#!/usr/bin/env bash

# DocMind startup script (Render + Local compatible)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting DocMind services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Default ports
API_PORT=8000
STREAMLIT_PORT=${PORT:-8501}

echo ""
echo "Starting FastAPI backend on port ${API_PORT}..."

uvicorn docmind.app.main:app \
--host 0.0.0.0 \
--port ${API_PORT} &
API_PID=$!

sleep 3

echo ""
echo "Starting Streamlit UI on port ${STREAMLIT_PORT}..."

streamlit run docmind/streamlit_app.py \
--server.port=${STREAMLIT_PORT} \
--server.address=0.0.0.0

# Cleanup
kill $API_PID 2>/dev/null || true