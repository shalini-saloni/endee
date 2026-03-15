#!/usr/bin/env bash

# run.sh  –  DocMind one-shot startup script
# Usage:  ./run.sh

set -e

CYAN="\033[0;36m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

banner() { echo -e "${CYAN}━━━  $1  ━━━${RESET}"; }
ok()     { echo -e "${GREEN}  $1${RESET}"; }
warn()   { echo -e "${YELLOW}   $1${RESET}"; }
err()    { echo -e "${RED}  $1${RESET}"; exit 1; }

banner "DocMind – RAG-powered Document Q&A"

# Check Docker 
banner "Checking Docker"
command -v docker &>/dev/null || err "Docker not found. Install from https://docs.docker.com/get-docker/"
docker info &>/dev/null       || err "Docker daemon is not running. Please start Docker."
ok "Docker is running"

# Start Endee 
banner "Starting Endee Vector DB"
if docker ps --format '{{.Names}}' | grep -q "^endee-server$"; then
    ok "Endee is already running"
else
    docker compose up -d
    echo "Waiting for Endee to be ready..."
    for i in {1..15}; do
        if curl -sf http://localhost:8080 &>/dev/null; then
            ok "Endee is ready at http://localhost:8080"
            break
        fi
        sleep 1
        echo -n "."
    done
fi

# Python virtual environment 
banner "Setting up Python environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "Virtual environment created"
fi
source venv/bin/activate
ok "Virtual environment activated"

# Install dependencies 
banner "Installing dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt
ok "Dependencies installed"

# Check .env 
banner "Checking configuration"
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from .env.example — please add your API key!"
    warn "Edit .env and set OPENAI_API_KEY or GROQ_API_KEY, then re-run."
    exit 0
fi
ok ".env found"

# Start FastAPI backend 
banner "Starting FastAPI backend (port 8000)"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
sleep 2
curl -sf http://localhost:8000/health &>/dev/null && ok "API running at http://localhost:8000" || warn "API may still be starting..."

# Start Streamlit frontend 
banner "Starting Streamlit UI (port 8501)"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}   DocMind is running!                              ${RESET}"
echo -e "${GREEN}   Streamlit UI  →  http://localhost:8501            ${RESET}"
echo -e "${GREEN}   FastAPI docs  →  http://localhost:8000/docs       ${RESET}"
echo -e "${GREEN}   Endee dashboard → http://localhost:8080           ${RESET}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

streamlit run streamlit_app.py --server.port 8501

# Cleanup on exit
kill $API_PID 2>/dev/null || true
