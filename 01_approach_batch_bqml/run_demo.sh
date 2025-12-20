#!/bin/bash

# ==============================================================================
# EVE ONLINE DEMO - RUN SHOWCASE
# ==============================================================================
# 1. Trains the K-Means Anomaly Detection Model
# 2. Launches the GenAI Security Agent (Google ADK)
# Usage: ./run_demo.sh <PROJECT_ID>
# ==============================================================================

if [ "$#" -lt 1 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <PROJECT_ID>"
    exit 1
fi

PROJECT_ID=$1

# Stop on error
set -e

echo "=========================================="
echo "    🚀 STARTING LIVE DEMO                 "
echo "    Project: $PROJECT_ID"
echo "    Agent:   Google ADK"
echo "=========================================="

# --- 1. Train Model ---
echo ""
echo "[1/2] Training Anomaly Detection Model..."
echo "      (This analyzes the behavioral vectors in BigQuery)"
if [ -f "train_model.py" ]; then
    python3 train_model.py --project_id "$PROJECT_ID"
else
    echo "❌ Error: train_model.py not found."
    exit 1
fi

# --- 2. Run Agent (ADK) ---
echo ""
echo "[2/2] Launching Security Agent (ADK)..."
echo "      (Type 'exit' to quit)"
echo "---------------------------------------------------"

# Check if ADK is installed
if ! pip show google-adk > /dev/null 2>&1; then
    echo "⚠️  Google ADK not found. Installing..."
    pip install google-adk
fi

# Check for agent directory
if [ ! -d "adk_agent" ]; then
    echo "❌ Error: 'adk_agent/' directory not found."
    exit 1
fi

# --- DYNAMIC CONFIGURATION ---
# We write the .env file dynamically to ensure it matches the CLI argument
echo "📝 Configuring adk_agent/.env for Project: $PROJECT_ID..."
cat > adk_agent/.env <<EOF
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=us-central1
EOF

# Run using the ADK CLI
# The CLI will automatically pick up the .env we just wrote
adk run adk_agent

echo ""
echo "=========================================="
echo "    ✅ DEMO COMPLETE"
echo "=========================================="

