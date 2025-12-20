#!/bin/bash

# ==============================================================================
# EVE ONLINE DEMO - RUN SHOWCASE
# ==============================================================================
# 1. Trains the K-Means Anomaly Detection Model
# 2. Launches the GenAI Security Agent
# Usage: ./run_demo.sh <PROJECT_ID>
# ==============================================================================

if [ "$#" -ne 1 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <PROJECT_ID>"
    exit 1
fi

PROJECT_ID=$1

START_TIME=$(date +%s)

# Stop on error
set -e

echo "=========================================="
echo "    🚀 STARTING LIVE DEMO                 "
echo "    Project: $PROJECT_ID"
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

# --- 2. Run Agent ---
echo ""
echo "[2/2] Launching Vertex AI Security Agent..."
echo "      (Type your questions in the prompt below)"
echo "---------------------------------------------------"
if [ -f "game_security_agent.py" ]; then
    # We use python3 -u to unbuffer output so you see logs instantly
    python3 -u game_security_agent.py --project_id "$PROJECT_ID"
else
    echo "❌ Error: game_agent_native.py not found."
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================="
echo "    ✅ DEMO COMPLETE                                     "
echo "    Total Time: $(($DURATION / 60))m $(($DURATION % 60))s"
echo "========================================================="


