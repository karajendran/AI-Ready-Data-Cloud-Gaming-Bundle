#!/bin/bash

# ==============================================================================
# APPROACH 2: REAL-TIME ANOMALY DETECTION DEMO RUNNER
# ==============================================================================
# Usage: ./run_demo.sh <PROJECT_ID> <STAGING_BUCKET_NAME>
# Example: ./run_demo.sh accelerated-platforms-dev accelerated-platforms-dev-dataflow-staging
# ==============================================================================

if [ "$#" -ne 2 ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: $0 <PROJECT_ID> <STAGING_BUCKET_NAME>"
    exit 1
fi

PROJECT_ID=$1
STAGING_BUCKET=$2
MODEL_BUCKET="eve-online-model-bucket"
REGION="us-central1"

# Enable strict mode to fail on any error
set -e

echo "=================================================="
echo "   🚀 LAUNCHING APPROACH 2: REAL-TIME AI AGENT   "
echo "   Project: $PROJECT_ID"
echo "   Staging: $STAGING_BUCKET"
echo "=================================================="

# --- STEP 1: TRAINING ---
echo ""
echo "[1/4] 🤖 Training Autoencoder Model..."
# We use the 'simple' version which works best in Cloud Shell/Linux
python3 02_approach_realtime_api/train_autoencoder_simple.py --project_id "$PROJECT_ID"

# --- STEP 2: DEPLOYMENT ---
echo ""
echo "[2/4] ☁️  Deploying to Vertex AI (This takes ~5-10 mins)..."
python3 02_approach_realtime_api/deploy_endpoint.py \
    --project_id "$PROJECT_ID" \
    --staging_bucket "$STAGING_BUCKET"

# --- STEP 3: SYNC CONFIGURATION ---
echo ""
echo "[3/4] 🔄 Syncing Agent Configuration to GCS..."
# The Agent needs these files in the bucket to run in the cloud/ADK
echo "   - Uploading endpoint_config.txt..."
gsutil cp endpoint_config.txt "gs://$MODEL_BUCKET/"

echo "   - Uploading stats.json..."
gsutil cp model_artifacts/stats.json "gs://$MODEL_BUCKET/"

echo "✅ Configuration synced."

# --- STEP 4: RUN AGENT ---
echo ""
echo "[4/4] 🕵️  Running Real-Time Security Agent..."
python3 02_approach_realtime_api/run_realtime_agent.py

echo ""
echo "=================================================="
echo "   🎉 DEMO COMPLETE"
echo "=================================================="

