#!/bin/bash

# ==============================================================================
# EVE ONLINE DEMO - MASTER SETUP SCRIPT
# ==============================================================================
# Usage: ./setup_demo.sh <PROJECT_ID> <GCS_BUCKET_NAME>
# Example: ./setup_demo.sh accelerated-platforms-dev my-sde-bucket
# ==============================================================================

# --- 1. Validation ---
if [ "$#" -ne 2 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <PROJECT_ID> <GCS_BUCKET_NAME>"
    exit 1
fi

PROJECT_ID=$1
GCS_BUCKET_NAME=$2
REGION="us-central1"

# --- NETWORK CONFIGURATION (UPDATE THESE IF NEEDED) ---
NETWORK_NAME="default" 
SUBNETWORK_NAME="" # Leave empty if not using a specific subnet (e.g., regions/us-central1/subnetworks/my-subnet)

# Derived Configuration
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"
SUBSCRIPTION_ID="eve-telemetry-sub"
DATASET_ID="eve_data_demo"
FACT_TABLE_ID="fact_game_events"

# Stop script on any error
set -e

echo "=========================================="
echo "   EVE ONLINE DEMO - INFRA SETUP        "
echo "   Project: $PROJECT_ID                 "
echo "   Bucket:  $GCS_BUCKET_NAME            "
echo "   Network: ${NETWORK_NAME:-Default}    "
echo "=========================================="

# --- 2. Check File Locations ---
if [ -f "infra_provisioning/provision_static_data.py" ]; then
    PATH_PREFIX="infra_provisioning/"
else
    PATH_PREFIX=""
fi

# --- 3. Provision Static Data ---
echo ""
echo "[1/3] Provisioning Static Data..."
python3 "${PATH_PREFIX}provision_static_data.py" \
    --project_id "$PROJECT_ID" \
    --region "$REGION" \
    --gcs_bucket "$GCS_BUCKET_NAME"

# --- 4. Provision Streaming Infra ---
echo ""
echo "[2/3] Provisioning Streaming Infrastructure..."
python3 "${PATH_PREFIX}provision_streaming_infra.py" \
    --project_id "$PROJECT_ID" \
    --region "$REGION"

# --- 5. Deploy Pipeline ---
echo ""
echo "[3/3] Pipeline Deployment..."
PIPELINE_SCRIPT="pubsub_to_bigquery.py"

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "Warning: $PIPELINE_SCRIPT not found in current directory. Skipping deployment."
else
    read -p "Do you want to deploy the Dataflow Pipeline now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Deploying Dataflow Pipeline..."
        
        # We pass the network args. If variables are empty string "", 
        # the Python script's logic (if args.network:) will correctly ignore them.
        python3 "$PIPELINE_SCRIPT" \
            --project_id "$PROJECT_ID" \
            --subscription_id "$SUBSCRIPTION_ID" \
            --dataset_id "$DATASET_ID" \
            --table_id "$FACT_TABLE_ID" \
            --gcs_staging_bucket "$GCS_STAGING_BUCKET" \
            --region "$REGION" \
            --network "$NETWORK_NAME" \
            --subnetwork "$SUBNETWORK_NAME"
    else
        echo "Skipping pipeline deployment."
    fi
fi

echo ""
echo "=========================================="
echo "   SETUP COMPLETE                         "
echo "=========================================="

