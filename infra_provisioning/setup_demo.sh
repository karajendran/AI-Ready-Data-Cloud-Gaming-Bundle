#!/bin/bash

# --- CONFIGURATION ---
# UPDATE THESE VALUES
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
GCS_BUCKET_NAME="eve-demo-static-data-${PROJECT_ID}" # Unique bucket name
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"

# Pub/Sub Config
TOPIC_ID="eve-telemetry-stream"
SUBSCRIPTION_ID="eve-telemetry-sub"

# BigQuery Config
DATASET_ID="eve_data_demo"
FACT_TABLE_ID="fact_game_events"

echo "=========================================="
echo "   EVE ONLINE DEMO - INFRA SETUP        "
echo "=========================================="

# 1. Provision Static Data (Dimensions)
echo "[1/3] Provisioning Static Data..."
python3 provision_static_data.py \
    --project_id "$PROJECT_ID" \
    --region "$REGION" \
    --gcs_bucket "$GCS_BUCKET_NAME"

# 2. Provision Streaming Infra (Topic, Sub, Fact Table)
echo "[2/3] Provisioning Streaming Infrastructure..."
python3 provision_streaming_infra.py \
    --project_id "$PROJECT_ID" \
    --region "$REGION"

# 3. Deploy Dataflow Pipeline
# Note: We ask the user if they want to deploy the pipeline now, as it costs money while running.
echo ""
read -p "[3/3] Do you want to deploy the Dataflow Pipeline now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Deploying Dataflow Pipeline..."
    python3 pubsub_to_bigquery.py \
        --project_id "$PROJECT_ID" \
        --subscription_id "$SUBSCRIPTION_ID" \
        --dataset_id "$DATASET_ID" \
        --table_id "$FACT_TABLE_ID" \
        --gcs_staging_bucket "$GCS_STAGING_BUCKET" \
        --region "$REGION"
else
    echo "Skipping pipeline deployment."
fi

echo "=========================================="
echo "   SETUP COMPLETE                         "
echo "=========================================="
