#!/bin/bash

# ==============================================================================
# EVE ONLINE DEMO - INFRASTRUCTURE TEARDOWN
# ==============================================================================
# WARNING: This script deletes ALL resources created for this demo.
# Usage: ./teardown_demo.sh <PROJECT_ID> <GCS_SOURCE_BUCKET_NAME>
# ==============================================================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <PROJECT_ID> <GCS_SOURCE_BUCKET_NAME>"
    exit 1
fi

PROJECT_ID=$1
GCS_SOURCE_BUCKET=$2

# Derived Config matches setup scripts
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"
TOPIC_ID="eve-telemetry-stream"
SUBSCRIPTION_ID="eve-telemetry-sub"
DATASET_ID="eve_data_demo"
JOB_NAME="eve-telemetry-pipeline"

echo "=========================================="
echo "   ⚠️  DESTRUCTIVE ACTION  ⚠️            "
echo "   Project: $PROJECT_ID                   "
echo "   This will delete:                      "
echo "   - Dataflow Job: $JOB_NAME              "
echo "   - BigQuery Dataset: $DATASET_ID        "
echo "   - Pub/Sub Topic: $TOPIC_ID             "
echo "   - GCS Bucket: $GCS_STAGING_BUCKET      "
echo "   - GCS Bucket: $GCS_SOURCE_BUCKET       "
echo "=========================================="
read -p "Are you sure you want to proceed? (Type 'DELETE' to confirm): " -r
echo ""
if [[ ! $REPLY == "DELETE" ]]; then
    echo "Aborted."
    exit 1
fi

echo "--- Starting Teardown ---"

# 1. Stop Dataflow Job
echo "[1/5] Stopping Dataflow Job..."
JOB_ID=$(gcloud dataflow jobs list --project="$PROJECT_ID" \
  --filter="name=$JOB_NAME AND state=Running" \
  --format="value(id)" \
  --region="us-central1")

if [ -n "$JOB_ID" ]; then
    echo "Found running job $JOB_ID. Cancelling..."
    gcloud dataflow jobs cancel "$JOB_ID" --project="$PROJECT_ID" --region="us-central1"
    echo "Job cancelled."
else
    echo "No running Dataflow job found."
fi

# 2. Delete Pub/Sub Subscription & Topic
echo "[2/5] Deleting Pub/Sub Resources..."
gcloud pubsub subscriptions delete "$SUBSCRIPTION_ID" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Subscription not found."
gcloud pubsub topics delete "$TOPIC_ID" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Topic not found."

# 3. Delete BigQuery Dataset (and all tables inside)
echo "[3/5] Deleting BigQuery Dataset..."
bq rm -r -f -d "${PROJECT_ID}:${DATASET_ID}" 2>/dev/null || echo "Dataset not found."

# 4. Delete Staging Bucket
echo "[4/5] Deleting Dataflow Staging Bucket..."
gcloud storage rm --recursive "gs://${GCS_STAGING_BUCKET}" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Staging bucket not found."

# 5. Delete Source Bucket (Optional Check)
# echo "[5/5] Deleting Source Data Bucket..."
# gcloud storage rm --recursive "gs://${GCS_SOURCE_BUCKET}" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Source bucket not found."

echo "=========================================="
echo "   TEARDOWN COMPLETE                      "
echo "=========================================="

