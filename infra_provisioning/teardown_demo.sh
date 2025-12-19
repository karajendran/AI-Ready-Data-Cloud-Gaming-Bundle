#!/bin/bash

# ==============================================================================
# EVE ONLINE DEMO - INFRASTRUCTURE TEARDOWN
# ==============================================================================
# WARNING: This script deletes ALL resources created for this demo.
# Usage: ./teardown_demo.sh <PROJECT_ID> <GCS_SOURCE_BUCKET_NAME>
# ==============================================================================

if [ "$#" -ne 2 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <PROJECT_ID> <GCS_SOURCE_BUCKET_NAME>"
    exit 1
fi

PROJECT_ID=$1
GCS_SOURCE_BUCKET=$2
REGION="us-central1"

# Derived Config matches setup scripts
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"
TOPIC_ID="eve-telemetry-stream"
SUBSCRIPTION_ID="eve-telemetry-sub"
DATASET_ID="eve_data_demo"
JOB_NAME="eve-telemetry-pipeline"
SA_NAME="game-dataflow-sa"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "=========================================="
echo "    ⚠️  DESTRUCTIVE ACTION  ⚠️             "
echo "    Project: $PROJECT_ID                    "
echo "    This will delete:                       "
echo "    - Dataflow Job: $JOB_NAME               "
echo "    - BigQuery Dataset: $DATASET_ID         "
echo "    - Pub/Sub Topic: $TOPIC_ID              "
echo "    - GCS Bucket: $GCS_STAGING_BUCKET       "
echo "    - Service Account: $SA_EMAIL            "
echo "    - (Optional) Source Bucket: $GCS_SOURCE_BUCKET"
echo "=========================================="
read -p "Are you sure you want to proceed? (Type 'DELETE' to confirm): " -r
echo ""
if [[ ! $REPLY == "DELETE" ]]; then
    echo "Aborted."
    exit 1
fi

echo "--- Starting Teardown ---"

# 1. Stop Dataflow Job
echo "[1/6] Stopping Dataflow Job..."
# We search for jobs that are Running, Draining, or other active states
JOB_ID=$(gcloud dataflow jobs list --project="$PROJECT_ID" \
  --filter="name=$JOB_NAME AND state=Running" \
  --format="value(id)" \
  --region="$REGION")

if [ -n "$JOB_ID" ]; then
    echo "Found running job $JOB_ID. Cancelling..."
    gcloud dataflow jobs cancel "$JOB_ID" --project="$PROJECT_ID" --region="$REGION"
    echo "Job cancelled. Note: It may take a few minutes to fully stop."
else
    echo "No running Dataflow job found."
fi

# 2. Delete Pub/Sub Subscription & Topic
echo "[2/6] Deleting Pub/Sub Resources..."
gcloud pubsub subscriptions delete "$SUBSCRIPTION_ID" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Subscription not found."
gcloud pubsub topics delete "$TOPIC_ID" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Topic not found."

# 3. Delete BigQuery Dataset (and all tables inside)
echo "[3/6] Deleting BigQuery Dataset..."
# -r: recursive (delete tables), -f: force (no prompt)
bq rm -r -f -d "${PROJECT_ID}:${DATASET_ID}" 2>/dev/null || echo "Dataset not found."

# 4. Delete Staging Bucket
echo "[4/6] Deleting Dataflow Staging Bucket..."
if gcloud storage ls "gs://${GCS_STAGING_BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud storage rm --recursive "gs://${GCS_STAGING_BUCKET}" --project="$PROJECT_ID" --quiet
    echo "Staging bucket deleted."
else
    echo "Staging bucket not found."
fi

# 5. Delete Service Account
echo "[5/6] Deleting Service Account..."
if gcloud iam service-accounts list --filter="email:$SA_EMAIL" --project="$PROJECT_ID" --format="value(email)" | grep -q "$SA_EMAIL"; then
    gcloud iam service-accounts delete "$SA_EMAIL" --project="$PROJECT_ID" --quiet
    echo "Service Account deleted."
else
    echo "Service Account not found."
fi

# 6. Delete Source Bucket (Optional - Uncomment if you want to be super destructive)
# echo "[6/6] Deleting Source Data Bucket..."
# if gcloud storage ls "gs://${GCS_SOURCE_BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
#    gcloud storage rm --recursive "gs://${GCS_SOURCE_BUCKET}" --project="$PROJECT_ID" --quiet
#    echo "Source bucket deleted."
# else
#    echo "Source bucket not found."
# fi

echo "=========================================="
echo "    TEARDOWN COMPLETE                     "
echo "=========================================="

