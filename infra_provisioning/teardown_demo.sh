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
START_TIME=$(date +%s)

# Derived Config matches setup scripts
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"
TOPIC_ID="eve-telemetry-stream"
SUBSCRIPTION_ID="eve-telemetry-sub"
DATASET_ID="eve_data_demo"
JOB_NAME_PREFIX="eve-telemetry-pipeline" # Matches the prefix used in Python
SA_NAME="game-dataflow-sa"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "=========================================="
echo "    ⚠️  DESTRUCTIVE ACTION  ⚠️             "
echo "    Project: $PROJECT_ID                    "
echo "    This will delete:                       "
echo "    - ALL Dataflow Jobs: $JOB_NAME_PREFIX* "
echo "    - BigQuery Dataset: $DATASET_ID         "
echo "      (Includes Tables, Views, & ML Models) "
echo "    - Pub/Sub Topic: $TOPIC_ID              "
echo "    - GCS Bucket: $GCS_STAGING_BUCKET       "
echo "    - Service Account: $SA_EMAIL            "
echo "    - Source Bucket: $GCS_SOURCE_BUCKET     "
echo "=========================================="
read -p "Are you sure you want to proceed? (Type 'DELETE' to confirm): " -r
echo ""
if [[ ! $REPLY == "DELETE" ]]; then
    echo "Aborted."
    exit 1
fi

echo "--- Starting Teardown ---"

# 1. Stop Dataflow Job (UPDATED to handle multiple timestamped jobs)
echo "[$(date +%T)] [1/6] Checking for running Dataflow Jobs..."

# Fetch ALL jobs that match the prefix and are in an active state (Running, Queued, etc.)
# We filter state server-side to avoid listing terminated jobs
JOB_IDS=$(gcloud dataflow jobs list --project="$PROJECT_ID" \
  --filter="name:$JOB_NAME_PREFIX AND state=Running" \
  --format="value(id)" \
  --region="$REGION")

if [ -n "$JOB_IDS" ]; then
    # Loop through IDs (newline separated)
    echo "$JOB_IDS" | while read -r job_id; do
        echo "Found active job: $job_id. Cancelling..."
        gcloud dataflow jobs cancel "$job_id" --project="$PROJECT_ID" --region="$REGION"
    done
    echo "All matching jobs cancelled."
else
    echo "No running Dataflow jobs found."
fi

# 2. Delete Pub/Sub Subscription & Topic
echo "[$(date +%T)] [2/6] Deleting Pub/Sub Resources..."
gcloud pubsub subscriptions delete "$SUBSCRIPTION_ID" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Subscription not found."
gcloud pubsub topics delete "$TOPIC_ID" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Topic not found."

# 3. Delete BigQuery Dataset
echo "[$(date +%T)] [3/6] Deleting BigQuery Dataset..."
bq rm -r -f -d "${PROJECT_ID}:${DATASET_ID}" 2>/dev/null || echo "Dataset not found."

# 4. Delete Staging Bucket
echo "[$(date +%T)] [4/6] Deleting Dataflow Staging Bucket..."
if gcloud storage ls "gs://${GCS_STAGING_BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud storage rm --recursive "gs://${GCS_STAGING_BUCKET}" --project="$PROJECT_ID" --quiet
    echo "Staging bucket deleted."
else
    echo "Staging bucket not found."
fi

# 5. Delete Service Account
echo "[$(date +%T)] [5/6] Deleting Service Account..."
if gcloud iam service-accounts list --filter="email:$SA_EMAIL" --project="$PROJECT_ID" --format="value(email)" | grep -q "$SA_EMAIL"; then
    gcloud iam service-accounts delete "$SA_EMAIL" --project="$PROJECT_ID" --quiet
    echo "Service Account deleted."
else
    echo "Service Account not found."
fi

# 6. Delete Source Bucket (Optional - Uncomment if you want to be super destructive)
# echo "[$(date +%T)] [6/6] Deleting Source Data Bucket..."
# if gcloud storage ls "gs://${GCS_SOURCE_BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
#    gcloud storage rm --recursive "gs://${GCS_SOURCE_BUCKET}" --project="$PROJECT_ID" --quiet
#    echo "Source bucket deleted."
# else
#    echo "Source bucket not found."
# fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "    TEARDOWN COMPLETE                     "
echo "    Total Time: $(($DURATION / 60))m $(($DURATION % 60))s"
echo "=========================================="

