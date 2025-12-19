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
SUBNETWORK_NAME="" 

# Derived Configuration
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"
SUBSCRIPTION_ID="eve-telemetry-sub"
DATASET_ID="eve_data_demo"
FACT_TABLE_ID="fact_game_events"
SA_NAME="game-dataflow-sa" # Service Account Name

# Stop script on any error
set -e

echo "=========================================="
echo "    EVE ONLINE DEMO - INFRA SETUP         "
echo "    Project: $PROJECT_ID                  "
echo "    Bucket:  $GCS_BUCKET_NAME             "
echo "=========================================="

echo ""
echo "[0/3] Enabling APIs & Configuring IAM..."

# 1. Enable ALL required APIs (Supersedes the checks in Python scripts)
gcloud config set project $PROJECT_ID
gcloud services enable \
    dataflow.googleapis.com \
    compute.googleapis.com \
    logging.googleapis.com \
    storage-component.googleapis.com \
    bigquery.googleapis.com \
    pubsub.googleapis.com \
    aiplatform.googleapis.com \
    iam.googleapis.com

# 2. Create Service Account for Dataflow (Security Best Practice)
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if ! gcloud iam service-accounts list --filter="email:$SA_EMAIL" --format="value(email)" | grep -q "$SA_EMAIL"; then
    gcloud iam service-accounts create $SA_NAME \
        --display-name="Game Dataflow SA"
    echo "✅ Service Account created: $SA_EMAIL"
else
    echo "⚠️ Service Account exists. Skipping creation."
fi

# 3. Grant Permissions
echo "🔑 Granting IAM Roles..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/dataflow.worker" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/dataflow.developer" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.dataEditor" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.jobUser" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/pubsub.subscriber" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectAdmin" --condition=None > /dev/null 2>&1 || true

# --- Check File Locations ---
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
        
        # Ensure Staging Bucket Exists (Dataflow needs this before running)
        if ! gcloud storage ls gs://$GCS_STAGING_BUCKET > /dev/null 2>&1; then
            echo "🪣 Creating Staging Bucket: gs://$GCS_STAGING_BUCKET"
            gcloud storage buckets create gs://$GCS_STAGING_BUCKET --location=$REGION
        fi

        # Pass the created Service Account to the pipeline
        python3 "$PIPELINE_SCRIPT" \
            --project_id "$PROJECT_ID" \
            --subscription_id "$SUBSCRIPTION_ID" \
            --dataset_id "$DATASET_ID" \
            --table_id "$FACT_TABLE_ID" \
            --gcs_staging_bucket "$GCS_STAGING_BUCKET" \
            --region "$REGION" \
            --network "$NETWORK_NAME" \
            --subnetwork "$SUBNETWORK_NAME" \
            --service_account_email "$SA_EMAIL" 
    else
        echo "Skipping pipeline deployment."
    fi
fi

echo ""
echo "=========================================="
echo "    SETUP COMPLETE                        "
echo "=========================================="

