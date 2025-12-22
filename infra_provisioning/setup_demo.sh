#!/bin/bash

# ==============================================================================
# EVE ONLINE DEMO - MASTER SETUP SCRIPT
# ==============================================================================
# Usage: ./setup_demo.sh <PROJECT_ID> <GCS_BUCKET_NAME>
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
START_TIME=$(date +%s)

# --- NETWORK CONFIGURATION ---
NETWORK_NAME="default" 
SUBNETWORK_NAME="" 

# Derived Configuration
GCS_STAGING_BUCKET="${PROJECT_ID}-dataflow-staging"
SUBSCRIPTION_ID="eve-telemetry-sub"
DATASET_ID="eve_data_demo"
FACT_TABLE_ID="fact_game_events"
SA_NAME="game-dataflow-sa"

# Stop script on any error
set -e

echo "=========================================="
echo "    EVE ONLINE DEMO - INFRA SETUP         "
echo "    Start Time: $(date)"
echo "    Project:    $PROJECT_ID"
echo "=========================================="

echo ""
echo "[$(date +%T)] [0/5] Enabling APIs & Configuring IAM..."

# 1. Enable APIs
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

echo "[$(date +%T)] ✅ APIs enabled successfully."

# 2. Create Service Account
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
if ! gcloud iam service-accounts list --filter="email:$SA_EMAIL" --format="value(email)" | grep -q "$SA_EMAIL"; then
    gcloud iam service-accounts create $SA_NAME --display-name="Game Dataflow SA"
    echo "[$(date +%T)] ✅ Service Account created."
else
    echo "[$(date +%T)] ⚠️ Service Account exists."
fi

# 3. Grant Permissions (Blindly add bindings to ensure they exist)
echo "[$(date +%T)] 🔑 Granting IAM Roles..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/dataflow.worker" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/dataflow.developer" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.dataEditor" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/bigquery.jobUser" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/pubsub.subscriber" --condition=None > /dev/null 2>&1 || true
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectAdmin" --condition=None > /dev/null 2>&1 || true

# --- FIX: Wait for IAM Propagation ---
echo "[$(date +%T)] ⏳ Sleeping 60s to allow IAM permissions to propagate..."
sleep 60
echo "[$(date +%T)] ✅ IAM Propagation likely complete."

# --- Path Handling ---
if [ -f "infra_provisioning/provision_static_data.py" ]; then
    PATH_PREFIX="infra_provisioning/"
else
    PATH_PREFIX=""
fi

# --- 3. Provision Static Data ---
echo ""
echo "[$(date +%T)] [1/5] Provisioning Static Data..."
python3 "${PATH_PREFIX}provision_static_data.py" \
    --project_id "$PROJECT_ID" \
    --region "$REGION" \
    --gcs_bucket "$GCS_BUCKET_NAME"

# --- 4. Provision Streaming Infra ---
echo ""
echo "[$(date +%T)] [2/5] Provisioning Streaming Infrastructure..."
python3 "${PATH_PREFIX}provision_streaming_infra.py" \
    --project_id "$PROJECT_ID" \
    --region "$REGION"

# --- 5. Data Generation (Step 3 - Moved UP) ---
echo ""
echo "[$(date +%T)] [3/5] Generating Training Data (History)..."
GEN_SCRIPT=""
if [ -f "generate_training_data.py" ]; then GEN_SCRIPT="generate_training_data.py"; fi
if [ -f "data_generation/generate_training_data.py" ]; then GEN_SCRIPT="data_generation/generate_training_data.py"; fi

if [ -n "$GEN_SCRIPT" ]; then
    echo "Generating 24h of synthetic history to train the model..."
    python3 "$GEN_SCRIPT" --project_id "$PROJECT_ID"
else
    echo "⚠️ generate_training_data.py not found. Skipping data generation."
fi

# --- 6. Feature Engineering (Step 4 - Moved DOWN) ---
echo ""
echo "[$(date +%T)] [4/5] Creating Feature Engineering Views..."
if [ -f "feature_engineering.py" ]; then
    python3 feature_engineering.py --project_id "$PROJECT_ID"
else
    echo "⚠️ feature_engineering.py not found. Skipping."
fi

# --- 7. Deploy Pipeline ---
echo ""
echo "[$(date +%T)] [5/5] Pipeline Deployment..."
PIPELINE_SCRIPT="pubsub_to_bigquery.py"

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "Warning: $PIPELINE_SCRIPT not found. Skipping deployment."
else
    read -p "Do you want to deploy the Dataflow Pipeline now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "[$(date +%T)] Deploying Dataflow Pipeline..."
        
        if ! gcloud storage ls gs://$GCS_STAGING_BUCKET > /dev/null 2>&1; then
            echo "🪣 Creating Staging Bucket: gs://$GCS_STAGING_BUCKET"
            gcloud storage buckets create gs://$GCS_STAGING_BUCKET --location=$REGION
        fi

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

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "    SETUP COMPLETE"
echo "    Total Time: $(($DURATION / 60))m $(($DURATION % 60))s"
echo "=========================================="

