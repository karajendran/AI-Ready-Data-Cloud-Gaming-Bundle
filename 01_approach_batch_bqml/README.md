# 🎮 Run Demo: Game Health & Anomaly Detection

This guide explains how to execute the live showcase using the `run_demo.sh` script. This script ties together the Machine Learning and GenAI components of the **Data Cloud for Games** solution.

## Prerequisites

Before running this demo, ensure you have completed the infrastructure setup:

1. **Infrastructure Provisioned:** You must have run `./setup_demo.sh` to create the BigQuery datasets, generate synthetic data, and configure IAM.
2. **Authentication:** Ensure you are authenticated with Google Cloud:

```bash
gcloud auth application-default login
```

## How to Run
Execute the master script with your Google Cloud Project ID:

```bash
chmod +x run_demo.sh
./run_demo.sh <YOUR_PROJECT_ID>
```
Example:

```bash 
./run_demo.sh accelerated-platforms-dev
```

## What Happens Next?
The script performs two major actions sequentially:

**Phase 1: Model Training (train_model.py)**
* **Action**: Triggers a BigQuery ML job to train a K-Means Clustering model on the behavioral vectors (stats_per_minute view).
* **Output:** It will display a "Cluster Analysis" table in your terminal.
* **What to look for:** Identifying the "Smoking Gun" cluster. You should see one cluster (likely Cluster 2 or 3) with an abnormally high velocity (e.g., 2,500+ APM) compared to normal players (~1-5 APM).

**Phase 2: The Security Agent (game_agent_native.py)**

* **Action:** Initializes a Vertex AI Agent powered by Gemini 2.5 Flash.
* **Interaction:** The script enters an interactive loop where the Agent answers pre-canned questions (or you can modify the script to accept user input).
* **Demo Flow:**
    1. "Did we catch anyone?" -> Agent queries BigQuery for players exceeding the threshold.
    2. "Why were they flagged?" -> Agent queries the Model Centroids to explain the statistical deviation (e.g., "Player APM is 40x higher than the industry norm").

## ⚠️ Troubleshooting
* **"Table not found":** Ensure setup_demo.sh completed successfully and the fact_game_events table exists in BigQuery.
* **"Quota exceeded":** If the Dataflow pipeline (from setup) is still catching up, you might have partial data. Wait 1-2 minutes and try again.

