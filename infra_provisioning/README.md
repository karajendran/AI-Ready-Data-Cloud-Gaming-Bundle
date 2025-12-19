# EVE Online Data Demo - Infrastructure & Pipeline

This repository contains the infrastructure-as-code, data pipeline, and simulation scripts for the EVE Online "Data Cloud for Games" demo. It demonstrates:
1.  **Ingestion:** Streaming game telemetry via Pub/Sub to BigQuery (Dataflow).
2.  **Warehousing:** Storing processed game events and static dimension tables.
3.  **Analytics:** Detecting "Economic Exploits" (Market Crashes & Infinite Crafting) using statistical baselines.

## 📋 Prerequisites

1.  **Google Cloud SDK** installed and authenticated (`gcloud auth application-default login`).
2.  **Python 3.8+** installed.
3.  **Dependencies**:
    ```bash
    pip install google-cloud-bigquery google-cloud-storage google-cloud-pubsub apache-beam[gcp]
    ```

---

## Setup Workflow

### Step 1: Data Preparation (The Notebook)
**You must run this first.**
Open `eve_online_eda_with_visualization.ipynb` in Google Colab or a local Jupyter environment.
1.  Set your `PROJECT_ID` and `BUCKET_NAME` in the configuration cell.
2.  Run all cells.
3.  **Outcome:** This downloads the EVE Static Data Export (SDE) and uploads the feature vector CSVs (`all_ships_vector.csv`, `player_ships_vector.csv`) to your GCS bucket.

### Step 2: Provision Infrastructure
Run the master setup script to create BigQuery datasets, Pub/Sub topics, and deploy the pipeline.

**Important:** Before running, open `setup_demo.sh` in a text editor and check the **Network Configuration** section (lines 20-24).
* If your project requires a specific VPC (e.g., `dataflow-network`), update `NETWORK_NAME` there.
* If you are using the default network, you can leave it as is.

```bash
chmod +x setup_demo.sh
./setup_demo.sh <YOUR_PROJECT_ID> <YOUR_BUCKET_NAME>
```

---

## Teardown

To delete all resources created by this demo (Dataflow job, BigQuery dataset, Pub/Sub topics, and GCS buckets) to avoid future costs:
```bash
chmod +x teardown_demo.sh
./teardown_demo.sh <YOUR_PROJECT_ID> <YOUR_BUCKET_NAME>
```

