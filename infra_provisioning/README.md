# EVE Online Data Demo - Infrastructure & Pipeline

This repository contains the infrastructure-as-code and data pipeline scripts for the EVE Online "Data Cloud for Games" demo.

## Prerequisities

1.  **Google Cloud SDK** installed and authenticated (`gcloud auth application-default login`).
2.  **Python 3.8+** installed.
3.  **Dependencies**:
    ```bash
    pip install google-cloud-bigquery google-cloud-storage google-cloud-pubsub apache-beam[gcp]
    ```

## Setup Workflow

### Step 1: Data Preparation (The Notebook)
**You must run this first.**
Open `eve_online_eda_with_visualization.ipynb` in Google Colab or a local Jupyter environment.
1.  Set your `PROJECT_ID` and `BUCKET_NAME` in the configuration cell.
2.  Run all cells.
3.  **Outcome:** This will download the EVE Static Data Export (SDE), process it, and upload the following files to your GCS bucket:
    * `all_ships_vector.csv`
    * `player_ships_vector.csv`

### Step 2: Provision Infrastructure
Run the master setup script to create BigQuery datasets, Pub/Sub topics, and load the static data you just generated.

```bash
chmod +x setup_demo.sh
./setup_demo.sh <YOUR_PROJECT_ID> <YOUR_BUCKET_NAME>

