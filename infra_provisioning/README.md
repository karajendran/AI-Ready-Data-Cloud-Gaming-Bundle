# EVE Online Data Demo - Infrastructure

This repository contains the infrastructure-as-code and data pipeline scripts for the EVE Online "Data Cloud for Games" demo. It provisions BigQuery datasets, Pub/Sub topics, and deploys a streaming Dataflow pipeline.

## Prerequisites

1.  **Google Cloud SDK** installed and authenticated (`gcloud auth application-default login`).
2.  **Python 3.8+** installed.
3.  **Dependencies** installed:
    ```bash
    pip install google-cloud-bigquery google-cloud-storage google-cloud-pubsub apache-beam[gcp]
    ```

## Usage

Make the runner script executable and run it with your Project ID and a name for your GCS bucket (which must contain your source CSVs).

```bash
chmod +x setup_demo.sh
./setup_demo.sh <YOUR_PROJECT_ID> <YOUR_BUCKET_NAME>

