#!/usr/bin/env python

import argparse
from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud import storage # Needed for exceptions
import subprocess

# --- Configuration ---
BIGQUERY_DATASET_ID = "eve_data_demo"

# Constants for the "All Ships" (raw) vector
GCS_ALL_SHIPS_CSV = "all_ships_vector.csv"
ALL_SHIPS_TABLE_ID = "dim_all_ships_features"

# Constants for the "Player Ships" (clean) vector
GCS_PLAYER_SHIPS_CSV = "player_ships_vector.csv"
PLAYER_SHIPS_TABLE_ID = "dim_player_ships_features"

def enable_apis(project_id):
    """
    Automatically enables necessary APIs using gcloud.
    """
    services = [
        "bigquery.googleapis.com",
        "storage-component.googleapis.com"
    ]
    
    print(f"--- Enabling APIs: {', '.join(services)} ---")
    try:
        subprocess.check_call([
            "gcloud", "services", "enable", *services, f"--project={project_id}"
        ])
        print("APIs enabled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error enabling APIs. Please ensure you are logged in (gcloud auth login). Error: {e}")
        exit(1)

def create_bigquery_dataset(project_id, dataset_id, location):
    """Creates a BigQuery dataset if it doesn't exist."""
    print(f"Attempting to create BigQuery dataset: {dataset_id}...")
    bq_client = bigquery.Client(project=project_id)
    dataset_ref = bq_client.dataset(dataset_id)
    
    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        bq_client.create_dataset(dataset, timeout=30)
        print(f"Successfully created BigQuery dataset: {dataset_id}")
    except exceptions.Conflict:
        print(f"BigQuery dataset '{dataset_id}' already exists.")
    except Exception as e:
        print(f"Error creating BigQuery dataset: {e}")
        return None
    return dataset_ref

def load_table_from_gcs(bq_client, dataset_ref, table_id, gcs_uri):
    """
    Loads data from a GCS CSV file into a BQ table.
    It uses autodetect for the schema and overwrites the table.
    """
    print(f"Attempting to load data into table: {table_id} from {gcs_uri}...")
    
    table_ref = dataset_ref.table(table_id)
    
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.autodetect = True  # Automatically detect schema
    job_config.skip_leading_rows = 1  # Skip the CSV header row
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    try:
        load_job = bq_client.load_table_from_uri(
            gcs_uri, table_ref, job_config=job_config
        )
        print(f"Starting load job {load_job.job_id} for table {table_id}...")
        
        load_job.result()  # Wait for the job to complete
        
        print(f"Successfully loaded data into table: {table_id}")
    
    except exceptions.NotFound:
        print(f"WARNING: GCS file not found: {gcs_uri}.")
        print(f"Table '{table_id}' was not created or updated.")
        print("Please run the analysis notebook to generate this CSV first.")
    except Exception as e:
        print(f"Error loading table {table_id} from GCS: {e}")

def main(project_id, region, gcs_bucket):

    # --- Feature Vector / Static Data Resources ---
    GCS_STATIC_DATA_BUCKET = gcs_bucket

    # Step 1: Enable APIs (manual step)
    enable_apis(project_id)
    
    # Step 2: Create BigQuery Dataset
    bq_client = bigquery.Client(project=project_id)
    dataset_ref = create_bigquery_dataset(project_id, BIGQUERY_DATASET_ID, region)
    
    if dataset_ref:
        # Step 3a: LOAD the "All Ships" (raw) feature table from GCS
        GCS_ALL_SHIPS_URI = f"gs://{GCS_STATIC_DATA_BUCKET}/{GCS_ALL_SHIPS_CSV}"
        load_table_from_gcs(bq_client,
                            dataset_ref,
                            ALL_SHIPS_TABLE_ID,
                            GCS_ALL_SHIPS_URI)
        
        # Step 3b: LOAD the "Player Ships" (clean) feature table from GCS
        GCS_PLAYER_SHIPS_URI = f"gs://{GCS_STATIC_DATA_BUCKET}/{GCS_PLAYER_SHIPS_CSV}"
        load_table_from_gcs(bq_client,
                            dataset_ref,
                            PLAYER_SHIPS_TABLE_ID,
                            GCS_PLAYER_SHIPS_URI)

    print("\n--- Static Data Provisioning Complete ---")
    print(f"BigQuery Dataset: {BIGQUERY_DATASET_ID}")
    print(f"BigQuery All Ships Table: {ALL_SHIPS_TABLE_ID}")
    print(f"BigQuery Player Ships Table: {PLAYER_SHIPS_TABLE_ID}")

if __name__ == "__main__":
    # pip install google-api-python-client google-cloud-bigquery
    # gcloud auth application-default login
    
    parser = argparse.ArgumentParser(
        description="Provision Static Data (Dimensions) for the EVE Demo."
    )
    parser.add_argument(
        "--project_id",
        required=True, # Force the user to provide this now
        help="Your GCP project ID."
    )
    parser.add_argument(
        "--region",
        required=True, # Force the user to provide this now
        help="The GCP region to create resources in (e.g., us-central1)."
    )
    parser.add_argument(
        "--gcs_bucket",
        required=True, # Force the user to provide this now
        help="The name of the GCS bucket where you uploaded the CSV files (from the notebook)."
    )
    
    args = parser.parse_args()
    
    main(args.project_id, args.region, args.gcs_bucket)

