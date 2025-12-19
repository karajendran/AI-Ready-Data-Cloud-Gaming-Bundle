#!/usr/bin/env python

import argparse
from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.cloud import storage
import subprocess

# --- Configuration ---

# --- Demo Infrastructure Resources ---
PUBSUB_TOPIC_ID = "eve-telemetry-stream"
PUBSUB_SUBSCRIPTION_ID = "eve-telemetry-sub" # Subscription to prevent data loss
BIGQUERY_DATASET_ID = "eve_data_demo"
BIGQUERY_FACT_TABLE_ID = "fact_game_events" 

# Schema for the LIVE fact_game_events table
BIGQUERY_TABLE_SCHEMA = [
    bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("event_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("location_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("item_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("quantity", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("price_per_item", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("is_buy_order", "BOOLEAN", mode="NULLABLE"),
]

def enable_apis(project_id):
    """
    Automatically enables necessary APIs using gcloud.
    """
    services = [
        "pubsub.googleapis.com",
        "dataflow.googleapis.com",
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

def create_gcs_bucket(project_id, bucket_name, location):
    """Creates a GCS bucket for Dataflow staging."""
    print(f"Attempting to create GCS bucket: {bucket_name}...")
    
    # Explicitly pass project_id to ensure bucket belongs to the correct project
    storage_client = storage.Client(project=project_id)
    
    try:
        bucket = storage_client.create_bucket(bucket_name, location=location)
        print(f"Successfully created GCS bucket: {bucket.name}")
    except exceptions.Conflict:
        print(f"GCS bucket '{bucket_name}' already exists.")
    except Exception as e:
        print(f"Error creating GCS bucket: {e}")

def create_pubsub_topic(project_id, topic_id):
    """Creates a Pub/Sub topic."""
    print(f"Attempting to create Pub/Sub topic: {topic_id}...")
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    try:
        topic = publisher.create_topic(request={"name": topic_path})
        print(f"Successfully created Pub/Sub topic: {topic.name}")
    except exceptions.AlreadyExists:
        print(f"Pub/Sub topic '{topic_id}' already exists.")
    except Exception as e:
        print(f"Error creating Pub/Sub topic: {e}")

def create_pubsub_subscription(project_id, topic_id, subscription_id):
    """Creates a Pub/Sub subscription for the topic."""
    print(f"Attempting to create Pub/Sub subscription: {subscription_id}...")
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = subscriber.topic_path(project_id, topic_id)
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    try:
        # Check if subscription exists by attempting to create it
        subscription = subscriber.create_subscription(
            request={"name": subscription_path, "topic": topic_path}
        )
        print(f"Successfully created Pub/Sub subscription: {subscription.name}")
    except exceptions.AlreadyExists:
        print(f"Pub/Sub subscription '{subscription_id}' already exists.")
    except Exception as e:
        print(f"Error creating Pub/Sub subscription: {e}")

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

def create_bigquery_table(bq_client, dataset_ref, table_id, schema):
    """Creates a BigQuery table with a defined schema."""
    print(f"Attempting to create BigQuery table: {table_id}...")
    table_ref = dataset_ref.table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    
    try:
        bq_client.create_table(table)
        print(f"Successfully created BigQuery table: {table_id}")
    except exceptions.Conflict:
        print(f"BigQuery table '{table_id}' already exists.")
    except Exception as e:
        print(f"Error creating BigQuery table: {e}")

def main(project_id, region):
    
    GCS_STAGING_BUCKET = f"{project_id}-dataflow-staging" # Bucket names must be globally unique

    # Step 1: Enable APIs (manual step)
    enable_apis(project_id)
    
    # Step 2: Create GCS Bucket for Dataflow staging
    # FIX: Passed project_id here
    create_gcs_bucket(project_id, GCS_STAGING_BUCKET, region)
    
    # Step 3: Create Pub/Sub Topic AND Subscription
    create_pubsub_topic(project_id, PUBSUB_TOPIC_ID)
    create_pubsub_subscription(project_id, PUBSUB_TOPIC_ID, PUBSUB_SUBSCRIPTION_ID)
    
    # Step 4: Create BigQuery Dataset
    bq_client = bigquery.Client(project=project_id)
    dataset_ref = create_bigquery_dataset(project_id, BIGQUERY_DATASET_ID, region)
    
    if dataset_ref:
        # Step 5: Create the EMPTY table for LIVE data
        create_bigquery_table(bq_client, 
                              dataset_ref, 
                              BIGQUERY_FACT_TABLE_ID, 
                              BIGQUERY_TABLE_SCHEMA)

    print("\n--- Streaming Infrastructure Provisioning Complete ---")
    print(f"Staging Bucket: {GCS_STAGING_BUCKET}")
    print(f"Pub/Sub Topic: {PUBSUB_TOPIC_ID}")
    print(f"Pub/Sub Subscription: {PUBSUB_SUBSCRIPTION_ID}")
    print(f"BigQuery Dataset: {BIGQUERY_DATASET_ID}")
    print(f"BigQuery Fact Table: {BIGQUERY_FACT_TABLE_ID}")

if __name__ == "__main__":
    # pip install google-api-python-client google-cloud-storage google-cloud-pubsub google-cloud-bigquery
    # gcloud auth application-default login
    
    parser = argparse.ArgumentParser(
        description="Provision Streaming Infrastructure for the EVE Demo."
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
    
    args = parser.parse_args()
    
    main(args.project_id, args.region)

