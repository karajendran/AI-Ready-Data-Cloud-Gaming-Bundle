import argparse
import logging
import json
import time
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

# --- Configuration ---

# This is the schema for our BigQuery table.
# It MUST match the table we created with provision_streaming_infra.py
TABLE_SCHEMA = (
    "event_timestamp:TIMESTAMP, "
    "event_type:STRING, "
    "player_id:STRING, "
    "location_id:INTEGER, "
    "item_id:INTEGER, "
    "quantity:INTEGER, "
    "price_per_item:FLOAT, "
    "is_buy_order:BOOLEAN"
)

class ParsePubSubMessage(beam.DoFn):
    """
    Parses the JSON message from Pub/Sub and yields a dictionary.
    """
    def process(self, element):
        try:
            # Decode the byte string from Pub/Sub
            message_str = element.decode('utf-8')
            message_dict = json.loads(message_str)
            
            # --- Data Validation ---
            if "event_timestamp" not in message_dict:
                # Log specific error for missing timestamp
                logging.error(f"DROPPED: Missing 'event_timestamp'. Data: {message_str[:100]}...")
                return # Stop processing this element

            yield message_dict
            
        except json.JSONDecodeError as e:
            logging.warning(f"DROPPED: JSON Decode Error: {e}. Data: {element}")
        except Exception as e:
            logging.warning(f"DROPPED: Unexpected error: {e}")

def run(project_id, subscription_id, dataset_id, table_id, gcs_staging_bucket, region, pipeline_args=None):
    """
    The main Apache Beam pipeline.
    """
    
    # Set up pipeline options
    options = PipelineOptions(pipeline_args, streaming=True)
    options.view_as(SetupOptions).save_main_session = True
    
    subscription_path = f"projects/{project_id}/subscriptions/{subscription_id}"
    table_spec = f"{project_id}:{dataset_id}.{table_id}"
    
    print("--- Starting Dataflow Pipeline ---")
    print(f"Reading from Pub/Sub Subscription: {subscription_path}")
    print(f"Writing to BigQuery table: {table_spec}")
    print("---------------------------------")
    print("Submitting job to Dataflow...")

    p = beam.Pipeline(options=options)
        
    (p
     | "1. Read from Pub/Sub" >> beam.io.ReadFromPubSub(subscription=subscription_path)
     | "2. Parse JSON Message" >> beam.ParDo(ParsePubSubMessage())
     | "3. Write to BigQuery" >> beam.io.WriteToBigQuery(
         table_spec,
         schema=TABLE_SCHEMA,
         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
         create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER
     )
    )

    result = p.run()
    print(f"Job submitted successfully! (Job ID: {result.job_id()})")
    print("You can monitor the job in the Google Cloud Console.")
    print("Exiting script now (the job will keep running in the cloud)...")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True, help="Your GCP project ID.")
    parser.add_argument("--subscription_id", required=True, help="The Pub/Sub subscription ID (e.g., eve-telemetry-sub).")
    parser.add_argument("--dataset_id", required=True, help="The BigQuery dataset ID.")
    parser.add_argument("--table_id", required=True, help="The BigQuery table ID.")
    parser.add_argument("--gcs_staging_bucket", required=True, help="GCS bucket for Dataflow staging.")
    parser.add_argument("--region", required=True, help="GCP region (e.g., us-central1).")
    
    # --- Network Configuration Arguments ---
    parser.add_argument("--network", help="GCP VPC network name (optional).")
    parser.add_argument("--subnetwork", help="GCP VPC subnetwork name (optional).")
    
    known_args, pipeline_args = parser.parse_known_args()
    
    # Add standard Dataflow runner arguments
    # We construct the temp_location here so it's consistent
    temp_location = f"gs://{known_args.gcs_staging_bucket}/temp"
   
    job_name = f"eve-telemetry-pipeline-{int(time.time())}"

    pipeline_args.extend([
        f"--runner=DataflowRunner",
        f"--project={known_args.project_id}",
        f"--region={known_args.region}",
        f"--temp_location={temp_location}",
        f"--job_name={job_name}" 
    ])

    # --- Append Network args if provided ---
    if known_args.network:
        pipeline_args.append(f"--network={known_args.network}")
    if known_args.subnetwork:
        pipeline_args.append(f"--subnetwork={known_args.subnetwork}")

    run(
        project_id=known_args.project_id,
        subscription_id=known_args.subscription_id, 
        dataset_id=known_args.dataset_id,
        table_id=known_args.table_id,
        gcs_staging_bucket=known_args.gcs_staging_bucket,
        region=known_args.region,
        pipeline_args=pipeline_args
    )

