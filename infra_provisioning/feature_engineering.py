#!/usr/bin/env python

import argparse
from google.cloud import bigquery
from google.api_core.exceptions import Conflict

# --- Configuration ---
DATASET_ID = "eve_data_demo"
VIEW_ID = "stats_per_minute"
SOURCE_TABLE = "fact_game_events"

def create_features_view(project_id):
    """
    Creates the 'stats_per_minute' view in BigQuery.
    This view transforms raw event logs into behavioral vectors for ML.
    """
    client = bigquery.Client(project=project_id)
    view_ref = f"{project_id}.{DATASET_ID}.{VIEW_ID}"
    
    print(f"--- Feature Engineering: Creating View {VIEW_ID} ---")

    # The SQL Definition of the View
    # Note: We construct the Fully Qualified Table Name explicitly
    view_query = f"""
    CREATE OR REPLACE VIEW `{view_ref}` AS
    SELECT
      item_id,
      location_id,
      TIMESTAMP_TRUNC(event_timestamp, MINUTE) as minute_window,
      
      -- Feature 1: VELOCITY (Spam Detection)
      COUNT(*) as transaction_count,

      -- Feature 2: VOLUME (Market Manipulation)
      SUM(quantity) as total_quantity,

      -- Feature 3: NETWORK (Organic vs Artificial)
      COUNT(DISTINCT player_id) as unique_players,

      -- Feature 4: ECONOMY (RMT Detection)
      AVG(price_per_item) as avg_price

    FROM
      `{project_id}.{DATASET_ID}.{SOURCE_TABLE}`
    GROUP BY
      item_id, location_id, TIMESTAMP_TRUNC(event_timestamp, MINUTE);
    """

    try:
        # We use query() here because DDL statements (CREATE VIEW) are standard queries
        job = client.query(view_query)
        job.result() # Wait for the job to complete
        print(f"✅ Successfully created view: {view_ref}")
        print("   -> Aggregates 'fact_game_events' into behavioral vectors.")
        
    except Exception as e:
        print(f"❌ Error creating view: {e}")
        # Helpful hint for common errors
        if "Not found" in str(e):
            print(f"   Hint: Does the source table '{SOURCE_TABLE}' exist yet?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    args = parser.parse_args()

    create_features_view(args.project_id)

