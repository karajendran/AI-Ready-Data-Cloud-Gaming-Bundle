import argparse
import random
import time
import json
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery

# --- Configuration ---
OUTPUT_FILENAME = "eve_24h_history.jsonl"
DATASET_ID = "eve_data_demo"
TABLE_ID = "game_events" 
DIM_TABLE_ID = "dim_player_ships_features"

# Anomaly Settings (Triggered 2 hours ago, lasts 1 hour)
ANOMALY_DURATION_MINS = 60 

def load_foundational_data(client, project_id):
    """Fetches real items/prices from BQ."""
    print("Fetching item definitions from BigQuery...")
    try:
        query = f"""
            SELECT ship_typeID as item_id, basePrice as average_price
            FROM `{project_id}.{DATASET_ID}.{DIM_TABLE_ID}`
            LIMIT 50
        """
        results = [dict(row) for row in client.query(query).result()]
        return results if results else [{"item_id": 34, "average_price": 5.0}]
    except Exception as e:
        print(f"⚠️ Warning: Using defaults. ({e})")
        return [{"item_id": 34, "average_price": 5.0}, {"item_id": 29668, "average_price": 3000000.0}]

def generate_file(project_id, days=1):
    client = bigquery.Client(project=project_id)
    items = load_foundational_data(client, project_id)
    
    print(f"--- Generating {days} Day(s) of Synthetic Game Data ---")
    
    # Use timezone-aware UTC to avoid DeprecationWarning
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    current_time = start_time
    
    # Anomaly Window (The last hour of the simulation)
    anomaly_start = end_time - timedelta(minutes=ANOMALY_DURATION_MINS)

    row_count = 0
    
    with open(OUTPUT_FILENAME, 'w') as f:
        # CRITICAL CHANGE: Loop by MINUTE, not by event
        # This ensures we control the 'Density' (Events per Minute)
        while current_time < end_time:
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # ---------------------------------------------------------
            # 1. SCENARIO: Background Noise (Normal Traders)
            # Signature: Low APM (5-20), Variance in items
            # ---------------------------------------------------------
            if random.random() < 0.9: 
                for _ in range(random.randint(5, 20)):
                    item = random.choice(items)
                    row = {
                        "event_timestamp": timestamp_str,
                        "event_type": "market_order",
                        "player_id": f"Player_{random.randint(100, 999)}",
                        "location_id": 60003760, # Jita
                        "item_id": int(item['item_id']),
                        "quantity": random.randint(1, 100),
                        "price_per_item": float(item['average_price'] or 100),
                        "is_buy_order": bool(random.getrandbits(1))
                    }
                    f.write(json.dumps(row) + "\n")
                    row_count += 1

            # ---------------------------------------------------------
            # 2. SCENARIO: Industrial Bursts (Multiboxing Fleets)
            # Signature: Medium APM (~60), HIGH Quantity, Specific Items
            # ---------------------------------------------------------
            if random.random() < 0.15: 
                # Loop 60 times within THIS SAME MINUTE
                for _ in range(60): 
                    row = {
                        "event_timestamp": timestamp_str,
                        "event_type": "manufacturing_job",
                        "player_id": "Industrial_Corp_01",
                        "location_id": 60003760,
                        "item_id": 34, # Tritanium
                        "quantity": random.randint(10000, 50000), 
                        "price_per_item": 5.0,
                        "is_buy_order": True
                    }
                    f.write(json.dumps(row) + "\n")
                    row_count += 1

            # ---------------------------------------------------------
            # 3. SCENARIO: The Anomaly (Exploit / Flooding)
            # Signature: EXTREME APM (2500+), Low Value, Single Player
            # ---------------------------------------------------------
            if current_time > anomaly_start and random.random() < 0.2:
                print(f"⚠️  Injecting Exploit Burst at {timestamp_str}")
                # Loop 2500 times within THIS SAME MINUTE
                for _ in range(2500): 
                    row = {
                        "event_timestamp": timestamp_str,
                        "event_type": "exploit_attempt",
                        "player_id": "Bugged_Player_001",
                        "location_id": 60003760,
                        "item_id": 603, # Kestrel
                        "quantity": 1,
                        "price_per_item": 0.01, # Crash Price
                        "is_buy_order": False
                    }
                    f.write(json.dumps(row) + "\n")
                    row_count += 1

            current_time += timedelta(minutes=1)

    print(f"✅ Generated {row_count} rows in {OUTPUT_FILENAME}")
    return client

def load_to_bigquery(client, project_id):
    """Bulk loads the JSONL file to BigQuery."""
    table_ref = f"{project_id}.{DATASET_ID}.{TABLE_ID}"
    print(f"🚀 Loading {OUTPUT_FILENAME} into {table_ref}...")

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        autodetect=True, 
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )

    with open(OUTPUT_FILENAME, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result() 
    print(f"✅ Loaded {job.output_rows} rows into BigQuery.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    args = parser.parse_args()

    bq_client = generate_file(args.project_id)
    load_to_bigquery(bq_client, args.project_id)

