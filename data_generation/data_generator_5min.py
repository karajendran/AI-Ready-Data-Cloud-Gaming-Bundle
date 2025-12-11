#!/usr/bin/env python

import argparse
import json
import random
import threading
import time
import datetime # <--- ADDED
from google.cloud import pubsub_v1
from google.cloud import bigquery

# --- Configuration ---
PROJECT_ID = "your-gcp-project-id" 
TOPIC_ID = "eve-telemetry-stream"

# --- Runtime Configuration ---
SIMULATION_DURATION_SECONDS = 240
ANOMALY_START_TIME_SECONDS = 120
ANOMALY_DURATION_SECONDS = 60

# --- BigQuery Configuration ---
BIGQUERY_DATASET_ID = "eve_data_demo"
PLAYER_SHIPS_TABLE_ID = "dim_player_ships_features"

# --- Anomaly Candidate ---
# "Frigate" (Kestrel, 603)
ANOMALY_ITEM_ID = 603 

# Global flag to signal threads to stop
stop_event = threading.Event()

def load_foundational_data_from_bq(project_id):
    """Loads foundational data from our dim_player_ships_features table."""
    print("Loading foundational data from BigQuery...")
    bq_client = bigquery.Client(project=project_id)
    
    query = f"""
        SELECT
          ship_typeID AS item_id,
          ship_name AS item_name,
          basePrice AS average_price
        FROM
          `{project_id}.{BIGQUERY_DATASET_ID}.{PLAYER_SHIPS_TABLE_ID}`
        WHERE
          group_name = 'Frigate'  
          AND ship_typeID != {ANOMALY_ITEM_ID}
    """
    
    try:
        query_job = bq_client.query(query)
        results = query_job.result()
        item_list = [dict(row) for row in results]
        
        if not item_list:
            print("Warning: Foundational data query returned no items.")
            return []
            
        print(f"Successfully loaded {len(item_list)} 'Frigate' items from BigQuery.")
        return item_list
        
    except Exception as e:
        print(f"Error loading foundational data from BigQuery: {e}")
        return []

def publish_message(publisher, topic_path, event_data):
    """Publishes a single JSON event to Pub/Sub."""
    try:
        data = json.dumps(event_data).encode("utf-8")
        future = publisher.publish(topic_path, data)
        future.result()
    except Exception as e:
        print(f"Error publishing message: {e}")

def simulate_industrialist(publisher, topic_path, normal_items):
    """Simulates a 'capitalist' player who manufactures Frigates."""
    print("[Industrialist] Simulation thread started (Building Frigates).")
    while not stop_event.is_set():
        time.sleep(random.uniform(10, 30))
        
        if stop_event.is_set(): break
            
        item = random.choice(normal_items)
        burst_size = random.randint(1000, 2000)
        
        print(f"[Industrialist] Starting burst: {burst_size}x item {item['item_id']} ({item['item_name']})")
        
        for _ in range(burst_size):
            # FIXED: Use datetime for correct ISO format
            event = {
                "event_timestamp": datetime.datetime.utcnow().isoformat(),
                "event_type": "item_manufactured",
                "player_id": "Industrial_Corp_01",
                "location_id": 10000002,
                "item_id": item['item_id'],
                "quantity": 1,
                "price_per_item": None,
                "is_buy_order": None
            }
            publish_message(publisher, topic_path, event)
            time.sleep(0.001) 

def simulate_trader(publisher, topic_path, normal_items):
    """Simulates a 'trader' player who creates market transactions for Frigates."""
    print("[Trader] Simulation thread started (Trading Frigates).")
    while not stop_event.is_set():
        time.sleep(random.uniform(5, 15))
        
        if stop_event.is_set(): break
            
        item = random.choice(normal_items)
        price = (item['average_price'] or 100000) * random.uniform(0.9, 1.1) 
        quantity = random.randint(1, 50)
        
        # FIXED: Use datetime for correct ISO format
        event = {
            "event_timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": "market_transaction",
            "player_id": "Market_Trader_01",
            "location_id": 10000002,
            "item_id": item['item_id'],
            "quantity": quantity,
            "price_per_item": round(price, 2),
            "is_buy_order": random.choice([True, False])
        }
        publish_message(publisher, topic_path, event)

def simulate_anomaly(publisher, topic_path):
    """The anomaly event: flooding the stream with our chosen Frigate."""
    print("\n" + "*"*50)
    print(f"!!! ANOMALY TRIGGERED !!! Flooding stream with item {ANOMALY_ITEM_ID} (Kestrel)")
    print("*"*50 + "\n")
    
    start_time = time.time()
    
    while (time.time() - start_time) < ANOMALY_DURATION_SECONDS:
        if stop_event.is_set(): break
            
        # FIXED: Use datetime for correct ISO format
        event = {
            "event_timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": "item_manufactured",
            "player_id": "Bugged_Player_77",
            "location_id": 60003760,
            "item_id": ANOMALY_ITEM_ID,
            "quantity": 100,
            "price_per_item": None,
            "is_buy_order": None
        }
        publish_message(publisher, topic_path, event)
        time.sleep(0.01)

    print("\n" + "*"*50)
    print("!!! ANOMALY CONCLUDED !!! Stream returning to normal.")
    print("*"*50 + "\n")

def main(project_id):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, TOPIC_ID)
    
    normal_items = load_foundational_data_from_bq(project_id)
    if not normal_items:
        print("Could not load foundational data. Exiting.")
        return
        
    print(f"Starting 'chaotic normal' simulation. Publishing to: {topic_path}")
    
    industrialist_thread = threading.Thread(
        target=simulate_industrialist, 
        args=(publisher, topic_path, normal_items),
        daemon=True
    )
    trader_thread = threading.Thread(
        target=simulate_trader, 
        args=(publisher, topic_path, normal_items),
        daemon=True
    )
    
    industrialist_thread.start()
    trader_thread.start()
    
    print("\nSimulation is running. 'Chaotic normal' data is being generated.")
    print(f"The anomaly will be triggered in {ANOMALY_START_TIME_SECONDS} seconds...")
    
    time.sleep(ANOMALY_START_TIME_SECONDS)
    
    simulate_anomaly(publisher, topic_path)
    
    remaining_time = SIMULATION_DURATION_SECONDS - ANOMALY_START_TIME_SECONDS - ANOMALY_DURATION_SECONDS
    if remaining_time > 0:
        print(f"Letting simulation run for another {remaining_time} seconds...")
        time.sleep(remaining_time)
        
    print("Demo generator has completed its run.")
    stop_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates synthetic EVE Online telemetry data.")
    parser.add_argument("--project_id", default=PROJECT_ID, help="Your GCP project ID.")
    args = parser.parse_args()
    
    if args.project_id == "your-gcp-project-id":
        print("Error: Please update the PROJECT_ID variable or pass --project_id.")
    else:
        main(args.project_id)


