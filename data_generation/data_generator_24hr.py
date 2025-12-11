#!/usr/bin/env python

import argparse
import json
import random
import datetime
from google.cloud import bigquery

# --- Configuration ---
OUTPUT_FILENAME = "eve_24h_history.jsonl"
PROJECT_ID = "your-gcp-project-id" # <--- UPDATE THIS
DATASET_ID = "eve_data_demo"
PLAYER_SHIPS_TABLE_ID = "dim_player_ships_features"

# --- Simulation Settings ---
# We simulate the LAST 24 hours
END_TIME = datetime.datetime.utcnow()
START_TIME = END_TIME - datetime.timedelta(hours=24)

# Anomaly Settings (Triggered 4 hours ago, lasts 1 hour)
ANOMALY_START = END_TIME - datetime.timedelta(hours=4)
ANOMALY_END = ANOMALY_START + datetime.timedelta(minutes=60)
ANOMALY_ITEM_ID = 603 # Kestrel

def load_foundational_data(project_id):
    """Fetches valid Frigates from BQ to ensure data realism."""
    print("Fetching item definitions from BigQuery...")
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT ship_typeID as item_id, basePrice as average_price
        FROM `{project_id}.{DATASET_ID}.{PLAYER_SHIPS_TABLE_ID}`
        WHERE group_name = 'Frigate' AND ship_typeID != {ANOMALY_ITEM_ID}
    """
    return [dict(row) for row in client.query(query).result()]

def generate_event(timestamp, event_type, player_id, item_id, price, quantity, is_buy):
    return {
        "event_timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "player_id": player_id,
        "location_id": 10000002, # Jita 4-4
        "item_id": item_id,
        "quantity": quantity,
        "price_per_item": price,
        "is_buy_order": is_buy
    }

def main(project_id):
    items = load_foundational_data(project_id)
    if not items: return

    print(f"Generating 24 hours of data ({START_TIME.isoformat()} to {END_TIME.isoformat()})...")
    
    current_time = START_TIME
    total_events = 0
    
    with open(OUTPUT_FILENAME, 'w') as f:
        while current_time < END_TIME:
            # 1. Determine if Anomaly is Active
            is_anomaly = ANOMALY_START <= current_time <= ANOMALY_END
            
            # 2. Simulate "Background Noise" (Traders)
            # Occurs frequently
            if random.random() < 0.3: 
                item = random.choice(items)
                evt = generate_event(
                    timestamp=current_time,
                    event_type="market_transaction",
                    player_id="Trader_Bot",
                    item_id=item['item_id'],
                    price=round((item['average_price'] or 100000) * random.uniform(0.95, 1.05), 2),
                    quantity=random.randint(1, 20),
                    is_buy=random.choice([True, False])
                )
                f.write(json.dumps(evt) + "\n")
                total_events += 1

            # 3. Simulate "Industrial Bursts"
            # Occurs occasionally (every ~10 mins virtual time)
            if random.random() < 0.05:
                item = random.choice(items)
                burst_qty = random.randint(100, 500)
                # Bursts create multiple events for the same manufacturing job
                evt = generate_event(
                    timestamp=current_time,
                    event_type="item_manufactured",
                    player_id="Indy_Corp",
                    item_id=item['item_id'],
                    price=None, # Manufacturing has no market price
                    quantity=burst_qty,
                    is_buy=None
                )
                f.write(json.dumps(evt) + "\n")
                total_events += 1

            # 4. Simulate ANOMALY (Flooding)
            if is_anomaly and random.random() < 0.8: # High probability during anomaly window
                evt = generate_event(
                    timestamp=current_time,
                    event_type="item_manufactured",
                    player_id="MALICIOUS_USER_77",
                    item_id=ANOMALY_ITEM_ID,
                    price=None,
                    quantity=random.randint(50, 100), # Smaller batches, but VERY frequent
                    is_buy=None
                )
                f.write(json.dumps(evt) + "\n")
                total_events += 1

            # Advance time by random seconds (simulating uneven traffic)
            # During anomaly, traffic is denser (time moves slower per loop)
            step = random.randint(1, 5) if is_anomaly else random.randint(5, 30)
            current_time += datetime.timedelta(seconds=step)

    print(f"Done! Generated {total_events} events.")
    print(f"File saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", default=PROJECT_ID)
    args = parser.parse_args()
    main(args.project_id)

