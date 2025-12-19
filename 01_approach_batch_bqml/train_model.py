#!/usr/bin/env python

import argparse
from google.cloud import bigquery
import sys

# --- Configuration ---
DATASET_ID = "eve_data_demo"
MODEL_ID = "behavior_anomaly_model"
SOURCE_VIEW = "stats_per_minute"

def train_kmeans_model(project_id):
    """
    Trains the Unsupervised K-Means model using BQML.
    Then, it fetches the centroid details to display the detected clusters.
    """
    client = bigquery.Client(project=project_id)
    model_ref = f"{project_id}.{DATASET_ID}.{MODEL_ID}"
    
    print(f"--- 🧠 Training K-Means Model: {MODEL_ID} ---")
    print(f"    Source: {SOURCE_VIEW} (Behavioral Vectors)")
    print("    Training in progress (this typically takes 30-60 seconds)...")

    # 1. CREATE MODEL QUERY
    training_query = f"""
    CREATE OR REPLACE MODEL `{model_ref}`
    OPTIONS(
      model_type='kmeans',
      num_clusters=5,
      standardize_features = TRUE
    ) AS
    SELECT
      transaction_count,
      total_quantity,
      unique_players,
      avg_price
    FROM
      `{project_id}.{DATASET_ID}.{SOURCE_VIEW}`;
    """

    try:
        job = client.query(training_query)
        job.result() # Wait for training to complete
        print(f"✅ Model trained successfully.")
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return

    # 2. INSPECT CENTROIDS (The "Smoking Gun")
    print("\n--- 📊 Cluster Analysis (The 'Smoking Gun') ---")
    print("Identifying the 'Bot' cluster based on Velocity (Actions/Min)...")
    
    # FIX: ML.CENTROIDS returns a long format (key-value pairs). We must pivot it.
    analysis_query = f"""
    SELECT
      centroid_id,
      ROUND(MAX(IF(feature = 'transaction_count', numerical_value, NULL)), 1) as avg_actions_per_min,
      ROUND(MAX(IF(feature = 'unique_players', numerical_value, NULL)), 1) as avg_unique_players,
      ROUND(MAX(IF(feature = 'total_quantity', numerical_value, NULL)), 0) as avg_quantity,
      ROUND(MAX(IF(feature = 'avg_price', numerical_value, NULL)), 2) as avg_price
    FROM
      ML.CENTROIDS(MODEL `{model_ref}`)
    GROUP BY
      centroid_id
    ORDER BY
      avg_actions_per_min DESC;
    """
    
    results = client.query(analysis_query)
    
    print(f"{'Cluster':<10} | {'APM (Velocity)':<15} | {'Unique Players':<15} | {'Avg Price':<15}")
    print("-" * 65)
    
    for row in results:
        cluster_label = f"Cluster {row.centroid_id}"
        
        # Simple heuristic to label the row for the demo output
        if row.avg_actions_per_min > 2000:
            cluster_label += " 🚨 (EXPLOIT)"
        elif row.avg_actions_per_min > 50:
            cluster_label += " 🏭 (Industry)"
        else:
            cluster_label += " 👤 (Normal)"

        print(f"{cluster_label:<20} | {row.avg_actions_per_min:<15} | {row.avg_unique_players:<15} | {row.avg_price:<15}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    args = parser.parse_args()

    train_kmeans_model(args.project_id)

