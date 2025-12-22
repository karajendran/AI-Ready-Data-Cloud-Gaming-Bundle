import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
from google.cloud import bigquery
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Configuration ---
DATASET_ID = "eve_data_demo"
SOURCE_VIEW = "stats_per_minute"
ARTIFACT_DIR = "model_artifacts"

def train_and_save(project_id):
    # 0. Cleanup previous run
    if os.path.exists(ARTIFACT_DIR):
        shutil.rmtree(ARTIFACT_DIR)
    os.makedirs(ARTIFACT_DIR)

    # 1. GET DATA
    print(f"🔌 Connecting to BigQuery project: {project_id}...")
    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT
            transaction_count,
            total_quantity,
            unique_players,
            avg_price
        FROM `{project_id}.{DATASET_ID}.{SOURCE_VIEW}`
        WHERE transaction_count < 1000
        LIMIT 50000
    """

    try:
        df = client.query(query).to_dataframe()
        print(f"✅ Data loaded successfully: {len(df)} rows")
    except Exception as e:
        print(f"❌ BigQuery Error: {e}")
        return

    # 2. PREPROCESS
    df = df.fillna(0)
    train_data = df.values.astype('float32')

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std == 0] = 1 

    train_data_norm = (train_data - mean) / std

    # 3. BUILD AUTOENCODER
    input_dim = 4 
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(8, activation="relu")(input_layer)
    encoder = layers.Dense(4, activation="relu")(encoder)
    bottleneck = layers.Dense(2, activation="relu")(encoder) 
    decoder = layers.Dense(4, activation="relu")(bottleneck)
    decoder = layers.Dense(8, activation="relu")(decoder)
    output_layer = layers.Dense(input_dim, activation="linear")(decoder)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # 4. TRAIN
    print("🤖 Training Autoencoder...")
    autoencoder.fit(
        train_data_norm, 
        train_data_norm, 
        epochs=20, 
        batch_size=64,
        shuffle=True,
        verbose=0
    )

    # 5. CALCULATE THRESHOLD
    reconstructions = autoencoder.predict(train_data_norm)
    mse = np.mean(np.power(train_data_norm - reconstructions, 2), axis=1)
    threshold = float(np.quantile(mse, 0.99))

    print(f"✅ Training Complete.")
    print(f"⚠️ Anomaly Threshold set to: {threshold:.4f}")

    # 6. EXPORT ARTIFACTS
    model_path = os.path.join(ARTIFACT_DIR, "saved_model")
    try:
        autoencoder.export(model_path)
    except AttributeError:
        autoencoder.save(model_path)
    
    print(f"💾 Model saved to: {model_path}")

    stats_path = os.path.join(ARTIFACT_DIR, "stats.json")
    stats = {
        "threshold": threshold,
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f"📝 Normalization stats saved to: {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    args = parser.parse_args()
    
    train_and_save(args.project_id)

