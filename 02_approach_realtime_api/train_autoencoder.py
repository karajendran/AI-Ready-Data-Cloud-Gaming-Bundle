import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from google.cloud import bigquery
import os

# ==========================================
# CONFIGURATION
# ==========================================
# ⚠️ REPLACE WITH YOUR ACTUAL PROJECT ID
project_id = "cloud-sa-ml" 
# ==========================================

def train_and_save():
    # 1. GET DATA
    # We use BigQuery to fetch the training history ("The Normal Behavior")
    print(f"🔌 Connecting to BigQuery project: {project_id}...")
    client = bigquery.Client(project=project_id)

    query = """
        SELECT
            transaction_count,
            total_quantity,
            unique_players,
            avg_price
        FROM `eve_data_demo.stats_per_minute`
        WHERE minute_window < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
        LIMIT 100000
    """

    try:
        df = client.query(query).to_dataframe()
        print(f"✅ Data loaded successfully: {len(df)} rows")
    except Exception as e:
        print(f"❌ BigQuery Error: {e}")
        print("Tip: Check if 'eve_data_demo.stats_per_minute' exists and if you have permission.")
        return

    # 2. PREPROCESS
    # Fill missing values (e.g., if avg_price is null for 0 transactions)
    df = df.fillna(0)

    # Normalize data to 0-1 range (Crucial for Neural Networks to converge)
    train_data = df.values.astype('float32')
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    # Prevent division by zero if a column has constant values (std=0)
    std[std == 0] = 1 

    train_data_norm = (train_data - mean) / std

    # 3. BUILD AUTOENCODER
    # Input Dimension = 4 (Velocity, Volume, Network, Price)
    input_dim = 4 

    # Encoder (Compress the data)
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(8, activation="relu")(input_layer)
    encoder = layers.Dense(4, activation="relu")(encoder)
    bottleneck = layers.Dense(2, activation="relu")(encoder) # The "Essence" of behavior

    # Decoder (Try to reconstruct the original data)
    decoder = layers.Dense(4, activation="relu")(bottleneck)
    decoder = layers.Dense(8, activation="relu")(decoder)
    output_layer = layers.Dense(input_dim, activation="linear")(decoder)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse') # MSE = Mean Squared Error

    # 4. TRAIN
    print("🤖 Training Autoencoder on 'Normal' behavior...")
    # We train the model to predict its own input (X -> X)
    autoencoder.fit(
        train_data_norm, 
        train_data_norm, 
        epochs=20, 
        batch_size=64,
        shuffle=True,
        verbose=1
    )

    # 5. SAVE FOR VERTEX AI
    # FIX: Use export() instead of save() to generate a TF SavedModel artifact
    # This creates the correct directory structure for Vertex AI deployment
    export_path = 'game_health_autoencoder'
    print(f"💾 Exporting model artifact to '{export_path}'...")
    autoencoder.export(export_path)

    # 6. CALCULATE THRESHOLD (The "Red Line")
    # We look at the reconstruction error on the training set.
    # Any future point with error higher than the 99th percentile is an anomaly.
    reconstructions = autoencoder.predict(train_data_norm)
    mse = np.mean(np.power(train_data_norm - reconstructions, 2), axis=1)
    threshold = np.quantile(mse, 0.99)

    print(f"\n✅ Training Complete.")
    print(f"⚠️ Anomaly Threshold set to: {threshold}")
    print(f"Model exported locally at: {os.path.abspath(export_path)}")
    print("Next Step: Upload this folder to a GCS bucket to deploy to Vertex AI.")

if __name__ == "__main__":
    train_and_save()

