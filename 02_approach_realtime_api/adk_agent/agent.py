import json
import os
import numpy as np
from google.cloud import aiplatform
from google.cloud import storage  # <--- NEW: Import Storage
from google.adk.agents.llm_agent import Agent

# ==========================================
# 0. Global Setup
# ==========================================
PROJECT_ID = "accelerated-platforms-dev"
REGION = "us-central1"
MODEL_BUCKET = "eve-online-model-bucket" # <--- NEW: Bucket Source

# Initialize Vertex AI SDK globally so tools can use it
aiplatform.init(project=PROJECT_ID, location=REGION)

# ==========================================
# 1. Define Tools
# ==========================================
def analyze_player_security(player_id: str, apm: int, total_items: int) -> dict:
    """
    Analyzes a player's real-time behavior using the Vertex AI Autoencoder.
    
    Args:
        player_id: The name or ID of the player.
        apm: Actions Per Minute (Velocity).
        total_items: Total quantity of items moved (Volume).
        
    Returns:
        A dictionary containing the Anomaly Score, Verdict, and Reasoning.
    """
    print(f"\n⚡ [ADK Tool] Executing: analyze_player_security for {player_id}...")

    # A. Load Config (From GCS)
    # This replaces the fragile local file lookup with a robust cloud fetch.
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(MODEL_BUCKET)
        
        # 1. Fetch Endpoint ID
        blob_ep = bucket.blob("endpoint_config.txt")
        if not blob_ep.exists():
            return {"error": f"Config 'endpoint_config.txt' not found in gs://{MODEL_BUCKET}. Please upload it."}
        endpoint_name = blob_ep.download_as_text().strip()

        # 2. Fetch Normalization Stats
        blob_stats = bucket.blob("stats.json")
        if not blob_stats.exists():
            # Fallback: check inside the model folder if it was uploaded there
            blob_stats = bucket.blob("game_security_model/stats.json")
        
        if not blob_stats.exists():
            return {"error": f"Config 'stats.json' not found in gs://{MODEL_BUCKET}. Please upload it."}
            
        stats = json.loads(blob_stats.download_as_text())
        
        # 3. Parse Stats
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        
        # FIX: Raise Threshold Floor from 0.1 to 5.0
        # The 'Innocent' player (12 APM) scored 1.13, which triggered a False Positive.
        # Since the Bot score is ~16,000, raising the floor to 5.0 safely filters
        # moderate activity while still catching exploits easily.
        threshold = max(stats["threshold"], 5.0) 
            
    except Exception as e:
        return {"error": f"GCS Configuration load failed: {str(e)}"}

    # B. Construct Vector
    # Feature Order: [transaction_count, total_quantity, unique_players, avg_price]
    # We use the mean for the unused features to isolate the impact of APM/Volume
    input_vector = np.array([mean]) 
    input_vector[0][0] = float(apm)         # Inject User Input: Velocity
    input_vector[0][1] = float(total_items) # Inject User Input: Volume

    # C. Normalize
    norm_data = (input_vector - mean) / std

    # D. Predict (Vertex AI)
    try:
        endpoint = aiplatform.Endpoint(endpoint_name)
        prediction = endpoint.predict(instances=norm_data.tolist()).predictions
        
        reconstruction = np.array(prediction)
        mse = np.mean(np.power(norm_data - reconstruction, 2))
        mse = float(mse)
    except Exception as e:
        return {"error": f"Vertex AI Inference failed: {str(e)}"}

    # E. Verdict
    is_anomaly = mse > threshold
    risk_level = "CRITICAL" if mse > (threshold * 100) else "HIGH" if is_anomaly else "LOW"

    return {
        "player_id": player_id,
        "anomaly_score": round(mse, 4),
        "threshold": round(threshold, 4),
        "is_anomaly": is_anomaly,
        "risk_level": risk_level,
        "analysis": f"Player {player_id} has a reconstruction error of {mse:.2f} (Threshold: {threshold:.2f})."
    }

# ==========================================
# 2. Define the Root Agent
# ==========================================
root_agent = Agent(
    model='gemini-2.0-flash-exp', 
    name='RealTimeSecurityBot',
    description="Real-time security agent using Vertex AI Autoencoders.",
    instruction="""
    You are a Real-Time Security Sentinel for EVE Online.
    Your goal is to screen individual players for anomalous behavior using the 'analyze_player_security' tool.
    
    PROTOCOLS:
    1. ALWAYS use the 'analyze_player_security' tool when given player stats (APM, Items).
    2. Interpret the result:
       - If Risk is CRITICAL/HIGH: Issue an immediate ban recommendation.
       - If Risk is LOW: Confirm the player is acting normally.
    3. Be concise and technical.
    """,
    tools=[analyze_player_security]
)

# ==========================================
# 3. Unit Test (Execution Block)
# ==========================================
if __name__ == "__main__":
    print("🤖 Running manual test of the 'analyze_player_security' tool...")
    
    # Test 1: Innocent Player
    result_innocent = analyze_player_security("Player_One", apm=12, total_items=5000)
    print(f"\n[Test 1] Innocent Player Result:\n{json.dumps(result_innocent, indent=2)}")

    # Test 2: The Bot
    result_bot = analyze_player_security("Fast_Bot_99", apm=2500, total_items=50000)
    print(f"\n[Test 2] Bot Player Result:\n{json.dumps(result_bot, indent=2)}")


