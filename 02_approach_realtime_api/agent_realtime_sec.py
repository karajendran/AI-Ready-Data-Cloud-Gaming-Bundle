import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part
from google.cloud import aiplatform
from google.cloud import bigquery
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
project_id = "cloud-sa-ml" 
endpoint_id = "4701642562253881344" # ✅ Your live Endpoint ID
location = "us-central1"

# 📊 NORMALIZATION STATS (Estimated from "Normal" Cluster 3)
# We must scale inputs to ~0-1 range so the Neural Network understands them.
SCALER = {
    "mean": np.array([27.8, 50000.0, 1.0, 100000.0]), # [Velocity, Volume, Players, Price]
    "std":  np.array([10.0, 20000.0, 1.0, 50000.0])
}
# ==========================================

# Initialize
vertexai.init(project=project_id, location=location)
aiplatform.init(project=project_id, location=location)
bq_client = bigquery.Client(project=project_id)

# ==========================================
# Tool 1: Get The Data (From BQ)
# ==========================================
def get_player_stats(player_id):
    """Fetches the latest behavioral vector for a specific player."""
    print(f"\n🔍 [Tool] Fetching stats for {player_id}...")
    
    query = f"""
        SELECT
            COUNT(*) as transaction_count,
            SUM(quantity) as total_quantity,
            1 as unique_players, 
            COALESCE(AVG(price_per_item), 0) as avg_price
        FROM `eve_data_demo.game_events`
        WHERE player_id = '{player_id}'
        GROUP BY TIMESTAMP_TRUNC(event_timestamp, MINUTE)
        ORDER BY transaction_count DESC 
        LIMIT 1
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
            return "Error: Player not found in recent logs."
        return df # Returns DataFrame (Agent will convert to JSON)
    except Exception as e:
        return f"Error querying BigQuery: {e}"

# ==========================================
# Tool 2: Get The Verdict (From Vertex AI Endpoint)
# ==========================================
def check_anomaly_score(transaction_count, total_quantity, unique_players, avg_price):
    """Sends player stats to the Autoencoder Endpoint to get an Anomaly Score."""
    print(f"\n🚀 [Tool] Pinging Vertex AI Endpoint for Anomaly Score...")
    
    # 1. PREPROCESS (Normalize Inputs)
    # The model expects scaled data (roughly 0-1 range). 
    # Sending raw '4746' breaks the math, causing the trillion-point explosion.
    features_raw = np.array([transaction_count, total_quantity, unique_players, avg_price])
    
    # Apply Z-Score Normalization: (Value - Mean) / Std
    # We add 1e-6 to std to prevent division by zero
    features_norm = (features_raw - SCALER["mean"]) / (SCALER["std"] + 1e-6)
    
    try:
        # 2. Call Endpoint
        endpoint_name = f"projects/192541734100/locations/{location}/endpoints/{endpoint_id}"
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        
        # Send the NORMALIZED features
        response = endpoint.predict(instances=[features_norm.tolist()])
        
        # 3. Calculate Error
        input_vec = features_norm
        output_vec = np.array(response.predictions[0])
        mse = np.mean(np.power(input_vec - output_vec, 2))
        
        # 4. Verdict Logic
        # Threshold is now relative to normalized variance (much smaller number)
        threshold = 10.0 
        status = "ANOMALY" if mse > threshold else "NORMAL"
        
        # Return rich context so the Agent can explain WHY
        return {
            "player_metrics": {
                "apm": transaction_count,
                "volume": total_quantity
            },
            "baseline_metrics": {
                "apm": SCALER["mean"][0], # 27.8
                "volume": SCALER["mean"][1]
            },
            "anomaly_score": round(mse, 2), # Normalized Score (e.g., 50.0)
            "threshold": threshold,
            "verdict": status
        }
    except Exception as e:
        return f"Error calling Vertex Endpoint: {e}"

# ==========================================
# Register Tools
# ==========================================
tools = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="get_player_stats",
            description="Get the latest raw stats (velocity, volume) for a player. Use this FIRST.",
            parameters={
                "type": "object",
                "properties": {"player_id": {"type": "string"}}
            }
        ),
        FunctionDeclaration(
            name="check_anomaly_score",
            description="Send stats to the AI Model to calculate the anomaly score. Use this SECOND.",
            parameters={
                "type": "object",
                "properties": {
                    "transaction_count": {"type": "number"},
                    "total_quantity": {"type": "number"},
                    "unique_players": {"type": "number"},
                    "avg_price": {"type": "number"}
                }
            }
        )
    ]
)

# ==========================================
# Agent Loop
# ==========================================
print("🤖 Initializing Security Analyst Agent...")

model = GenerativeModel(
    "gemini-2.5-flash", 
    tools=[tools],
    system_instruction="""
    You are a Senior Game Security Analyst.
    
    YOUR GOAL: Explain potential exploits clearly to Game Masters.
    
    RESPONSE STYLE GUIDELINES:
    1. **Be Comparative:** Always contrast the suspect's stats against the "Normal Baseline" returned by the tool.
       - Bad: "Score is high."
       - Good: "Suspect APM is 4,700, which is 170x higher than the normal baseline of 27.8."
    2. **Be Decisive:** State clearly if this is an exploit or normal play.
    3. **Use the Data:** Cite the specific Anomaly Score.
    
    WORKFLOW:
    1. Call `get_player_stats` to get raw numbers.
    2. Call `check_anomaly_score` to get the AI verdict and Baselines.
    3. Synthesize the answer.
    """
)
chat = model.start_chat()

def query_agent(user_input):
    print(f"\n👤 User: {user_input}")
    
    # 1. Send message
    response = chat.send_message(user_input)
    
    # 2. Handle Tool Calls
    while response.candidates[0].content.parts[0].function_call:
        call = response.candidates[0].content.parts[0].function_call
        name = call.name
        args = call.args
        
        result = ""
        if name == "get_player_stats":
            result = get_player_stats(args["player_id"])
            if not isinstance(result, str):
                result = result.to_json(orient="records")
                
        elif name == "check_anomaly_score":
            result = check_anomaly_score(
                args["transaction_count"], args["total_quantity"], 
                args["unique_players"], args["avg_price"]
            )
            
        # 3. Return Tool Output
        response = chat.send_message(
            Part.from_function_response(name=name, response={"content": result})
        )
        
    print(f"🤖 Agent: {response.text}")

# Run
if __name__ == "__main__":
    # Step 1: Ask about the suspicious player
    query_agent("Is Bugged_Player_001 acting weird?")

