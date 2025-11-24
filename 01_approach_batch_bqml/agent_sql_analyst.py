import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part
from google.cloud import bigquery

# Initialize
project_id = "cloud-sa-ml" # ⚠️ REPLACE WITH YOUR PROJECT ID
vertexai.init(project=project_id, location="us-central1")

# ==========================================
# 1. Define the Logic (The "Tools")
# ==========================================
bq_client = bigquery.Client(project=project_id)

def get_suspicious_players():
    """Queries BigQuery to find players exceeding the 2000 actions/min threshold."""
    print("\n⚡ [Tool execution] Querying BigQuery for suspicious players...")
    query = """
        SELECT
            player_id,
            item_id,
            CAST(TIMESTAMP_TRUNC(event_timestamp, MINUTE) as STRING) as time_window,
            COUNT(*) as actions_per_minute
        FROM
            `eve_data_demo.game_events`
        GROUP BY
            player_id, item_id, time_window
        HAVING
            actions_per_minute > 2000
        ORDER BY
            actions_per_minute DESC
        LIMIT 5
    """
    try:
        df = bq_client.query(query).to_dataframe()
        return df.to_json(orient='records')
    except Exception as e:
        return f"Error: {str(e)}"

def get_cluster_stats():
    """Retrieves the K-Means Centroid definitions to explain 'Normal' vs 'Bot' behavior."""
    print("\n⚡ [Tool execution] Retrieving K-Means cluster statistics...")
    query = """
        SELECT
            centroid_id,
            ROUND(MAX(IF(feature = 'transaction_count', numerical_value, NULL)), 1) as avg_transactions_per_min,
            ROUND(MAX(IF(feature = 'unique_players', numerical_value, NULL)), 1) as avg_unique_players
        FROM
            ML.CENTROIDS(MODEL `eve_data_demo.behavior_anomaly_model`)
        GROUP BY
            centroid_id
        ORDER BY
            avg_transactions_per_min DESC
    """
    try:
        df = bq_client.query(query).to_dataframe()
        return df.to_json(orient='records')
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 2. Register Tools with Gemini
# ==========================================

# FIX: Even functions with NO arguments require an empty parameter schema
tools = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="get_suspicious_players",
            description="Get a list of players currently flagged for high-frequency exploit behavior.",
            parameters={"type": "object", "properties": {}}, 
        ),
        FunctionDeclaration(
            name="get_cluster_stats",
            description="Get the statistical definition of 'Normal' vs 'Bot' clusters from the K-Means model.",
            parameters={"type": "object", "properties": {}}, 
        ),
    ]
)

# ==========================================
# 3. The Agent Loop
# ==========================================

print("🤖 Initializing SQL Analyst Agent (BQML)...")

model = GenerativeModel(
    "gemini-2.5-flash",
    tools=[tools],
    system_instruction="""
    You are a Senior Game Security Analyst for EVE Online.
    Your job is to detect economic exploits using BigQuery data.
    
    PROTOCOLS:
    1. Always check for suspicious players first using get_suspicious_players().
    2. If asked 'Why' a player is flagged, you MUST compare them to the 'Normal' cluster stats using get_cluster_stats().
    3. Be concise and professional. Use 'actions per minute' (APM) as your key metric.
    """
)

chat = model.start_chat()

def query_agent(user_input):
    print(f"\n👤 User: {user_input}")
    
    # 1. Send message to Gemini
    response = chat.send_message(user_input)
    
    # 2. Handle Tool Calls (Function Calling)
    while response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        tool_name = function_call.name
        
        # Execute the tool locally
        tool_output = ""
        if tool_name == "get_suspicious_players":
            tool_output = get_suspicious_players()
        elif tool_name == "get_cluster_stats":
            tool_output = get_cluster_stats()
        else:
            tool_output = "Error: Unknown tool"
        
        # Pass the result back to Gemini
        response = chat.send_message(
            Part.from_function_response(
                name=tool_name,
                response={"content": tool_output}
            )
        )
    
    # 3. Final Answer
    print(f"🤖 Agent: {response.text}")

# --- Run Simulation ---
if __name__ == "__main__":
    query_agent("Did our anomaly detection system catch anyone today?")
    query_agent("Can you explain why Bugged_Player_001 was flagged? What makes them different from a normal player?")

