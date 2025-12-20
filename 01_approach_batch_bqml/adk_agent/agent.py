import os
from google.cloud import bigquery
from google.adk.agents.llm_agent import Agent

# Initialize BigQuery Client
# Ensure PROJECT_ID is set in your environment or passed explicitly
project_id = os.getenv("GOOGLE_CLOUD_PROJECT") 
if not project_id:
    # Fallback or error if not set
    print("⚠️ GOOGLE_CLOUD_PROJECT env var not set. Tools may fail.")
    project_id = "accelerated-platforms-dev" 

bq_client = bigquery.Client(project=project_id)

# ==========================================
# 1. Define Tools
# ==========================================
def get_suspicious_players() -> str:
    """
    Queries the game database to find players with abnormally high actions per minute.
    Returns a list of player IDs and their metrics.
    """
    print("\n⚡ [ADK Tool] Executing: get_suspicious_players...")
    query = """
        SELECT
            player_id,
            item_id,
            CAST(TIMESTAMP_TRUNC(event_timestamp, MINUTE) as STRING) as time_window,
            COUNT(*) as actions_per_minute
        FROM
            `eve_data_demo.fact_game_events`
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
        if df.empty:
            return "No suspicious players found matching criteria (>2000 APM)."
        return df.to_json(orient='records')
    except Exception as e:
        return f"Error querying suspicious players: {str(e)}"

def get_cluster_stats() -> str:
    """
    Retrieves the K-Means clustering statistics (centroids) to establish the 'Normal' vs 'Bot' baseline.
    Returns the average APM (actions per minute) for each identified cluster.
    """
    print("\n⚡ [ADK Tool] Executing: get_cluster_stats...")
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
        return f"Error retrieving cluster stats: {str(e)}"

# ==========================================
# 2. Define the Root Agent
# ==========================================
# This variable 'root_agent' is what the 'adk' command looks for.
root_agent = Agent(
    model='gemini-2.0-flash-exp', 
    name='GameSecurityBot',
    description="Analyzes game telemetry for exploits.",
    instruction="""
    You are a Senior Game Security Analyst for EVE Online.
    Your goal is to detect and explain economic exploits using BigQuery data.
    
    PROTOCOLS:
    1. When asked about cheats, ALWAYS check 'get_suspicious_players' first.
    2. When asked for an explanation, ALWAYS compare the player's APM to the 'Normal' cluster APM from 'get_cluster_stats'.
    3. Be professional, concise, and cite the data (e.g. "2500 APM vs 60 APM").
    """,
    tools=[get_suspicious_players, get_cluster_stats]
)

