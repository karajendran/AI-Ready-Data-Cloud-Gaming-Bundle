## Common Foundation (Data Engineering)

Before running batch approach or real time approach, you must establish the Feature Store.
1. Open BigQuery Console.
2. Run `feature_store_setup.sql`.
    * Input: `eve_data_demo.game_events` (Raw Stream)
    * Output: `eve_data_demo.stats_per_minute` (Aggregated Behavioral Vectors)
