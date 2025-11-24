-- ==========================================================
-- STEP 2: MODEL TRAINING (Unsupervised K-Means)
-- ==========================================================
-- We train the model to find "Normal" clusters of behavior.
-- We do NOT need labeled data (e.g., "This is a bot"). 
-- We just ask the model to group similar behaviors together.
-- 
-- num_clusters=5: We assume ~5 standard behavior types (e.g., Trading, Mining, Idle, etc.)
-- standardize_features=TRUE: Critical because Quantity (millions) and Players (tens) have different scales.

CREATE OR REPLACE MODEL `eve_data_demo.behavior_anomaly_model`
OPTIONS(model_type='kmeans', num_clusters=5, standardize_features = TRUE) AS
SELECT
  transaction_count,
  total_quantity,
  unique_players,
  avg_price
FROM
  `eve_data_demo.stats_per_minute`;

-- ==========================================================
-- STEP 3: CLUSTER ANALYSIS (The "Smoking Gun")
-- ==========================================================
-- Before banning, we must identify WHICH cluster is the "Bot" cluster.
-- We pivot the ML.CENTROIDS output to see the profile of each group.
-- Look for: High Transaction Count + Low Unique Players (1.0).

SELECT
  centroid_id,
  ROUND(MAX(IF(feature = 'transaction_count', numerical_value, NULL)), 1) as avg_transactions_per_min,
  ROUND(MAX(IF(feature = 'unique_players', numerical_value, NULL)), 1) as avg_unique_players,
  ROUND(MAX(IF(feature = 'total_quantity', numerical_value, NULL)), 0) as avg_quantity,
  ROUND(MAX(IF(feature = 'avg_price', numerical_value, NULL)), 2) as avg_price
FROM
  ML.CENTROIDS(MODEL `eve_data_demo.behavior_anomaly_model`)
GROUP BY
  centroid_id
ORDER BY
  avg_transactions_per_min DESC;

-- ==========================================================
-- STEP 4: THE BAN LIST (Applying the Threshold)
-- ==========================================================
-- TUNING UPDATE: Raised threshold to 2000 APM.
-- 1,000 - 1,900 APM ranges are now considered "High Skill / Multibox Industry"
-- which is allowed in EVE Online rules. We only ban physically impossible speeds.

SELECT
  player_id,
  item_id,
  TIMESTAMP_TRUNC(event_timestamp, MINUTE) as minute_window,
  COUNT(*) as actions_per_minute
FROM
  `eve_data_demo.game_events`
GROUP BY
  player_id, item_id, minute_window
HAVING
  actions_per_minute > 2000 -- Threshold Tuned for False Positives (Industrial Corp Protection)
ORDER BY
  actions_per_minute DESC;

-- ==========================================================
-- OPTIONAL: RAW ANOMALY SCORING
-- ==========================================================
-- If you want to see the raw "Distance" score for every minute
-- (Useful for finding new, unknown exploit types)

SELECT
  *
FROM
  ML.DETECT_ANOMALIES(MODEL `eve_data_demo.behavior_anomaly_model`,
                      STRUCT(0.05 AS contamination),
                      (SELECT * FROM `eve_data_demo.stats_per_minute`))
ORDER BY
  normalized_distance DESC
LIMIT 20;
