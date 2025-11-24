-- ==========================================================
-- STEP 1: FEATURE ENGINEERING (The "Static Fact" Layer)
-- ==========================================================
-- We cannot detect anomalies on raw event streams because they lack context.
-- We must aggregate the stream into "Behavior Vectors" (Stats per Minute).
--
-- Features Created:
-- 1. transaction_count (Velocity): How fast are they clicking? (Bots click fast)
-- 2. total_quantity (Volume): Are they moving massive amounts of items?
-- 3. unique_players (Network): Is it a natural market (many players) or a bot farm (1 player)?
-- 4. avg_price (Value): Are they dumping items for 1 ISK?

CREATE OR REPLACE VIEW `eve_data_demo.stats_per_minute` AS
SELECT
  item_id,
  location_id,
  TIMESTAMP_TRUNC(event_timestamp, MINUTE) as minute_window,
  
  -- FEATURE 1: Velocity (Spamming)
  -- How many distinct interactions per minute?
  COUNT(*) as transaction_count,

  -- FEATURE 2: Volume (Hoarding)
  -- Total quantity of items moved
  SUM(quantity) as total_quantity,

  -- FEATURE 3: Networking (Botting?)
  -- How many unique players touched this item in this window?
  -- A healthy market has many players. A bot farm has 1.
  COUNT(DISTINCT player_id) as unique_players,

  -- FEATURE 4: Value Density
  -- The average price traded. Useful for finding RMT (Real Money Trading) dumps.
  AVG(price_per_item) as avg_price

FROM
  `eve_data_demo.game_events`
GROUP BY
  item_id, location_id, TIMESTAMP_TRUNC(event_timestamp, MINUTE);

