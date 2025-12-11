
## 🎮 Part 2: Data Generation

You have two options for generating data, depending on what you want to test.

### Option A: Real-Time Simulation (Best for Demos)

Use this to watch data flow through the pipeline (Pub/Sub -\> Dataflow -\> BigQuery) in real-time. It simulates 4 minutes of "chaotic normal" traffic followed by a 1-minute anomaly event.

**Run the Publisher:**

```bash
python3 data_publisher.py --project_id <YOUR_PROJECT_ID>
```

  * **Duration:** \~5 minutes.
  * **Volume:** \~2,000 events.
  * **View it:** Watch the "Dataflow" page in Cloud Console or query the table `fact_game_events` in real-time.

### Option B: 24h Historical Backfill (Best for Anomaly Detection)

Use this to instantly generate a full day's worth of data. This is required if you want to run the Anomaly Detection SQL queries, as they rely on 24-hour statistical baselines.

**1. Generate the JSONL file:**

```bash
python3 generate_24h_history.py --project_id <YOUR_PROJECT_ID>
```

  * **Output:** `eve_24h_history.jsonl` (contains \~50k events).

**2. Load into BigQuery:**

```bash
bq load \
 --source_format=NEWLINE_DELIMITED_JSON \
 --autodetect \
 eve_data_demo.fact_game_events \
 eve_24h_history.jsonl
```

