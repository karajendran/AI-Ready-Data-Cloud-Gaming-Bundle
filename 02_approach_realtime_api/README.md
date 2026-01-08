# Approach 2: Real-Time Anomaly Detection (Vertex AI)

This module demonstrates Production ML Engineering. Instead of batch SQL queries, we train a TensorFlow Autoencoder and deploy it as a low-latency microservice on Vertex AI.

## Architecture
1. Training: Fetches clean historical data from BigQuery and trains an Autoencoder locally (or in Cloud Build).

2. Deployment: Uploads the model to Vertex AI Model Registry and deploys it to an Endpoint.

3. Inference: The Agent fetches live player stats and "pings" the endpoint.
 - Low Error: Normal Behavior.
 - High Error: Anomaly Detected.

## Quickstart

We provide a single script to handle training, deployment, and execution.

### Prerequisites

- You must have run the Master Setup `(../infra_provisioning/setup_demo.sh)` first to generate the BigQuery data.

- Ensure you have a staging bucket (created by the setup script).

### Run the Demo

```bash
chmod +x run_demo.sh
./run_demo.sh <PROJECT_ID> <STAGING_BUCKET_NAME>
```

### Example:

```bash
./run_demo.sh my-project-id my-project-id-dataflow-staging
```

## Logical Flow

1. Train: The Autoencoder learns to reconstruct "Normal" traffic (e.g., trading, mining).

2. Threshold: We calculate the reconstruction error on the training set and set the 99th percentile as the "Anomaly Threshold."

3. Detect: When a player like `Bugged_Player_001` performs 2,500 actions/min, the model fails to reconstruct this pattern, resulting in a massive error score that triggers the Agent.


