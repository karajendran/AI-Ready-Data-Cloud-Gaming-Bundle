# Data Cloud for Games: Game Health & Anomaly Detection
Project Status: Active (FY26 Sales Play)

Tech Stack: Google Cloud Pub/Sub, Dataflow, BigQuery, Vertex AI, Gemini 2.5 Flash
## 1. Project Overview
This repository demonstrates two architectural patterns for detecting economic exploits (e.g., item duplication, bot farms) in massive multiplayer games like EVE Online.
It showcases the "Data Cloud" journey from **Ingestion** (Dataflow) to **Intelligence** (ML) to **Action** (GenAI Agents).
## 2. Architecture Options
We provide two implementation approaches depending on your latency and complexity requirements.

| Feature   | Approach 1: Batch / SQL-First | Approach 2: Real-Time / API-First                     |
|-----------|-------------------------------|-------------------------------------------------------|
| Core Tech | BigQuery ML (K-Means)         | Vertex AI Endpoint (TensorFlow Autoencoder)           |
| Latency   | Minutes (Micro-batch)         | Sub-Second (Online Prediction)                        |
| Complexity| Low (Pure SQL)                | High (Python MLOps & Infrastructure)                  |
| Cost Model| Pay-per-query                 | Always-on Compute Node                                |
| Best For  | BI Teams, Analysts            | Production Engineering, Blocking Exploits Live        |
| Directory | 01_approach_batch_bqml/       | 02_approach_realtime_api/                             |

## 3. Setup & Installation

**Prerequisites:**
* Google Cloud Project with Billing enabled.
* APIs Enabled: Compute Engine, Vertex AI, BigQuery, Cloud Storage.
* Python 3.9+ environment.

**Installation:**

Clone the repo:
```
git clone https://github.com/karajendran/AI-Ready-Data-Cloud-Gaming-Bundle.git
cd AI-Ready-Data-Cloud-Gaming-Bundle
```

Install Dependencies (Shared):
```
pip install -r requirements.txt
```

## 4. Run Approach 1: Batch Analytics (BQML)

Located in: *01_approach_batch_bqml/*
Philosophy: "Bring the AI to the Data." We run unsupervised clustering directly inside the Data Warehouse.

### End-to-End Execution Guide

Follow these commands to provision, run the demo, and teardown resources.

``` bash
# 0. Configuration
export PROJECT_ID="your-project-id" 
export GCS_BUCKET_NAME="your-bucket-name"
```

``` bash
# 1. Setup (Infrastructure & Data)
gcloud auth application-default login
cd infra_provisioning/
chmod +x setup_demo.sh
./setup_demo.sh ${PROJECT_ID} ${GCS_BUCKET_NAME}
```

``` bash
# 2. Run Demo (Model & Agent)
cd ../01_approach_batch_bqml/
chmod +x run_demo.sh
./run_demo.sh ${PROJECT_ID}
```

``` bash
# 3. Teardown (Cleanup)
cd ../infra_provisioning/
chmod +x teardown_demo.sh
./teardown_demo.sh ${PROJECT_ID} ${GCS_BUCKET_NAME}
cd ..

```


## 5. Run Approach 2: Real-Time Detection (Vertex AI)

### End-to-End Execution Guide

Follow these commands to provision, run the demo, and teardown resources.

``` bash
# 0. Configuration
export PROJECT_ID="your-project-id" 
export GCS_BUCKET_NAME="your-bucket-name"
export DATAFLOW-STAGING_BUCKET_NAME="your-dataflow-staging-bucket-name"
```

``` bash
# 1. Setup (Infrastructure & Data)
gcloud auth application-default login
cd infra_provisioning/
chmod +x setup_demo.sh
./setup_demo.sh ${PROJECT_ID} ${GCS_BUCKET_NAME}
```

``` bash
# 2. Run Demo (Model & Agent)
cd ../02_approach_realtime_api/
chmod +x run_demo.sh
./run_demo.sh <PROJECT_ID> <DATAFLOW-STAGING_BUCKET_NAME>
```

``` bash
# 3. Teardown (Cleanup)
cd ../infra_provisioning/
chmod +x teardown_demo.sh
./teardown_demo.sh ${PROJECT_ID} ${GCS_BUCKET_NAME}
cd ..

```

## 6. Troubleshooting
* Error: 404 NotFound or Dataset not found
    * Fix: Ensure your project_id is correct and you have run the SQL in 00_common_data_engineering to create the views.
* Error: avg_price is null (Approach 2)
    * Fix: The SQL query inside agent_realtime_sec.py must use COALESCE(AVG(price), 0) to handle items with no market history.
* Deployment Timeout:
    * Context: deploy_endpoint.py can take 15-20 minutes to provision infrastructure. Do not interrupt the script.



