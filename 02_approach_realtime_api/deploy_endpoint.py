import argparse
import os
import subprocess
import time
from google.cloud import aiplatform
from google.cloud import storage

# --- Configuration ---
REGION = "us-central1"
MODEL_DISPLAY_NAME = "eve-game-security-ae"
ENDPOINT_DISPLAY_NAME = "eve-live-detect-endpoint"
ARTIFACT_DIR = "model_artifacts" # Local folder from step 1
MODEL_BUCKET = "eve-online-model-bucket"

def upload_local_directory_to_gcs(local_path, bucket_name, gcs_path):
    """Recursively uploads a directory to GCS."""
    storage_client = storage.Client()
    
    # Create bucket if it doesn't exist
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        print(f"🪣 Creating bucket: {bucket_name}...")
        bucket = storage_client.create_bucket(bucket_name, location=REGION)

    print(f"📤 Uploading {local_path} to gs://{bucket_name}/{gcs_path}...")
    
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Calculate relative path to maintain structure
            relative_path = os.path.relpath(local_file_path, local_path)
            blob_path = os.path.join(gcs_path, relative_path)
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            
    print("✅ Upload complete.")
    return f"gs://{bucket_name}/{gcs_path}"

def grant_bucket_access_to_vertex_sa(project_id, bucket_name):
    """
    Grants the Vertex AI Service Agent permission to read from the SPECIFIC GCS bucket.
    Uses Bucket-Level IAM via Python Client to avoid Project-Level conditional policy conflicts.
    """
    print(f"🔑 Granting Storage Object Viewer on gs://{bucket_name} to Vertex AI Service Agent...")
    try:
        # 1. Get Project Number (needed to construct SA email)
        project_number = subprocess.check_output(
            f"gcloud projects describe {project_id} --format='value(projectNumber)'", 
            shell=True
        ).decode().strip()
        
        vertex_sa = f"service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
        
        # 2. Update Bucket IAM Policy (Native Python, no gcloud CLI)
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        policy = bucket.get_iam_policy(requested_policy_version=3)
        
        role = "roles/storage.objectViewer"
        member = f"serviceAccount:{vertex_sa}"
        
        # Add binding if not present
        binding_exists = False
        for binding in policy.bindings:
            if binding["role"] == role and member in binding["members"]:
                binding_exists = True
                break
        
        if not binding_exists:
            policy.bindings.append({"role": role, "members": [member]})
            bucket.set_iam_policy(policy)
            print(f"✅ Permission granted to {vertex_sa} on bucket {bucket_name}")
        else:
            print(f"ℹ️ Permission already exists for {vertex_sa}")
            
    except Exception as e:
        print(f"⚠️ Could not automatically grant IAM permissions: {e}")
        print("Please manually ensure the Vertex AI Service Agent has 'Storage Object Viewer' on the bucket.")

def deploy_to_vertex(project_id, staging_bucket):
    total_start_time = time.time()

    # 1. Upload Model Artifacts (Creates bucket if needed)
    print(f"\n[⏱️ {time.strftime('%H:%M:%S')}] Step 1: Uploading Model Artifacts...")
    step_start = time.time()
    
    local_model_path = os.path.join(ARTIFACT_DIR, "saved_model")
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"❌ Could not find {local_model_path}. Did you run training?")
        
    artifact_uri = upload_local_directory_to_gcs(
        local_model_path, 
        MODEL_BUCKET, 
        "game_security_model"
    )
    print(f"   Duration: {time.time() - step_start:.1f}s")

    # 2. Fix IAM Permissions (Bucket Level)
    print(f"\n[⏱️ {time.strftime('%H:%M:%S')}] Step 2: Fixing IAM Permissions...")
    step_start = time.time()
    # Must happen AFTER bucket creation
    grant_bucket_access_to_vertex_sa(project_id, MODEL_BUCKET)
    print(f"   Duration: {time.time() - step_start:.1f}s")

    # 3. Initialize Vertex AI
    print(f"\n[⏱️ {time.strftime('%H:%M:%S')}] Step 3: Initializing Vertex AI SDK...")
    step_start = time.time()
    aiplatform.init(project=project_id, location=REGION, staging_bucket=staging_bucket)
    print(f"   Duration: {time.time() - step_start:.1f}s")

    print(f"🚀 Starting Vertex AI Deployment for {project_id}...")

    # 4. Upload Model
    # We use the pre-built TensorFlow container
    print(f"\n[⏱️ {time.strftime('%H:%M:%S')}] Step 4: Registering model in Vertex AI Model Registry...")
    step_start = time.time()
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    )
    print(f"✅ Model Registered. Duration: {time.time() - step_start:.1f}s")
    
    # 5. Create Endpoint
    print(f"\n[⏱️ {time.strftime('%H:%M:%S')}] Step 5: Creating Endpoint (this takes ~5-10 mins)...")
    step_start = time.time()
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
    print(f"✅ Endpoint Created. Duration: {time.time() - step_start:.1f}s")

    # 6. Deploy Model to Endpoint
    print(f"\n[⏱️ {time.strftime('%H:%M:%S')}] Step 6: Deploying Model to Endpoint (this also takes time)...")
    step_start = time.time()
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )
    print(f"✅ Model Deployed. Duration: {time.time() - step_start:.1f}s")

    total_duration = time.time() - total_start_time
    print(f"\n✅ Deployment Complete! Total Time: {int(total_duration // 60)}m {int(total_duration % 60)}s")
    print(f"Endpoint ID: {endpoint.name}")
    print(f"Resource Name: {endpoint.resource_name}")
    
    # Save Endpoint ID for the Agent to use
    with open("endpoint_config.txt", "w") as f:
        f.write(endpoint.resource_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--staging_bucket", required=True)
    args = parser.parse_args()

    deploy_to_vertex(args.project_id, args.staging_bucket)

