from google.cloud import aiplatform
from google.cloud import storage
import os

# ==========================================
# CONFIGURATION
# ==========================================
project_id = "cloud-sa-ml"   # ⚠️ REPLACE
bucket_name = "eve-online-model-bucket" # ⚠️ REPLACE (Must exist!)
location = "us-central1"
model_display_name = "game_health_autoencoder_v1"
local_model_path = "game_health_autoencoder"
# ==========================================

def upload_folder_to_gcs(bucket_name, source_folder, destination_blob_prefix):
    """Uploads a local directory to GCS."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    print(f"⬆️ Uploading '{source_folder}' to gs://{bucket_name}/{destination_blob_prefix}...")
    
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, source_folder)
            blob_path = os.path.join(destination_blob_prefix, relative_path)
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)

    return f"gs://{bucket_name}/{destination_blob_prefix}"

def deploy_to_vertex():
    aiplatform.init(project=project_id, location=location)

    # 1. Upload Artifacts to GCS
    gcs_model_uri = upload_folder_to_gcs(bucket_name, local_model_path, "models/game_health")

    # 2. Register Model in Vertex AI
    print("\n®️ Registering model in Vertex AI...")
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=gcs_model_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    )
    print(f"✅ Model Uploaded: {model.resource_name}")

    # 3. Create Endpoint
    print("\n🔌 Creating Endpoint (This takes ~1-2 mins)...")
    endpoint = aiplatform.Endpoint.create(display_name=f"{model_display_name}_endpoint")
    print(f"✅ Endpoint Created: {endpoint.resource_name}")

    # 4. Deploy Model
    print("\n🚀 Deploying Model to Endpoint (This takes ~10-15 mins)...")
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )
    
    print("\n🎉 DEPLOYMENT COMPLETE!")
    print("------------------------------------------------")
    print(f"ENDPOINT ID: {endpoint.name}")
    print("------------------------------------------------")
    print(f"👉 Copy this ID into 'agent_realtime_sec.py' as 'endpoint_id'")

if __name__ == "__main__":
    deploy_to_vertex()

