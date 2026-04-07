#!/bin/bash
set -e

PROJECT_ID="good-doctor-titan-389604"
REGION="asia-southeast2"
AR_REPO="ml-containers"
CLOUD_RUN_SERVICE_NAME="gd-recommendation"

set +e
gcloud artifacts repositories describe "${AR_REPO}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null
REPO_EXISTS=$?
set -e

if [ $REPO_EXISTS -ne 0 ]; then
    gcloud artifacts repositories create "${AR_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --project="${PROJECT_ID}" \
        --description="ML model containers"
fi

echo "Artifact Registry repository: ${AR_REPO}"

AR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${CLOUD_RUN_SERVICE_NAME}"

echo "Building container image: ${AR_IMAGE}:latest"
echo "--------------------------------------------------"

# Assuming you are running this from the directory containing the Dockerfile
gcloud builds submit \
    --tag "${AR_IMAGE}:latest" \
    --project="${PROJECT_ID}"

echo "Deploying ${CLOUD_RUN_SERVICE_NAME} to Cloud Run..."
echo "--------------------------------------------------"

gcloud run deploy "${CLOUD_RUN_SERVICE_NAME}" \
    --image="${AR_IMAGE}:latest" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300 \
    --concurrency=80 \
    --allow-unauthenticated