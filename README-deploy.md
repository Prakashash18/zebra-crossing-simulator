# Deploying the Zebra Crossing Simulator to Google Cloud Run

This guide provides step-by-step instructions for deploying the Zebra Crossing Simulator application to Google Cloud Run, both manually and using continuous deployment with Cloud Build.

## Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured
2. Docker installed locally (for local testing)
3. A Google Cloud Platform account with billing enabled
4. The following APIs enabled in your GCP project:
   - Cloud Run API
   - Container Registry API
   - Cloud Build API (for automated deployments)

## Manual Deployment

### 1. Set up your environment

```bash
# Set your project ID
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Set default region
gcloud config set run/region us-central1
```

### 2. Build the Docker image locally

```bash
# Build the image
docker build -t gcr.io/$PROJECT_ID/zebra-crossing-simulator .

# Test the image locally (optional)
docker run -p 8080:8080 gcr.io/$PROJECT_ID/zebra-crossing-simulator
```

### 3. Push the image to Google Container Registry

```bash
# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker

# Push the image
docker push gcr.io/$PROJECT_ID/zebra-crossing-simulator
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy zebra-crossing-simulator \
  --image gcr.io/$PROJECT_ID/zebra-crossing-simulator \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 3600 \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10
```

## Automated Deployment with Cloud Build

### 1. Enable required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 2. Grant necessary permissions

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# Grant Cloud Run Admin role to the Cloud Build service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/run.admin"

# Grant IAM Service Account User role
gcloud iam service-accounts add-iam-policy-binding \
  $PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### 3. Setup Cloud Build Trigger

1. Go to Cloud Build > Triggers in the Google Cloud Console
2. Click "Create Trigger"
3. Connect your repository (GitHub, Bitbucket, or Cloud Source Repositories)
4. Configure the trigger:
   - Name: `deploy-zebra-crossing-simulator`
   - Event: Push to a branch
   - Source: Your repository and branch (e.g., `^main$`)
   - Build configuration: Cloud Build configuration file
   - Cloud Build configuration file location: `cloudbuild.yaml`
5. Click "Create"

### 4. Push your code

Push your code to the connected repository to trigger the build and deployment.

## Troubleshooting

- **Build Failures**: Check the Cloud Build logs for errors
- **Deployment Issues**: Verify your service account has the correct permissions
- **Application Errors**: Check the Cloud Run logs for application-specific errors

## Important Notes

- The application expects model files to be present in the Docker image
- Ensure your `requirements.txt` includes all dependencies
- The deployed service will use TLS by default (https)
- By default, Cloud Run scales to zero when not in use, which means the first request may experience cold start latency
- Memory settings (2GB) are optimized for running the pose detection model

## Monitoring and Maintenance

- Monitor your application using Cloud Monitoring
- Set up alerts for errors or high latency
- Regularly update dependencies in requirements.txt for security
- Check Cloud Run logs for debugging information 