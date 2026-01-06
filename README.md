# MLOps End-to-End Demo

This project demonstrates a complete, production-grade MLOps pipeline including data generation, model training, tracking with MLflow, containerization, and deployment via Seldon Core.

## Project Structure

```
.
├── .github/workflows   # CI/CD Pipelines (GitOps Pattern)
├── data/               # Data directory (tracked by DVC)
├── k8s/                # Kubernetes Manifests (Seldon Core)
├── models/             # Trained models (Local Dev only)
├── src/
│   ├── app/            # FastAPI Inference Application (Hybrid: Local/Seldon)
│   └── model/          # Model Training Scripts
├── Dockerfile          # App Container (Model-agnostic)
├── docker-compose.yml  # Local Development Stack (MLflow + App)
├── requirements.txt    # Python Dependencies
└── README.md
```

## Architecture

This project implements a **Hybrid Deployment Pattern**:

1.  **Development**: The Inference App loads the model directly from the local file system (Scikit-Learn).
2.  **Production (Triton)**: The Inference App acts as a proxy/gateway. It forwards requests to an **NVIDIA Triton Inference Server** which orchestrates an ensemble pipeline:
    *   **Preprocessing (Python)**: Validates and orders inputs.
    *   **Inference (ONNX)**: Runs the ElasticNet model.
    *   **Postprocessing (Python)**: Formats the output.

## Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Git
- (Optional) Kubernetes Cluster (Minikube/Kind) for Seldon Core deployment

## Local Setup & Demo

### 1. Data Generation & Training

First, install dependencies and generate the initial dataset and model locally.

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Generate dummy data
python src/model/data_gen.py

# Train model
# - Logs to local MLflow (http://localhost:5001)
# - Exports model to ONNX format
# - Saves artifact locally to models/wine_model/
python src/model/train.py
```

### 2. Prepare Triton Model Repository

The training script saves the ONNX model. We need to ensure it's in the correct place for Triton (already handled by the repository structure, but for reference):

```bash
# Copy ONNX model to Triton repo (if not already done by script/CI)
cp models/wine_model/model.onnx model_repository/wine_model/1/model.onnx
```

### 3. Run with Docker Compose

Spin up the MLflow tracking server, Triton Inference Server, and the Inference App.

```bash
docker-compose up -d --build
```

- **MLflow UI**: [http://localhost:5001](http://localhost:5001)
- **Triton Metrics**: [http://localhost:8002/metrics](http://localhost:8002/metrics)
- **Inference API**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Test Inference

The FastAPI app automatically detects the Triton server and proxies requests to it.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "alcohol": 12.8,
  "malic_acid": 2.0,
  "ash": 2.4,
  "alcalinity_of_ash": 20.0,
  "magnesium": 100.0,
  "total_phenols": 2.5,
  "flavanoids": 2.5,
  "nonflavanoid_phenols": 0.3,
  "proanthocyanins": 1.5,
  "color_intensity": 5.0,
  "hue": 1.0,
  "od280_od315_of_diluted_wines": 3.0,
  "proline": 800.0
}'
```

## CI/CD Pipeline (GitOps)

The `.github/workflows/main.yml` defines a production-grade pipeline:

1.  **Data Provisioning**:
    *   **DVC Pull**: Attempts to pull the latest versioned dataset from the configured remote storage (e.g., S3).
    *   **Fallback**: If no remote is configured (or credentials missing in this demo), it automatically generates synthetic data to ensure the pipeline succeeds.
2.  **Train & Register**:
    *   Trains the model using the provisioned data.
    *   Logs metrics and artifacts to MLflow.
    *   Exports `run_info.json` containing the specific Run ID and Artifact URI.
3.  **Build Image**:
    *   Builds the `mlops-wine-app` Docker image tagged with the Git SHA.
    *   **Crucial**: The model is *not* baked into this image.
3.  **Generate Manifest (GitOps)**:
    *   Downloads the `run_info.json`.
    *   Injects the specific `Artifact URI` and `Image Tag` into the `k8s/seldon-deployment.yaml` template.
    *   Generates a final `k8s/final-deployment.yaml` which can be committed to a config repo or applied directly.

## DVC (Data Version Control)

Data is tracked using DVC.

```bash
dvc init
dvc add data/wine_quality.csv
git add data/wine_quality.csv.dvc .gitignore
git commit -m "Add data versioning"
```

## Seldon Core Deployment

To deploy on Kubernetes:

1.  **Install Seldon Core Operator**.
2.  **Apply the Manifest**:
    The pipeline generates a fully resolved manifest. For manual testing, you can update `k8s/seldon-deployment.yaml` with your model URI and apply it.

    ```bash
    kubectl apply -f k8s/seldon-deployment.yaml
    ```

    The Inference App will detect it is running in production (via `SELDON_PREDICT_URL` env var) and proxy requests to the Seldon predictor.
