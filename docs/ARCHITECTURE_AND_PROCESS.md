# MLOps System Architecture & Process Documentation

This document details the end-to-end MLOps architecture, Software Development Life Cycle (SDLC), CI/CD workflows, and branching strategies implemented for the Wine Quality Prediction system.

## 1. High-Level Architecture

The system follows a **Hybrid Deployment Pattern**, bridging local development flexibility with a robust Kubernetes-based production environment using Seldon Core and NVIDIA Triton Inference Server.

### System Components Flow

```mermaid
graph TD
    subgraph DevEnv ["Development Environment (Local/Docker)"]
        DS[Data Scientist] -->|Trains| NB[Jupyter/Scripts]
        NB -->|Logs Exp| MLflow_Dev[Local MLflow]
        NB -->|Saves| Local_Model[Local Artifacts]
        App_Dev["FastAPI App (Dev Mode)"] -->|Loads| Local_Model
    end

    subgraph CI_CD ["CI/CD Pipeline (GitHub Actions)"]
        Git[GitHub Repo] -->|Trigger| CI[CI Workflow]
        CI -->|1. Data Provision| DVC["DVC / Data Gen"]
        CI -->|2. Train & Eval| Trainer[Training Job]
        Trainer -->|Logs Run| MLflow_Remote[Remote MLflow]
        Trainer -->|Exports| ONNX[ONNX Model]
        CI -->|3. Build| Docker[Docker Build]
        CI -->|4. Deploy| K8s_Manifest[K8s Manifests]
    end

    subgraph ProdEnv ["Production Environment (Kubernetes/Kind)"]
        Ingress["Ingress / LoadBalancer"] -->|HTTP/REST| WineApp_Prod["FastAPI App (Proxy)"]
        
        subgraph SeldonDep ["Seldon Core Deployment"]
            WineApp_Prod -->|Seldon Protocol| Seldon_Ens[Seldon Ensemble]
            
            subgraph Triton ["Triton Inference Server"]
                Seldon_Ens -->|gRPC/HTTP| Triton_Ens[Ensemble Model]
                Triton_Ens -->|Step 1| Pre["Preprocessing (Python)"]
                Triton_Ens -->|Step 2| Model["ElasticNet (ONNX)"]
                Triton_Ens -->|Step 3| Post["Postprocessing (Python)"]
            end
            
            Seldon_Ens -.->|Async Logging| Drift_Svc[Drift Detector Service]
        end
        
        Drift_Svc -->|Metrics| Prom[Prometheus]
        Drift_Svc -->|Alerts| Alert[AlertManager]
    end

    DS -->|Push Code| Git
```

---

## 2. Software Development Life Cycle (SDLC) Phases

The lifecycle is divided into 5 distinct phases, ensuring code and model quality before production deployment.

| Phase | Activity | Tools/Tech | Outcome |
| :--- | :--- | :--- | :--- |
| **1. Data Ops** | Data generation, cleaning, versioning. | DVC, Pandas, Python | Versioned Dataset (`.dvc` files) |
| **2. Model Dev** | Feature engineering, training, hyperparam tuning. | Jupyter, Scikit-Learn, MLflow | Trained Model Artifacts, Experiment Logs |
| **3. Integration** | Packaging model code, creating Docker containers, unit tests. | Docker, Pytest, Makefile | Docker Images, Test Reports |
| **4. Deployment** | Deploying to Staging/Prod, configuring Inference Graph. | Kubernetes, Seldon Core, Triton | Running Inference Service |
| **5. Monitoring** | Tracking drift, latency, and errors. Feedback loop. | Alibi Detect, Prometheus, Grafana | Live Metrics, Drift Alerts |

---

## 3. CI/CD & Drift Feedback Loop

### CI/CD Workflow (GitHub Actions)

The pipeline automates the path from code commit to deployed service.

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Git as GitHub
    participant Action as GitHub Action
    participant MLflow as MLflow Tracking
    participant Reg as Container Registry
    participant K8s as Kubernetes Cluster

    Dev->>Git: Push to 'main'
    Git->>Action: Trigger Pipeline
    
    rect rgb(240, 248, 255)
        Note right of Action: 1. Continuous Integration
        Action->>Action: Checkout Code
        Action->>Action: Install Deps & Lint
        Action->>Action: Pull Data (DVC) / Generate
        Action->>Action: Run Unit Tests (Pytest)
    end
    
    rect rgb(255, 240, 245)
        Note right of Action: 2. Continuous Training (CT)
        Action->>Action: Run Train Script
        Action->>MLflow: Log Metrics, Params, Model (ONNX)
        MLflow-->>Action: Return Run ID & Artifact URI
    end
    
    rect rgb(240, 255, 240)
        Note right of Action: 3. Continuous Delivery (CD)
        Action->>Reg: Build & Push App Image
        Action->>K8s: Update Manifests (GitOps/Helm)
        K8s->>MLflow: InitContainer Downloads Model (Run ID)
        K8s->>K8s: Rollout Seldon Deployment
    end
```

### Drift Detection Feedback Loop

This loop ensures the model remains valid over time.

1.  **Inference**: User sends data to `WineApp`.
2.  **Logging**: `Seldon Core` asynchronously forwards the request payload to `Drift Detector`.
3.  **Analysis**: `Alibi Detect` (running in `Drift Detector`) compares the live batch against the Reference Training Data.
4.  **Metric**: If drift is detected (p-value < threshold), `seldon_metric_drift_found` is set to 1.
5.  **Alert**: Prometheus scrapes the metric; AlertManager triggers notification.
6.  **Action**: Triggers a **Retraining Pipeline** (CT) on new data.

```mermaid
graph LR
    User[User Request] --> Model[Seldon Model]
    Model --> Response[Prediction]
    Model -.->|Payload| Drift[Drift Detector]
    Drift -->|Calc| Metric[Prometheus Metric]
    Metric -->|Threshold Violated| Alert[Alert System]
    Alert -->|Trigger| Retrain[Retraining Job]
    Retrain -->|Update| Model
```

---

## 4. Branching Strategy

We utilize a **GitFlow-inspired** strategy adapted for MLOps to manage code and model versions stability.

### Branches

1.  **`main` (Production)**
    *   **Purpose**: Stable, deployable code. Reflects what is running in the Production environment.
    *   **Protection**: Protected branch. Requires Pull Request (PR) approval and passing CI checks.
    *   **Deployment**: Automatically deploys to Production Cluster upon merge.

2.  **`develop` (Integration/Staging)**
    *   **Purpose**: Integration branch for features. Reflects the Staging environment.
    *   **Deployment**: Automatically deploys to Staging Cluster (e.g., `namespace: staging`) for end-to-end testing.

3.  **`feature/*` (Feature Development)**
    *   **Purpose**: Short-lived branches for new features, experiments, or model improvements.
    *   **Naming**: `feature/add-drift-detection`, `feature/xgboost-model`.
    *   **Workflow**: Branch from `develop`, work locally, PR to `develop`.

4.  **`experiment/*` (Data Science Experiments)**
    *   **Purpose**: Sandbox for Data Scientists to try radical model changes without affecting engineering code.
    *   **Note**: Often these don't merge directly but result in updated training scripts or parameters moved to `feature` branches.

### Workflow Stages

#### Stage 1: Feature Development
*   **Action**: Create `feature/new-model` from `develop`.
*   **Work**: Modify `train.py`, update `parameters.yaml`.
*   **Test**: Run `make test` locally.

#### Stage 2: Pull Request & Review
*   **Action**: Open PR `feature/new-model` -> `develop`.
*   **CI Checks**:
    *   Code Linting.
    *   Unit Tests.
    *   *Small-scale* training run to verify script integrity.
*   **Review**: Peers review code and initial model metrics.

#### Stage 3: Staging Deployment
*   **Action**: Merge to `develop`.
*   **CD**: Deploys to Kubernetes `dev`/`staging` namespace.
*   **Verification**: Run integration tests and drift simulations.

#### Stage 4: Production Release
*   **Action**: Create Release PR `develop` -> `main`.
*   **CD**: Deploys to Kubernetes `prod` namespace.
*   **Tagging**: Semantic version tag (e.g., `v1.2.0`) created for rollback capability.

### Visual Guide

```mermaid
gitGraph
    commit
    branch develop
    checkout develop
    commit
    branch feature/drift
    checkout feature/drift
    commit id: "Add Drift Code"
    commit id: "Update Manifests"
    checkout develop
    merge feature/drift
    branch release/v1.0
    checkout release/v1.0
    commit id: "Bump Version"
    checkout main
    merge release/v1.0 tag: "v1.0.0"
    checkout develop
    merge release/v1.0
```
