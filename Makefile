# Makefile for MLOps Pipeline & Deployment

IMAGE_TAG := latest
APP_IMAGE := mlops-wine-app:$(IMAGE_TAG)
KUBE_NAMESPACE ?= default
DAGSHUB_USERNAME ?= hemantku1990
DAGSHUB_TOKEN ?= $(shell echo $$DAGSHUB_TOKEN)

.PHONY: all build deploy-dev deploy-qa deploy-stage deploy-prod clean logs lint

all: build deploy-dev

# --- Build ---
build:
	@echo "Building App Image..."
	docker build -t $(APP_IMAGE) -f docker/Dockerfile.app .
	@echo "Building Drift Server Image..."
	$(MAKE) build-drift-server

# --- Drift Detection ---

build-drift-trainer:
	docker build -t mlops-drift-trainer:latest -f docker/Dockerfile.drift .

train-drift-detector: build-drift-trainer
	mkdir -p models/drift_detector
	docker run --rm \
		-v $(PWD)/models/drift_detector:/app/models/drift_detector \
		mlops-drift-trainer:latest

build-drift-server: train-drift-detector
	docker build -t mlops-drift-server:latest -f docker/Dockerfile.drift-server .

# --- Environments ---

# DEV: Deploy locally with Minikube/Kind, using local secrets
deploy-dev: check-env
	@echo "Deploying to DEV environment..."
	$(MAKE) deploy-common ENV=dev REPLICAS=1

# QA: Simulate QA env (can be same cluster, different namespace)
deploy-qa: check-env
	@echo "Deploying to QA environment..."
	kubectl create namespace qa || true
	$(MAKE) deploy-common ENV=qa KUBE_NAMESPACE=qa REPLICAS=2

# STAGE: Pre-prod
deploy-stage: check-env
	@echo "Deploying to STAGE environment..."
	kubectl create namespace stage || true
	$(MAKE) deploy-common ENV=stage KUBE_NAMESPACE=stage REPLICAS=2

# PROD: Production deployment
deploy-prod: check-env
	@echo "Deploying to PROD environment..."
	kubectl create namespace prod || true
	$(MAKE) deploy-common ENV=prod KUBE_NAMESPACE=prod REPLICAS=3

# --- Deployment Logic ---

deploy-common:
	@echo "Deploying Stack to $(KUBE_NAMESPACE)..."
	
	# 1. create secrets (substituting env vars)
	cat k8s/local/secrets.yaml | envsubst | kubectl apply -n $(KUBE_NAMESPACE) -f -
	
	# 2. Get Run ID from local run_info.json (Simulating CI/CD fetching this artifact)
	$(eval MLFLOW_RUN_ID := $(shell cat run_info.json | grep -o '"run_id": "[^"]*"' | cut -d'"' -f4))
	@echo "Using MLflow Run ID: $(MLFLOW_RUN_ID)"
	
	# 3. Apply Seldon Deployment (Substitute Run ID)
	sed 's|$${MLFLOW_RUN_ID}|$(MLFLOW_RUN_ID)|g' k8s/local/production-deployment.yaml | \
	sed 's|namespace: default|namespace: $(KUBE_NAMESPACE)|g' | \
	sed 's|wine-drift-detector.default|wine-drift-detector.$(KUBE_NAMESPACE)|g' | \
	sed 's|wine-model-default.default|wine-model-default.$(KUBE_NAMESPACE)|g' | \
	kubectl apply -n $(KUBE_NAMESPACE) -f -
	
	@echo "Waiting for Seldon Deployment to be ready..."
	kubectl rollout status deployment/wine-model-production-0-ensemble-model -n $(KUBE_NAMESPACE) --timeout=300s
	kubectl rollout status deployment/wine-drift-detector-default-0-detector -n $(KUBE_NAMESPACE) --timeout=300s
	
	@echo "Waiting for App..."
	kubectl rollout status deployment/wine-app -n $(KUBE_NAMESPACE) --timeout=120s
	
	@echo "Deployment Complete! Service URL: http://localhost:80/predict (via LoadBalancer) or Port Forward."

# --- Utilities ---

check-env:
ifndef DAGSHUB_TOKEN
	$(error DAGSHUB_TOKEN is undefined. Please export it)
endif

# Run DVC Pipeline
pipeline:
	dvc repro

# Run linting
lint:
	flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Test Inference
test-api:
	@echo "Testing prediction endpoint..."
	curl -X POST "http://localhost:80/predict" \
		-H "Content-Type: application/json" \
		-d '{"alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2, "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76, "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38, "hue": 1.05, "od280_od315_of_diluted_wines": 3.4, "proline": 1050.0}'

clean:
	@echo "Cleaning up..."
	kubectl delete -f k8s/local/production-deployment.yaml --ignore-not-found -n default
	kubectl delete -f k8s/local/production-deployment.yaml --ignore-not-found -n qa
	kubectl delete -f k8s/local/production-deployment.yaml --ignore-not-found -n stage
	kubectl delete -f k8s/local/production-deployment.yaml --ignore-not-found -n prod
	kubectl delete secret dagshub-secret --ignore-not-found -n default


