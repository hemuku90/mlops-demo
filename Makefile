# Makefile for Local Kubernetes Deployment

IMAGE_TAG := latest
APP_IMAGE := mlops-wine-app:$(IMAGE_TAG)
TRITON_IMAGE := mlops-triton:$(IMAGE_TAG)
KUBE_NAMESPACE ?= dev

.PHONY: all build deploy test clean logs

all: build deploy

# Build Docker images
build:
	@echo "Building App Image..."
	docker build -t $(APP_IMAGE) .
	@echo "Building Triton Image (with baked-in models)..."
	docker build -f Dockerfile.triton -t $(TRITON_IMAGE) .

# Load images into Kind (Optional - run 'make load-kind' if using Kind)
load-kind:
	@echo "Loading images into Kind cluster..."
	kind load docker-image $(APP_IMAGE)
	kind load docker-image $(TRITON_IMAGE)

# Deploy to Kubernetes (Triton)
deploy-triton:
	@echo "Deploying Triton Stack to namespace: $(KUBE_NAMESPACE)..."
	kubectl create namespace $(KUBE_NAMESPACE) || true
	kubectl apply -f k8s/local/triton-deployment.yaml -n $(KUBE_NAMESPACE)
	kubectl apply -f k8s/local/app-deployment.yaml -n $(KUBE_NAMESPACE)
	# Configure App for Triton
	kubectl set env deployment/wine-app TRITON_URL=triton-service:8000 SELDON_URL- -n $(KUBE_NAMESPACE)
	@echo "Waiting for pods..."
	kubectl wait --for=condition=ready pod -l app=triton-server --timeout=120s -n $(KUBE_NAMESPACE)
	kubectl wait --for=condition=ready pod -l app=wine-app --timeout=120s -n $(KUBE_NAMESPACE)

# Install Seldon Core via Helm
install-seldon:
	@echo "Installing Seldon Core Operator..."
	helm repo add datawire https://www.getambassador.io
	helm repo add seldonio https://storage.googleapis.com/seldon-charts
	helm repo update
	kubectl create namespace seldon-system || true
	helm install seldon-core seldonio/seldon-core-operator --namespace seldon-system --set usageMetrics.enabled=true --set istio.enabled=false
	@echo "Waiting for Seldon Operator..."
	kubectl rollout status deployment/seldon-controller-manager -n seldon-system

# Deploy to Kubernetes (Seldon)
deploy-seldon:
	@echo "Deploying Seldon Stack to namespace: $(KUBE_NAMESPACE)..."
	kubectl create namespace $(KUBE_NAMESPACE) || true
	
	# 1. Deploy MLflow (Storage)
	kubectl apply -f k8s/local/mlflow-deployment.yaml -n $(KUBE_NAMESPACE)
	
	# 2. Prepare Run ID for Seldon Model Download
	$(eval MLFLOW_RUN_ID := $(shell cat run_info.json | grep -o '"run_id": "[^"]*"' | cut -d'"' -f4))
	@echo "Deploying Seldon Model using Run ID: $(MLFLOW_RUN_ID)"
	
	# 3. Deploy Seldon Model (Substitute MLFLOW_RUN_ID)
	kubectl delete sdep wine-model -n $(KUBE_NAMESPACE) --ignore-not-found
	sed 's|$${MLFLOW_RUN_ID}|$(MLFLOW_RUN_ID)|g' k8s/local/seldon-deployment.yaml | kubectl apply -f - -n $(KUBE_NAMESPACE)
	
	# Wait for Seldon to create the Deployment object
	@echo "Waiting for Seldon Deployment object..."
	@for i in {1..30}; do \
		if kubectl get deployment wine-model-default-0-ensemble-model -n $(KUBE_NAMESPACE) > /dev/null 2>&1; then break; fi; \
		echo "Waiting for deployment creation..."; \
		sleep 2; \
	done
	
	# 4. Deploy App (Configured for Seldon)
	kubectl apply -f k8s/local/app-deployment.yaml -n $(KUBE_NAMESPACE)
	# Configure App for Seldon (Unset TRITON_URL, Set SELDON_URL)
	# Seldon Service URL: http://<sdep-name>-<predictor-name>.<namespace>:8000
	# Default Seldon creates a service named: wine-model-default
	kubectl set env deployment/wine-app TRITON_URL- SELDON_URL=http://wine-model-default.$(KUBE_NAMESPACE):8000 -n $(KUBE_NAMESPACE)
	
	@echo "Waiting for pods..."
	kubectl wait --for=condition=ready pod -l app=mlflow-server --timeout=120s -n $(KUBE_NAMESPACE)
	kubectl wait --for=condition=ready pod -l app=wine-app --timeout=120s -n $(KUBE_NAMESPACE)
	@echo "Waiting for Seldon Deployment..."
	kubectl wait --for=condition=available deployment/wine-model-default-0-ensemble-model --timeout=300s -n $(KUBE_NAMESPACE)

# Test Inference
test:
	@echo "Testing Inference Endpoint..."
	@# Check if port forwarding is needed or if LoadBalancer is working (Docker Desktop)
	@echo "Sending prediction request to http://localhost:8000/predict"
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2, "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76, "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38, "hue": 1.05, "od280_od315_of_diluted_wines": 3.4, "proline": 1050.0}'

# Clean up resources
clean:
	@echo "Cleaning up namespace: $(KUBE_NAMESPACE)..."
	kubectl delete -f k8s/local/app-deployment.yaml --ignore-not-found -n $(KUBE_NAMESPACE)
	kubectl delete -f k8s/local/triton-deployment.yaml --ignore-not-found -n $(KUBE_NAMESPACE)
	kubectl delete -f k8s/local/seldon-deployment.yaml --ignore-not-found -n $(KUBE_NAMESPACE)
	kubectl delete -f k8s/local/mlflow-deployment.yaml --ignore-not-found -n $(KUBE_NAMESPACE)

# Show logs
logs-app:
	kubectl logs -l app=wine-app -f -n $(KUBE_NAMESPACE)

logs-triton:
	kubectl logs -l app=triton-server -f -n $(KUBE_NAMESPACE)

