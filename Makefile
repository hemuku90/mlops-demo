# Makefile for Local Kubernetes Deployment

IMAGE_TAG := latest
APP_IMAGE := mlops-wine-app:$(IMAGE_TAG)
TRITON_IMAGE := mlops-triton:$(IMAGE_TAG)
KUBE_NAMESPACE := default

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

# Deploy to Kubernetes
deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/local/triton-deployment.yaml
	kubectl apply -f k8s/local/app-deployment.yaml
	@echo "Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=triton-server --timeout=120s
	kubectl wait --for=condition=ready pod -l app=wine-app --timeout=120s
	@echo "Deployment complete!"

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
	@echo "Cleaning up..."
	kubectl delete -f k8s/local/app-deployment.yaml --ignore-not-found
	kubectl delete -f k8s/local/triton-deployment.yaml --ignore-not-found

# Show logs
logs-app:
	kubectl logs -l app=wine-app -f

logs-triton:
	kubectl logs -l app=triton-server -f
