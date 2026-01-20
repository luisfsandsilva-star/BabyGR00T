.PHONY: help build shell train finetune distill jupyter test clean

# Configuration
IMAGE_NAME := babygr00t:latest
CUDA_DEVICES ?= 0

help: ## Show this help message
	@echo "BabyGR00T Makefile Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	@echo "Building BabyGR00T Docker image..."
	docker build -t $(IMAGE_NAME) .
	@echo "✓ Build complete"

shell: ## Start interactive shell in container
	./scripts/docker-run.sh shell

train: ## Run training (use: make train ARGS="your hydra args")
	./scripts/docker-run.sh train $(ARGS)

finetune: ## Run finetuning (use: make finetune ARGS="your hydra args")
	./scripts/docker-run.sh finetune $(ARGS)

distill: ## Run GR00T distillation
	./scripts/docker-run.sh distill $(ARGS)

jupyter: ## Start Jupyter notebook server
	./scripts/docker-run.sh jupyter

test: ## Run tests
	./scripts/docker-run.sh test

clean: ## Remove Docker images and containers
	@echo "Cleaning up Docker resources..."
	docker stop $$(docker ps -aq --filter ancestor=$(IMAGE_NAME)) 2>/dev/null || true
	docker rm $$(docker ps -aq --filter ancestor=$(IMAGE_NAME)) 2>/dev/null || true
	@echo "✓ Cleanup complete"

clean-all: clean ## Remove images, containers, and build cache
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
	docker system prune -f
	@echo "✓ Full cleanup complete"

# Shortcuts
dev: shell ## Alias for 'shell'
nb: jupyter ## Alias for 'jupyter'
