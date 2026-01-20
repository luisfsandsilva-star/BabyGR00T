#!/bin/bash
# ==============================================================================
# BabyGR00T Docker Quick Start Script
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="babygr00t:latest"
CONTAINER_NAME="babygr00t-${1:-dev}"

# Functions
print_usage() {
    echo -e "${BLUE}BabyGR00T Docker Helper${NC}"
    echo ""
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  build           Build the Docker image"
    echo "  shell           Start interactive shell"
    echo "  train           Run training with pretrain.py"
    echo "  finetune        Run finetuning with finetune.py"
    echo "  distill         Run GR00T distillation"
    echo "  jupyter         Start Jupyter notebook server"
    echo "  test            Run tests"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build image"
    echo "  $0 shell                    # Interactive shell"
    echo "  $0 train [training args]    # Train model"
    echo "  $0 jupyter                  # Start Jupyter on port 8888"
    echo ""
}

check_gpu() {
    if ! nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
        GPU_FLAG=""
    else
        echo -e "${GREEN}✓ GPU detected${NC}"
        GPU_FLAG="--gpus all"
    fi
}

build_image() {
    echo -e "${BLUE}Building BabyGR00T Docker image...${NC}"
    docker build -t ${IMAGE_NAME} .
    echo -e "${GREEN}✓ Image built successfully${NC}"
}

run_shell() {
    check_gpu
    echo -e "${BLUE}Starting interactive shell...${NC}"
    docker run -it --rm ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/outputs:/workspace/outputs \
        -v $(pwd)/config:/workspace/config \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        ${IMAGE_NAME} bash
}

run_train() {
    check_gpu
    echo -e "${BLUE}Starting training...${NC}"
    docker run -it --rm ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/outputs:/workspace/outputs \
        -v $(pwd)/config:/workspace/config \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        ${IMAGE_NAME} \
        python pretrain.py "${@:2}"
}

run_finetune() {
    check_gpu
    echo -e "${BLUE}Starting finetuning...${NC}"
    docker run -it --rm ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/outputs:/workspace/outputs \
        -v $(pwd)/config:/workspace/config \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} \
        ${IMAGE_NAME} \
        python finetune.py "${@:2}"
}

run_distill() {
    check_gpu
    echo -e "${BLUE}Starting GR00T distillation...${NC}"
    docker run -it --rm ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/outputs:/workspace/outputs \
        -e HF_TOKEN=${HF_TOKEN:-} \
        ${IMAGE_NAME} \
        python build_gr1_omnivlm_embeddings_fixed.py "${@:2}"
}

run_jupyter() {
    check_gpu
    echo -e "${BLUE}Starting Jupyter notebook server...${NC}"
    echo -e "${YELLOW}Access at: http://localhost:8888${NC}"
    docker run -it --rm ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/outputs:/workspace/outputs \
        -v $(pwd):/workspace/notebooks \
        -p 8888:8888 \
        ${IMAGE_NAME} \
        jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token=''
}

run_test() {
    check_gpu
    echo -e "${BLUE}Running tests...${NC}"
    docker run -it --rm ${GPU_FLAG} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/tests:/workspace/tests \
        ${IMAGE_NAME} \
        pytest tests/ -v "${@:2}"
}

# Main
case "${1:-}" in
    build)
        build_image
        ;;
    shell|sh|bash)
        run_shell
        ;;
    train)
        run_train "$@"
        ;;
    finetune|ft)
        run_finetune "$@"
        ;;
    distill)
        run_distill "$@"
        ;;
    jupyter|notebook)
        run_jupyter
        ;;
    test)
        run_test "$@"
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '${1}'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
