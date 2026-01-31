# ==============================================================================
# BabyGR00T: Unified Dockerfile
# ==============================================================================
# Purpose: Single image for GR00T distillation, TRM training, and inference
# Base: PyTorch 2.6.0 with CUDA 12.4 and cuDNN 9
# ==============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

# ==============================================================================
# System Dependencies
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Version control
    git git-lfs \
    # Build tools
    build-essential cmake ninja-build \
    # OpenCV dependencies
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    # Media processing
    ffmpeg \
    # Utilities
    vim less htop tmux wget curl zip unzip \
    ca-certificates \
    # Network tools (for debugging)
    netcat dnsutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ==============================================================================
# Python Build Dependencies
# ==============================================================================
RUN pip install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0 \
    ninja \
    packaging

# ==============================================================================
# GR00T Dependencies (Teacher Model for Distillation)
# ==============================================================================
# Copy GR00T package definition
COPY GR00T_N1.5/pyproject.toml /tmp/GR00T_N1.5/pyproject.toml
WORKDIR /tmp/GR00T_N1.5

# Install GR00T base dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.26.0 \
    huggingface-hub \
    decord \
    pyarrow \
    && pip install --no-cache-dir -e .[base] --no-deps || true

# Install Flash Attention (optional, for faster inference)
# This may take 5-10 minutes to compile
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.7.1.post4 --no-build-isolation || \
    echo "WARNING: Flash Attention installation failed. Continuing without it..."

# ==============================================================================
# BabyGR00T Dependencies (TRM Student Model)
# ==============================================================================
WORKDIR /workspace

# Core dependencies for TRM training
RUN pip install --no-cache-dir \
    # Deep Learning
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    numpy==1.26.4 \
    # Optimizer
    lion-pytorch \
    # Configuration & Logging
    hydra-core>=1.3.0 \
    pydantic>=2.0.0 \
    omegaconf \
    coolname \
    tqdm \
    # Data Processing
    opencv-python==4.8.0.74 \
    pillow \
    # Optional: Experiment tracking
    wandb==0.19.0 \
    matplotlib

# ==============================================================================
# Install GR00T Package
# ==============================================================================
COPY GR00T_N1.5/ /workspace/GR00T_N1.5/
WORKDIR /workspace/GR00T_N1.5
RUN pip install --no-cache-dir -e . --no-deps

# ==============================================================================
# Install BabyGR00T Package
# ==============================================================================
WORKDIR /workspace

# Copy BabyGR00T source code
COPY models/ /workspace/models/
COPY dataset/ /workspace/dataset/
COPY evaluators/ /workspace/evaluators/
COPY utils/ /workspace/utils/
COPY config/ /workspace/config/
COPY pretrain.py /workspace/
COPY finetune.py /workspace/
COPY visual_embedding_builder.py /workspace/
COPY gr00t_distiller.py /workspace/
COPY requirements.txt /workspace/

# Install BabyGR00T dependencies
RUN pip install --no-cache-dir -r requirements.txt || echo "Some requirements may have failed"

# ==============================================================================
# Create Standard Directories
# ==============================================================================
RUN mkdir -p \
    /workspace/data \
    /workspace/outputs \
    /workspace/checkpoints \
    /workspace/logs

# ==============================================================================
# Environment Variables
# ==============================================================================
# Disable torch compilation warnings
ENV TORCH_COMPILE_DEBUG=0
# Set default device
ENV CUDA_VISIBLE_DEVICES=0
# Optimize memory usage
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Health Check & Verification
# ==============================================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')" && \
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
    echo "✓ All core dependencies verified"

# ==============================================================================
# Entry Point
# ==============================================================================
WORKDIR /workspace

# Default command: interactive shell
CMD ["/bin/bash"]

# ==============================================================================
# Usage Examples:
# ==============================================================================
# Build:
#   docker build -t babygr00t:latest .
#
# Interactive Shell:
#   docker run -it --gpus all \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/outputs:/workspace/outputs \
#     babygr00t:latest bash
#
# Training:
#   docker run --gpus all \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/outputs:/workspace/outputs \
#     babygr00t:latest \
#     python pretrain.py [args...]
#
# Distillation:
#   docker run --gpus all \
#     -v $(pwd)/data:/workspace/data \
#     babygr00t:latest \
#     python visual_embedding_builder.py [args...]
# ==============================================================================
