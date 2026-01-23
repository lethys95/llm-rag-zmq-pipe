# Multi-stage Dockerfile for LLM RAG Response Pipe
# Supports both CPU-only and GPU-enabled variants

# =============================================================================
# Stage 1: Base CPU Image
# =============================================================================
FROM python:3.12-slim-bookworm AS base-cpu

LABEL maintainer="LLM RAG Response Pipe"
LABEL description="CPU-optimized variant for OpenRouter + sentence-transformers"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# =============================================================================
# Stage 2: Base GPU Image
# =============================================================================
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04 AS base-gpu

LABEL maintainer="LLM RAG Response Pipe"
LABEL description="GPU-enabled variant with CUDA 13.1 support"

WORKDIR /app

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies with CUDA support
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -e . && \
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

# =============================================================================
# Stage 3: Production CPU
# =============================================================================
FROM base-cpu AS production-cpu

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/data /app/models /app/config && \
    chown -R appuser:appuser /app

# Copy application source
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser main.py /app/

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Expose ports
EXPOSE 5555 5556

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import zmq; ctx = zmq.Context(); sock = ctx.socket(zmq.REQ); sock.connect('tcp://localhost:5555'); sock.close(); ctx.term()" || exit 1

# Default command
CMD ["python", "-m", "src.cli", "remote", "--input-endpoint", "tcp://*:5555", "--output-endpoint", "tcp://localhost:5556"]

# =============================================================================
# Stage 4: Production GPU
# =============================================================================
FROM base-gpu AS production-gpu

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/data /app/models /app/config && \
    chown -R appuser:appuser /app

# Copy application source
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser main.py /app/

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 5555 5556

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import zmq; ctx = zmq.Context(); sock = ctx.socket(zmq.REQ); sock.connect('tcp://localhost:5555'); sock.close(); ctx.term()" || exit 1

# Default command
CMD ["python", "-m", "src.cli", "remote", "--input-endpoint", "tcp://*:5555", "--output-endpoint", "tcp://localhost:5556"]
