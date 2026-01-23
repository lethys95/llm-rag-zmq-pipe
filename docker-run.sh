#!/bin/bash
# Run script for LLM RAG Response Pipe Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Parse arguments
RUN_VARIANT="${1:-cpu}"
RUN_MODE="${2:-detached}"

# Validate variant
if [[ ! "$RUN_VARIANT" =~ ^(cpu|gpu)$ ]]; then
    print_error "Invalid variant: $RUN_VARIANT"
    echo "Usage: $0 [cpu|gpu] [detached|foreground]"
    echo ""
    echo "Options:"
    echo "  cpu        Run CPU-only variant (default)"
    echo "  gpu        Run GPU-enabled variant"
    echo "  detached   Run in background (default)"
    echo "  foreground Run in foreground with logs"
    exit 1
fi

# Check for .env file
if [ ! -f .env ]; then
    print_warn ".env file not found"
    print_info "Creating .env from .env.example..."
    cp .env.example .env
    print_warn "Please edit .env and add your OPENROUTER_API_KEY before running"
    echo ""
    echo "  nano .env"
    echo ""
    exit 1
fi

# Check for OPENROUTER_API_KEY
if ! grep -q "^OPENROUTER_API_KEY=..*" .env; then
    print_error "OPENROUTER_API_KEY not set in .env file"
    echo "Please edit .env and add your API key:"
    echo ""
    echo "  nano .env"
    echo ""
    exit 1
fi

# Create necessary directories
print_step "Creating data directories..."
mkdir -p data/qdrant_storage models config

# Run based on variant and mode
if [ "$RUN_MODE" == "foreground" ]; then
    RUN_OPTS=""
    print_info "Running in foreground mode (Ctrl+C to stop)..."
else
    RUN_OPTS="-d"
    print_info "Running in detached mode..."
fi

if [ "$RUN_VARIANT" == "gpu" ]; then
    print_step "Starting GPU variant..."
    docker-compose --profile gpu up $RUN_OPTS
    
    if [ "$RUN_MODE" == "detached" ]; then
        print_info "Services started successfully!"
        echo ""
        echo "GPU variant is running. To view logs:"
        echo "  docker-compose logs -f llm-rag-pipe-gpu"
        echo ""
        echo "To check GPU usage:"
        echo "  docker exec llm_rag_pipe_gpu nvidia-smi"
        echo ""
        echo "To stop:"
        echo "  docker-compose --profile gpu down"
    fi
else
    print_step "Starting CPU variant..."
    docker-compose up $RUN_OPTS
    
    if [ "$RUN_MODE" == "detached" ]; then
        print_info "Services started successfully!"
        echo ""
        echo "CPU variant is running. To view logs:"
        echo "  docker-compose logs -f llm-rag-pipe"
        echo ""
        echo "To stop:"
        echo "  docker-compose down"
    fi
fi

if [ "$RUN_MODE" == "detached" ]; then
    echo ""
    print_info "Check service status:"
    echo "  docker-compose ps"
    echo ""
    print_info "Test the pipeline:"
    echo "  python examples/basic_client.py"
fi
