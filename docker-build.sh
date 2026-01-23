#!/bin/bash
# Build script for LLM RAG Response Pipe Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Parse arguments
BUILD_VARIANT="${1:-cpu}"
NO_CACHE="${2:-}"

# Validate variant
if [[ ! "$BUILD_VARIANT" =~ ^(cpu|gpu|both)$ ]]; then
    print_error "Invalid variant: $BUILD_VARIANT"
    echo "Usage: $0 [cpu|gpu|both] [--no-cache]"
    echo ""
    echo "Options:"
    echo "  cpu        Build CPU-only variant (default)"
    echo "  gpu        Build GPU-enabled variant"
    echo "  both       Build both variants"
    echo "  --no-cache Build without using Docker cache"
    exit 1
fi

# Build options
BUILD_OPTS=""
if [[ "$NO_CACHE" == "--no-cache" || "$2" == "--no-cache" ]]; then
    BUILD_OPTS="--no-cache"
    print_warn "Building without cache"
fi

# Build CPU variant
build_cpu() {
    print_info "Building CPU variant..."
    docker-compose build $BUILD_OPTS llm-rag-pipe
    print_info "✓ CPU variant built successfully"
}

# Build GPU variant
build_gpu() {
    print_info "Building GPU variant..."
    docker-compose --profile gpu build $BUILD_OPTS llm-rag-pipe-gpu
    print_info "✓ GPU variant built successfully"
}

# Main build logic
case "$BUILD_VARIANT" in
    cpu)
        build_cpu
        ;;
    gpu)
        build_gpu
        ;;
    both)
        build_cpu
        echo ""
        build_gpu
        ;;
esac

print_info "Build complete!"
echo ""
echo "To start the services:"
echo "  CPU variant:  docker-compose up -d"
echo "  GPU variant:  docker-compose --profile gpu up -d"
