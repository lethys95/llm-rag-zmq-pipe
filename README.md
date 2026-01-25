# LLM RAG Response Pipe

A professionally architected ZMQ-based pipeline for LLM (Large Language Model) and RAG (Retrieval-Augmented Generation) processing with advanced features including sentiment analysis, context interpretation, and memory decay algorithms.

## Features

- 🔌 **ZMQ Pipeline Architecture**: ROUTER socket for receiving prompts, PUSH socket for forwarding responses
- 🤖 **Multiple LLM Providers**: Support for both local (llama-cpp-python) and remote (OpenRouter) inference
- 🔍 **Production RAG**: Full Qdrant vector database integration with semantic search
- 🧠 **Memory Decay Algorithm**: Time-weighted relevance scoring for conversation history
- 💭 **Sentiment Analysis**: Advanced sentiment tracking with relevance and temporal decay metrics
- 🎯 **Context Interpreter**: LLM-powered context reformulation for better response quality
- 💾 **Dual-Tier Storage**: SQLite for conversation history + Qdrant for semantic retrieval
- ⚙️ **Flexible Configuration**: Cascading config system (CLI → file → defaults)
- 🎯 **Clean Architecture**: Separation of concerns, SOLID principles, high cohesion and loose coupling
- 📊 **Comprehensive Logging**: Detailed logging at all levels for debugging and monitoring
- 🧩 **Node-Based Architecture**: Dynamic node selection and execution for flexible processing
- 🤝 **Trust Analysis**: Multi-factor trust scoring for relationship maturity tracking
- 🔄 **Memory Evaluation**: AI-driven memory importance assessment with dynamic scoring
- 🧹 **Detox Protocol**: Background self-correction to prevent sycophancy and drift
- 📅 **Task Scheduling**: Periodic background tasks for maintenance and optimization


## Advanced Features

### Sentiment Analysis

The pipeline includes sophisticated sentiment analysis that tracks:

- **Sentiment Classification**: Positive, negative, or neutral
- **Relevance Score** (0.0-1.0): General impact/importance of the subject
- **Chrono-Relevance Score** (0.0-1.0): How long the subject stays relevant over time
- **Context Summary**: Brief description of the specific situation
- **Key Topics**: Main topics identified in the message

This enables intelligent memory management and context-aware responses.

### Memory Decay Algorithm

Implements time-weighted relevance scoring to prioritize recent and chronologically-relevant memories:

- **Exponential decay** based on configurable half-life
- **Chrono-relevance weighting** to preserve long-term important memories
- **Automatic pruning** of low-relevance documents
- **Semantic + temporal ranking** for optimal context retrieval

### Context Interpreter

An intermediate LLM handler that:

- Synthesizes information from multiple retrieved documents
- Removes redundancy and contradictions
- Highlights relevant information
- Organizes context logically for the primary LLM
- Reduces token usage while improving response quality

### Dual-Tier Storage

- **SQLite**: Fast, ordered conversation history for recent context
- **Qdrant**: Semantic vector search across all conversations
- **Automatic synchronization** between both storage tiers
- **Integrated embeddings** using sentence-transformers

### Node-Based Architecture

The system uses a flexible node-based architecture where:

- **Dynamic Node Selection**: Decision engine selects appropriate nodes based on context
- **Priority-Based Execution**: Nodes execute based on priority and queue type
- **Background Processing**: Long-running tasks execute asynchronously
- **Extensible Design**: Easy to add new processing nodes

Available nodes include:
- `sentiment_analysis`: Analyze emotional tone and sentiment
- `trust_analysis`: Evaluate user trust and relationship maturity
- `memory_evaluator`: Re-evaluate memory importance in current context
- `primary_response`: Generate the main response
- `detox_session`: Background self-correction protocol

### Trust Analysis

Multi-factor trust scoring system that tracks:

- **Relationship Age**: How long the user has been interacting
- **Interaction Frequency**: How often the user engages
- **Positive/Negative Ratio**: Balance of positive vs negative interactions
- **Consistency Score**: Variance in sentiment patterns
- **Depth Score**: Content analysis for interaction depth

Trust scores (0.0-1.0) are used to:
- Adjust memory importance weighting
- Influence nudging algorithm behavior
- Guide response personalization

### Memory Evaluation

AI-driven memory importance assessment that:

- **Re-evaluates memories** in current conversation context
- **Considers trust level** when scoring importance
- **Tracks access patterns** for relevance scoring
- **Applies dynamic boosts** to important memories
- **Provides reasoning** for evaluation decisions

### Detox Protocol

Background self-correction system that:

- **Triggers during idle time** (configurable)
- **Runs nudging algorithm** to prevent sycophancy
- **Consolidates similar memories** for efficiency
- **Stores companion state** in RAG for persistence
- **Generates guidance** for future conversations

The detox protocol helps maintain:
- Balanced, grounded responses
- Reduced extremism over time
- Healthy user-companion relationship
- Efficient memory management

## Installation

### Step 1: Install Base Dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

This installs core dependencies: `click`, `pyzmq`, `requests`, `qdrant-client`, `pydantic`, `msgpack`, and `sentence-transformers`.

### Step 2: Set Up Qdrant

Start Qdrant using Docker:

```bash
docker-compose up -d
```

This starts Qdrant on `localhost:6333` with persistent storage in `./data/qdrant_storage`.

### Step 3: (Optional) Install Local LLM Support

If you want to use local LLM inference with GPU acceleration:

```bash
./setup.sh
```

This script installs `llama-cpp-python` with CUDA support. **Note**: This is only needed if you plan to use the `llama` provider. The `openrouter` provider works without this step.

## Docker Deployment

The project includes production-ready Docker containers with both CPU and GPU support. Docker deployment is **recommended for production** as it provides:

- ✅ Isolated, reproducible environment
- ✅ Easy dependency management
- ✅ Built-in Qdrant integration
- ✅ Health checks and monitoring
- ✅ Volume mounts for data persistence
- ✅ GPU acceleration (optional)

### Quick Start with Docker

**1. Clone and configure:**
```bash
cd /path/to/llm_rag_response_pipe
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

**2. Build and run (CPU variant):**
```bash
./docker-build.sh cpu
./docker-run.sh cpu
```

**3. Test the pipeline:**
```bash
python examples/basic_client.py
```

### Docker Variants

#### CPU-Only (Default)
- **Best for**: OpenRouter-based inference, cloud deployments
- **Image size**: ~1.5-2 GB
- **Embedding speed**: 50-100 texts/second

```bash
docker-compose up -d
```

#### GPU-Enabled
- **Best for**: Faster embeddings, local LLM inference
- **Image size**: ~8-10 GB
- **Embedding speed**: 500-1000 texts/second (10x faster)
- **Requires**: NVIDIA GPU with CUDA 13.1 support

```bash
# First, install NVIDIA Container Toolkit (one-time setup)
# Then build and run GPU variant
docker-compose --profile gpu up -d
```

### Docker Commands Reference

```bash
# Build images
./docker-build.sh cpu        # CPU variant
./docker-build.sh gpu        # GPU variant
./docker-build.sh both       # Both variants

# Run containers
./docker-run.sh cpu          # CPU variant in background
./docker-run.sh gpu          # GPU variant in background

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Check status
docker-compose ps
```

## Configuration

### Configuration Precedence

The configuration system uses cascading precedence (highest to lowest):

1. **CLI Arguments**: Directly provided command-line options
2. **Environment Variables**: Environment variables (see below)
3. **Config File**: JSON configuration file (if specified)
4. **Defaults**: Built-in default values

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_RAG_PIPE_INPUT_ADDRESS` | Input endpoint address | `tcp://*:6666` |
| `OPENROUTER_API_KEY` | OpenRouter API key (required for remote provider) | `sk-or-...` |

### Core Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_endpoint` | ZMQ ROUTER bind address | `tcp://*:5555` |
| `output_endpoint` | ZMQ PUSH connect address | `tcp://localhost:5556` |
| `rag_enabled` | Enable RAG functionality | `true` |
| `rag_type` | RAG provider type | `qdrant` |
| `log_level` | Logging level | `INFO` |

### LLM Configuration

Each LLM component (primary, sentiment, interpreter) can be configured independently:

| Option | Description | Default |
|--------|-------------|---------|
| `primary_llm.provider` | LLM provider (llama/openrouter) | `openrouter` |
| `primary_llm.openrouter_model` | OpenRouter model identifier | `anthropic/claude-3.5-sonnet` |
| `primary_llm.model_path` | Path to local model file (for llama) | `None` |
| `sentiment_llm.provider` | Sentiment analysis LLM provider | `openrouter` |
| `interpreter_llm.provider` | Context interpreter LLM provider | `openrouter` |

### Feature Toggles

| Option | Description | Default |
|--------|-------------|---------|
| `enable_sentiment_analysis` | Enable sentiment analysis | `true` |
| `enable_context_interpreter` | Enable context interpreter | `true` |
| `sentiment_max_retries` | Max retries for sentiment analysis | `3` |
| `sentiment_retry_delay` | Delay between retries (seconds) | `1.0` |

### Memory Decay Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `memory_decay.half_life_days` | Half-life for memory decay | `7.0` |
| `memory_decay.chrono_weight` | Weight for chrono-relevance | `1.5` |
| `memory_decay.retrieval_threshold` | Minimum score to retrieve | `0.3` |
| `memory_decay.prune_threshold` | Score below which to prune | `0.1` |
| `memory_decay.max_documents` | Max documents to retrieve | `10` |

### Detox Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `detox.idle_trigger_minutes` | Minutes of inactivity before triggering detox | `60` |
| `detox.min_interval_minutes` | Minimum minutes between detox sessions | `120` |
| `detox.max_duration_minutes` | Maximum duration of a detox session | `30` |

### Nudging Algorithm Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `nudge_strength` | Strength of nudging adjustments | `0.15` |
| `max_companion_drift` | Maximum allowed companion drift | `0.3` |
| `base_user_influence` | Base influence of user position | `0.3` |
| `base_companion_influence` | Base influence of companion position | `0.7` |
| `max_trust_boost` | Maximum trust-based boost | `0.3` |

### Memory Consolidation Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `consolidation_threshold` | Similarity threshold for consolidation | `0.7` |
| `max_memories_per_batch` | Max memories to consolidate per batch | `10` |

### Qdrant Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `qdrant.collection_name` | Qdrant collection name | `conversations` |
| `qdrant.embedding_dim` | Embedding dimension | `384` |
| `qdrant.url` | Qdrant server URL | `http://localhost:6333` |
| `qdrant.path` | Local storage path (if not using URL) | `None` |

### Conversation Store Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `conversation_store.db_path` | SQLite database path | `./data/conversations.db` |
| `conversation_store.max_messages` | Max messages to keep | `1000` |
| `conversation_store.context_limit` | Max messages for context | `20` |

### Example Config File

Create a `config.json` file:

```json
{
  "input_endpoint": "tcp://*:5555",
  "output_endpoint": "tcp://localhost:5556",
  "primary_llm": {
    "provider": "openrouter",
    "openrouter_model": "anthropic/claude-3.5-sonnet"
  },
  "sentiment_llm": {
    "provider": "openrouter",
    "openrouter_model": "google/gemini-flash-1.5"
  },
  "enable_sentiment_analysis": true,
  "enable_context_interpreter": true,
  "rag_enabled": true,
  "memory_decay": {
     "half_life_days": 7.0,
     "max_documents": 10
   },
   "detox": {
     "idle_trigger_minutes": 60,
     "min_interval_minutes": 120,
     "max_duration_minutes": 30
   },
   "nudge_strength": 0.15,
   "max_companion_drift": 0.3,
   "base_user_influence": 0.3,
   "base_companion_influence": 0.7,
   "max_trust_boost": 0.3,
   "consolidation_threshold": 0.7,
   "max_memories_per_batch": 10,
   "log_level": "INFO"
 }
```

## Usage

### Using OpenRouter (Remote API)

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Run the server:

```bash
# Using default configuration
llm-rag-pipe remote

# With custom endpoints
llm-rag-pipe remote --input-endpoint tcp://*:5555 --output-endpoint tcp://localhost:5556

# With specific model
llm-rag-pipe remote --openrouter-model anthropic/claude-3.5-sonnet

# With config file
llm-rag-pipe remote --config-file config.json

# With CLI overrides
llm-rag-pipe remote --config-file config.json --temperature 0.9 --log-level DEBUG
```

### Using Local Llama Model

First, install local LLM support:

```bash
./setup.sh
```

Then run the server:

```bash
# Basic usage with model path
llm-rag-pipe local --model-path /path/to/your/model.gguf

# With GPU acceleration
llm-rag-pipe local \
  --model-path /path/to/your/model.gguf \
  --n-ctx 4096 \
  --n-gpu-layers -1 \
  --n-threads 8

# With config file
llm-rag-pipe local \
  --config-file config.json \
  --model-path /path/to/model.gguf
```

## ZMQ Protocol

### Request Format (DialogueInput)

The pipeline expects JSON messages conforming to the `DialogueInput` schema:

```json
{
  "content": "What did we discuss about the project yesterday?",
  "speaker": "user",
  "system_prompt_override": null
}
```

**Fields:**
- `content` (required): The dialogue content/text
- `speaker` (required): Identifier for who is speaking (e.g., "user", "john", "assistant")
- `system_prompt_override` (optional): Custom system prompt for this request

### Response Format

The pipeline sends two responses per request:

1. **Acknowledgment** (via ROUTER socket to requester):
   ```
   Request processed successfully | Sentiment: negative
   ```

2. **Generated Response** (via PUSH socket to downstream):
   ```
   Based on our discussion yesterday, the project deadline is next Friday...
   ```

### Example Client Code

See the `examples/` directory for complete working examples. Here's a basic example:

```python
import zmq
import json

# Connect to pipeline
context = zmq.Context()

# DEALER socket to send requests
dealer = context.socket(zmq.DEALER)
dealer.connect("tcp://localhost:5555")

# PULL socket to receive responses
pull = context.socket(zmq.PULL)
pull.bind("tcp://*:5556")

# Send a request
dialogue_input = {
    "content": "What is the capital of France?",
    "speaker": "user"
}
dealer.send_json(dialogue_input)

# Receive acknowledgment
ack = dealer.recv_string()
print(f"ACK: {ack}")

# Receive response
response = pull.recv_string()
print(f"Response: {response}")
```

## Pipeline Flow

1. **Request Reception**: ZMQ ROUTER socket receives DialogueInput from client
2. **Node Selection**: Decision engine selects appropriate processing nodes
3. **Sentiment Analysis**: Analyze emotional tone and sentiment
4. **Trust Analysis**: Evaluate user trust (periodically, every 10th message)
5. **Context Retrieval**:
   - Recent messages from SQLite conversation store
   - Semantic search in Qdrant with memory decay scoring
   - Combined using memory decay algorithm
6. **Memory Evaluation**: Re-evaluate memory importance (if documents retrieved)
7. **Context Interpretation**: Optional LLM-powered context reformulation
8. **Response Generation**: Primary LLM generates response using retrieved context
9. **Storage**: Conversation stored in both SQLite and Qdrant with embeddings
10. **Acknowledgment**: Server sends ACK with sentiment info back via ROUTER
11. **Response Forwarding**: Final response forwarded via PUSH socket
12. **Background Tasks**: Detox protocol runs during idle periods

## Examples

See the `examples/` directory for comprehensive working examples:

- `basic_client.py` - Simple request/response
- `dialogue_client.py` - DialogueInput format usage
- `conversation_demo.py` - Multi-turn conversation with memory
- `sentiment_aware_client.py` - Sentiment analysis demonstration
- `rag_retrieval_demo.py` - RAG and memory retrieval
- `test_pipeline.py` - End-to-end pipeline testing

## Development

### Adding New LLM Providers

1. Create new provider in `src/llm/` implementing `BaseLLM`
2. Add factory logic in `src/llm/factory.py`
3. Update configuration in `src/config/settings.py`

### Adding RAG Providers

1. Create new provider in `src/rag/` implementing `BaseRAG`
2. Add factory logic in `src/rag/factory.py`
3. Update configuration as needed

### Project Structure

- `src/`: Main source code
- `docs/`: Comprehensive documentation
- `examples/`: Working example scripts
- `data/`: Runtime data (SQLite DB, Qdrant storage)
- `pyproject.toml`: Package configuration
- `docker-compose.yml`: Qdrant Docker setup

## Documentation

- [Memory Decay Algorithm](docs/MEMORY_DECAY_ALGORITHM.md) - Detailed algorithm implementation
- [Testing Guide](docs/TESTING.md) - Testing infrastructure and practices
- [Implementation Progress](docs/IMPLEMENTATION_PROGRESS.md) - Phase 2 completion status
- [Detox Protocol](docs/algorithms/DETOX_PROTOCOL.md) - Background self-correction system

## Future Enhancements

- [ ] Additional LLM providers (Anthropic direct, OpenAI direct, etc.)
- [ ] Request queuing and batching
- [ ] Metrics and monitoring (Prometheus, Grafana)
- [ ] Health check endpoints
- [ ] Multi-user conversation isolation
- [ ] Advanced caching strategies
- [ ] Adaptive detox scheduling based on usage patterns
- [ ] Multi-user trust tracking
- [ ] Memory importance auto-adjustment

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure your code follows the established architecture patterns and includes appropriate documentation.
