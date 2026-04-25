# LLM RAG Response Pipe

A ZMQ-based AI companion pipeline. Sits between an STT system and a TTS system, processes each turn through a dynamically-selected chain of analysis and advisory nodes, and returns a response that feels like it comes from a friend who knows the user — not a chatbot.

## What it is

The companion's goal is long-term mental health support delivered invisibly through authentic friendship. The psychological machinery (Maslow needs scoring, VAD emotional state, therapeutic strategy selection, memory decay) operates entirely behind the scenes. The user experiences a friend who remembers them, reads the room, and knows when to push back.

Read `docs/PROJECT_VISION.md` for the full philosophy. Read `docs/PSYCHOLOGY.md` for the psychological frameworks. Read `docs/ADVISOR_PATTERN_CONCEPT.md` before touching the analysis or response layer.

## Architecture

### Transport

ZMQ ROUTER socket receives multipart frames from upstream (STT or direct client). ZMQ DEALER socket forwards the final response downstream to TTS. ACKs go back to the caller via ROUTER immediately after the request is received — response generation is async.

### Event processing

Each incoming message creates a fresh `KnowledgeBroker` (typed shared workspace). A worker LLM (the **coordinator**) decides which node to run next at each step, reading the broker's current state to make adaptive routing decisions. Nodes write their results back to the broker. The loop runs until the coordinator signals completion or the 20-node safety limit is hit.

This means the pipeline is not fixed. A greeting and a grief disclosure do not take the same path.

### The advisory layer

Analysis nodes (classifiers) produce structured data. Advisor nodes consume that data and produce natural language guidance for the primary LLM. The primary LLM never sees raw scores — it sees human-readable advice with a potency signal indicating how much weight to put on it this turn.

```
Classifiers → Advisors → Primary LLM
```

See `docs/ADVISOR_PATTERN_CONCEPT.md`.

### Memory

Two tiers:
- **SQLite** — recent conversation history for in-context use
- **Qdrant** — long-term vector storage with time-weighted decay scoring

The memory decay algorithm applies exponential decay modulated by `chrono_relevance` — a score set at storage time indicating how long a memory should matter. A mother's death decays slowly. What the user had for lunch decays fast.

## Running it

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

Run with remote LLM (default):

```bash
llm-rag-pipe remote
llm-rag-pipe remote --input-endpoint tcp://*:5555 --output-endpoint tcp://localhost:20501
```

Run with local llama model:

```bash
llm-rag-pipe local --model-path /path/to/model.gguf
```

Send a single test message through the full pipeline (ephemeral in-memory storage):

```bash
llm-rag-pipe test-run --message "Hey, how are you doing?"
```

## ZMQ protocol

Send to the ROUTER socket as a multipart message: `[topic, payload]`

Topic is either `dialogue` or `stt`. Payload is msgpack or JSON.

For `dialogue`:
```json
{ "content": "I feel really alone today.", "speaker": "user" }
```

For `stt` (from transcription system):
```json
{ "status": "success", "text": "I feel really alone today.", "speaker": "user" }
```

The DEALER socket on the other end receives the plain response string.

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_RAG_PIPE_INPUT_ADDRESS` | `tcp://*:5555` | ROUTER bind address |
| `TTS_INPUT_ADDRESS` | `tcp://localhost:20501` | DEALER connect address |
| `OPENROUTER_API_KEY` | — | Required for remote provider |

## Docs

- `docs/ADVISOR_PATTERN_CONCEPT.md` — read before touching the analysis/response layer
- `docs/ORCHESTRATOR.md` — event model and coordinator design
- `docs/PSYCHOLOGY.md` — OCEAN, VAD, tonic/phasic, detox framework
- `docs/MEMORY_DECAY_ALGORITHM.md` — decay math and tuning
- `docs/algorithms/DETOX_PROTOCOL.md` — background recalibration system
- `docs/NODE_META_FEEDBACK.md` — far-future prompt evaluation system
- `NEXT_SESSION.md` — current state and what needs building next
