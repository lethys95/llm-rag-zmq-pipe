# Implementation Notes: Node-Based Execution System

## Overview

Migrating from rigid sequential pipeline to flexible node-based execution system with:
- **Knowledge Broker**: Central context pool
- **Execution Nodes**: Discrete, composable processing units
- **Task Queue**: Priority-based async execution
- **Decision Engine**: LLM-driven node selection
- **Node Registry**: Plugin-style extensibility

## Why "Nodes" Not "Pipeline"?

Pipeline implies sequential, rigid flow. Our system has:
- **Dynamic routing**: Different nodes for different situations
- **Parallel execution**: Multiple nodes can run simultaneously
- **Conditional activation**: Nodes run only when needed
- **Flexible topology**: Nodes can communicate in any pattern

More like a **computational graph** or **workflow engine** than a pipeline.

## Architecture Summary

```
User Message
    ↓
Decision Engine (LLM analyzes needs)
    ↓
Enqueue Nodes (priority-based)
    ↓
Execute Immediate Nodes → Knowledge Broker (accumulates context)
    ↓                              ↓
Background Nodes         Primary Response Node
(continue async)                   ↓
                              Response to User
```

## Phase 1: Core Framework (THIS SESSION)

### 1.1 Knowledge Broker (`src/nodes/knowledge_broker.py`)

**Purpose**: Central knowledge pool that accumulates context from all nodes

**Key Methods**:
- `add_knowledge(key, data)` - Nodes write results here
- `get_knowledge(key)` - Nodes read context from here
- `get_full_context()` - Primary response gets everything
- `clear()` - Reset for new request

**Design Notes**:
- Thread-safe (use asyncio.Lock if needed)
- Extensible schema (any node can add new keys)
- Type hints for common keys

### 1.2 Base Node (`src/nodes/base.py`)

**Purpose**: Abstract base class for all execution nodes

**Key Properties**:
- `name` - Unique identifier
- `priority` - Execution order (0 = highest)
- `dependencies` - List of node names that must complete first
- `queue_type` - "immediate" or "background"

**Key Methods**:
- `execute(broker)` - Main execution logic (abstract)
- `should_run(broker)` - Conditional execution (optional)
- `validate_dependencies(broker)` - Check dependencies met

**Design Notes**:
- All nodes inherit from this
- Async by default
- Return `NodeResult` with status, data, and optional next nodes to enqueue

### 1.3 Task Queue Manager (`src/nodes/queue_manager.py`)

**Purpose**: Manages execution of nodes with priorities and dependencies

**Key Components**:
- `immediate_queue` - PriorityQueue for user-facing nodes
- `background_queue` - PriorityQueue for async processing
- `completed_nodes` - Track what's finished for dependency checking

**Key Methods**:
- `enqueue(node, queue_type)` - Add node to appropriate queue
- `execute_immediate(broker)` - Execute all immediate nodes, block until complete
- `execute_background(broker)` - Execute background nodes without blocking
- `_check_dependencies(node, broker)` - Ensure dependencies met

**Design Notes**:
- Use `asyncio.PriorityQueue`
- Dependency resolution before execution
- Graceful error handling (failed node doesn't block others)

### 1.4 Decision Engine (`src/nodes/decision_engine.py`)

**Purpose**: LLM-based reasoning engine to decide which nodes to run

**Key Methods**:
- `analyze_and_select_nodes(message, broker)` - Main decision logic
- `_quick_analysis(message)` - Fast LLM call to analyze needs
- `_select_nodes_for_conversation()` - Default conversation nodes
- `_select_nodes_for_crisis()` - Crisis-specific nodes

**Design Notes**:
- Use lightweight LLM (gpt-4o-mini from settings)
- Start simple: rule-based with optional LLM enhancement
- Structured output (JSON) for node selection
- Fallback to default nodes if LLM fails

**Phase 1 Decision Logic** (Simple Rules):
```python
# Always for conversation:
- SentimentAnalysisNode (priority 1)
- PrimaryResponseNode (priority 3)

# Conditionally:
- If idle > 60min: DetoxProtocolNode (priority 10, background)
- If keyword "crisis": CrisisDetectionNode (priority 0)
```

### 1.5 Node Registry (`src/nodes/registry.py`)

**Purpose**: Central registry of all available nodes

**Key Methods**:
- `register(node_class)` - Register a node class
- `create(node_name, **kwargs)` - Factory to create node instances
- `list_available()` - Get all registered nodes
- `get_node_info(node_name)` - Get metadata about a node

**Design Notes**:
- Singleton pattern
- Auto-discovery via decorators (future)
- Config-based enable/disable (future)

## Phase 2: Wrap Existing Handlers (NEXT SESSION)

### 2.1 SentimentAnalysisNode
- Wraps `SentimentAnalysisHandler`
- Priority: 1 (early)
- Writes: `sentiment_analysis` to broker

### 2.2 PrimaryResponseNode
- Wraps `PrimaryResponseHandler`
- Priority: 3 (after sentiment)
- Dependencies: `["sentiment_analysis"]`
- Reads: All knowledge from broker
- Writes: `primary_response` to broker

### 2.3 StorageNode
- Extracts storage logic from server
- Priority: 5 (after response)
- Background: True

## Phase 3: Rewrite Server (COMPLETED)

### 3.1 Refactoring Complete ✅
- **Renamed**: `src/pipeline/` → `src/communication/`
- **Deleted**: `src/pipeline/server.py` (old PipelineServer)
- **Created**: `src/orchestrator.py` (new node-based orchestrator)
- **Created**: `src/communication/zmq_connection_manager.py` (persistent socket manager)
- **Created**: `src/nodes/communication_nodes/` (ZMQ message handling nodes)
  - `ReceiveMessageNode` - Parse incoming messages
  - `SendAcknowledgmentNode` - Send ACKs via ROUTER
  - `ForwardResponseNode` - Forward via DEALER
  - `CheckFeedbackNode` - Check downstream feedback
- **Updated**: `src/cli.py` to use Orchestrator instead of PipelineServer

### 3.2 Architecture Changes
**Old Pipeline (Sequential)**:
```
Receive → Sentiment → RAG → Response → ACK → Forward
```

**New Node System (Dynamic)**:
```
ZMQConnectionManager (persistent sockets)
    ↓
Orchestrator (main loop)
    ↓
ReceiveMessageNode → KnowledgeBroker
    ↓
[Sentiment + Primary handlers - temporary, to be wrapped in Phase 2]
    ↓
SendAcknowledgmentNode + ForwardResponseNode + CheckFeedbackNode
    ↓
(via TaskQueueManager)
```

### 3.3 Backwards Compatibility ✅
- Same ZMQ interface (ROUTER/DEALER)
- Same response format
- Same CLI commands (remote/local)
- Migration is transparent to clients

**✅ Phase 3 COMPLETE - Pipeline successfully removed!**

## Phase 4: Add New Nodes (FUTURE SESSIONS)

### Planned Nodes (from BRAINSTORMING.md)
- `DetoxProtocolNode` - Background self-correction
- `NeedsAnalysisNode` - Psychological needs assessment
- `StrategySelectionNode` - Choose therapeutic approach
- `TrustAnalysisNode` - Relationship maturity scoring
- `CrisisDetectionNode` - Safety checks
- `InterventionSchedulerNode` - Timed check-ins
- `CharacterConsistencyNode` - Personality grounding
- `IcebreakerNode` - Onboarding conversation
- `StalenessInjectionNode` - New content introduction
- etc.

## Directory Structure

```
src/
├── nodes/                     # NEW (replaces pipeline/ eventually)
│   ├── __init__.py
│   ├── base.py               # BaseNode abstract class
│   ├── knowledge_broker.py   # KnowledgeBroker class
│   ├── queue_manager.py      # TaskQueueManager class
│   ├── decision_engine.py    # DecisionEngine class
│   ├── registry.py           # NodeRegistry class
│   ├── result.py             # NodeResult dataclass
│   └── server.py             # NodeExecutionServer (Phase 3)
├── pipeline/                  # KEEP for now (Phase 3 migration)
│   ├── server.py             # Old sequential server
│   └── zmq_handler.py        # Keep (reuse in new server)
├── handlers/                  # KEEP (wrapped by nodes in Phase 2)
│   ├── sentiment_analysis.py
│   ├── primary_response.py
│   └── context_interpreter.py
```

## Implementation Order (This Session - Phase 1)

1. ✅ Create `src/nodes/__init__.py`
2. ✅ Create `src/nodes/result.py` - NodeResult dataclass
3. ✅ Create `src/nodes/base.py` - BaseNode abstract class
4. ✅ Create `src/nodes/knowledge_broker.py` - KnowledgeBroker
5. ✅ Create `src/nodes/queue_manager.py` - TaskQueueManager
6. ✅ Create `src/nodes/decision_engine.py` - DecisionEngine
7. ✅ Create `src/nodes/registry.py` - NodeRegistry
8. ✅ Write tests to validate Phase 1 components (`tests/test_nodes_phase1.py`)

**Phase 1 Status: ✅ COMPLETE**

## Testing Strategy

### Unit Tests
- Test `KnowledgeBroker` add/get/clear
- Test `BaseNode` dependency validation
- Test `TaskQueueManager` priority ordering
- Test `DecisionEngine` node selection logic
- Test `NodeRegistry` registration/creation

### Integration Tests (Phase 3)
- Test full request flow
- Compare output with old pipeline
- Verify backward compatibility

## Design Decisions & Rationale

### Why async/await?
- Background nodes need non-blocking execution
- Future nodes may have I/O (network, DB)
- Python's asyncio provides clean concurrency

### Why Priority Queue?
- Some nodes are more urgent (crisis > casual chat)
- Background nodes should run after immediate ones
- Dependencies create implicit priority

### Why Knowledge Broker pattern?
- Decouples nodes from each other
- Any node can contribute context
- Primary response sees accumulated knowledge
- Easy to add new data types

### Why LLM in Decision Engine?
- Complex decisions (crisis? intervention needed?)
- Adapts to conversation context
- Can learn patterns over time
- Fallback to rules if LLM fails

## Migration Risks & Mitigations

**Risk**: Breaking existing functionality
**Mitigation**: Keep old pipeline, A/B test, fallback option

**Risk**: Performance regression
**Mitigation**: Benchmark before/after, optimize queue

**Risk**: Complexity overhead
**Mitigation**: Start simple, add complexity only when needed

## Future Enhancements

### Node Composition Patterns
- **Sequential Chain**: A → B → C (current handlers)
- **Parallel Fan-out**: A → [B, C, D] → E (multiple analyses)
- **Conditional Branch**: A → (if X then B else C) → D
- **Feedback Loop**: A → B → (re-analyze) → A

### Node Communication Beyond Broker
- Direct node-to-node messaging
- Event bus for pub/sub patterns
- Streaming results (partial updates)

### Configuration & Discovery
- YAML-based node pipeline definitions
- Auto-discover nodes via decorators
- Enable/disable nodes via config
- Per-user node customization

### Monitoring & Observability
- Node execution timing
- Dependency graph visualization
- Failed node alerts
- Queue depth monitoring

## Notes to Future Self

- Don't over-engineer Phase 1 - simple works
- Test each component in isolation before integration
- Keep backward compatibility until Phase 3 validates
- Document node interfaces clearly for future nodes
- Consider renaming throughout: pipeline → node system
- The "pipeline" directory might become "execution" or "nodes"
- Keep the vision from PROJECT_VISION.md in mind
- Every node should ask: "Does this help the user feel understood?"

## Open Questions

1. Should nodes be able to enqueue other nodes dynamically?
   - Example: CrisisNode enqueues EmergencyResourceNode
   - Answer: YES - return `next_nodes` in NodeResult

2. How to handle node failures gracefully?
   - Retry logic per node?
   - Fallback nodes?
   - Continue without failed node's data?
   - Answer: Log error, continue, but mark dependency as failed

3. Should background nodes have a timeout?
   - DetoxProtocol might take 30+ seconds
   - Answer: Yes, configurable per node

4. How to persist broker state between requests?
   - For detox notes to surface in next conversation
   - Answer: Phase 4 - separate persistence layer

## Success Criteria

### Phase 1 Complete When:
- [x] All core classes implemented
- [x] Unit tests passing
- [x] Can enqueue and execute simple nodes
- [x] Knowledge broker accumulates context
- [x] Decision engine selects nodes (simple rules)
- [x] Registry manages node lifecycle

**✅ Phase 1 COMPLETE - All criteria met!**

### Phase 2 Complete When:
- [ ] Existing handlers wrapped in nodes
- [ ] Nodes execute same logic as before
- [ ] Integration tests with real handlers

### Phase 3 Complete When:
- [ ] New server replaces old pipeline
- [ ] Same behavior as old system
- [ ] Existing examples/clients work unchanged
- [ ] Performance benchmark acceptable

---

**Current Status**: Ready to implement Phase 1

**Next Step**: Create `src/nodes/` directory and implement core classes
