# Unified Thinking Architecture: Tool-Calling Mid-Thought

## Overview

An alternative to the multi-node system where instead of having multiple LLMs processing different aspects in parallel, a **single LLM with extended reasoning** conducts the entire evaluation while calling tools mid-thought to gather information as needed.

## Current Multi-LLM Node System vs Unified Thinking

### Current Approach (Node-Based)
```
User Message
    ↓
Decision LLM → "What nodes should run?"
    ↓
[Parallel/Sequential Execution]
    ├─ Sentiment Analysis LLM  → Analyzes emotion
    ├─ Needs Analysis LLM      → Identifies psychological needs
    ├─ Strategy Selection LLM  → Chooses approach
    ↓
Primary Response LLM → Generates response
```

**Characteristics:**
- Multiple LLM calls
- Each specialized for one task
- Knowledge accumulates in broker
- Higher token cost (multiple contexts)
- Parallel execution possible

### Unified Thinking Approach (Tool-Calling)
```
User Message
    ↓
Single Extended-Thinking LLM:
    "Let me think about this user's message...
     
     <thinking>
     They seem upset. Let me check sentiment.
     [CALLS: sentiment_tool(message)]
     → Returns: {sentiment: 'negative', intensity: 0.8}
     
     Okay, they're quite negative. What are their needs?
     [CALLS: needs_analysis_tool(message, sentiment)]
     → Returns: {primary_need: 'belonging', urgency: 0.7}
     
     They need connection. What memories do we have?
     [CALLS: rag_retrieve_tool(query='loneliness, belonging')]
     → Returns: [memory_1, memory_2, memory_3]
     
     Given their loneliness history, I should use reflective listening.
     [CALLS: strategy_recommend_tool(needs, memories)]
     → Returns: {strategy: 'reflective_listening', tone: 'warm'}
     
     Now I can respond appropriately...
     </thinking>
     
     [Final Response]"
```

**Characteristics:**
- Single LLM call with extended context
- Tools provide specialized information
- LLM reasons about what to check next
- Higher reasoning quality (connected thinking)
- Sequential by nature (thinking flows)

## How It Would Work

### 1. Extended Thinking Models

Modern LLMs with extended thinking capabilities:
- **OpenAI o1/o3**: Built-in reasoning tokens
- **Claude Opus**: Strong chain-of-thought
- **Gemini Deep**: Extended reasoning mode
- **Custom**: Fine-tuned models with thinking prompts

### 2. Tool Registry

Instead of nodes, we define **tools** the LLM can call:

```python
class ToolRegistry:
    """Registry of tools available to the thinking LLM."""
    
    tools = {
        "analyze_sentiment": {
            "description": "Analyze emotional tone and sentiment",
            "parameters": {
                "message": "str - The user's message",
                "context": "dict - Optional conversation context"
            },
            "returns": "SentimentAnalysis object"
        },
        
        "identify_needs": {
            "description": "Identify psychological needs (Maslow hierarchy)",
            "parameters": {
                "message": "str",
                "sentiment": "SentimentAnalysis",
                "memories": "list[Memory]"
            },
            "returns": "NeedsAnalysis object"
        },
        
        "retrieve_memories": {
            "description": "Retrieve relevant past conversations",
            "parameters": {
                "query": "str - Semantic search query",
                "filters": "dict - Optional metadata filters",
                "limit": "int - Max memories to return"
            },
            "returns": "list[Memory]"
        },
        
        "recommend_strategy": {
            "description": "Recommend therapeutic approach",
            "parameters": {
                "needs": "NeedsAnalysis",
                "memories": "list[Memory]",
                "session_state": "SessionState"
            },
            "returns": "StrategyRecommendation"
        },
        
        "check_crisis_indicators": {
            "description": "Detect safety concerns",
            "parameters": {
                "message": "str",
                "conversation_history": "list[Message]"
            },
            "returns": "CrisisAssessment"
        },
        
        "get_detox_notes": {
            "description": "Get self-correction notes from detox protocol",
            "parameters": {
                "topics": "list[str] - Topics to check"
            },
            "returns": "list[DetoxNote]"
        }
    }
```

### 3. Execution Flow

```python
class UnifiedThinkingExecutor:
    """Executes requests using unified thinking + tool calling."""
    
    def __init__(self, llm_provider, tool_registry):
        self.llm = llm_provider  # Must support tool calling
        self.tools = tool_registry
        self.max_iterations = 10  # Prevent infinite loops
    
    async def process_message(self, user_message: str) -> str:
        """Process user message with unified thinking."""
        
        # Build initial prompt with available tools
        system_prompt = self._build_system_prompt()
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        iteration = 0
        
        while iteration < self.max_iterations:
            # LLM thinks and optionally calls tools
            response = await self.llm.generate_with_tools(
                messages=conversation,
                tools=self.tools.get_definitions(),
                thinking_mode=True  # Enable extended reasoning
            )
            
            # Check if LLM is done thinking
            if response.finish_reason == "stop":
                # LLM has final answer
                return response.content
            
            elif response.finish_reason == "tool_calls":
                # LLM wants to call tools
                tool_results = []
                
                for tool_call in response.tool_calls:
                    result = await self._execute_tool(
                        tool_call.name,
                        tool_call.arguments
                    )
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "result": result
                    })
                
                # Add tool results to conversation
                conversation.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })
                conversation.append({
                    "role": "tool",
                    "content": json.dumps(tool_results)
                })
                
                iteration += 1
            
            else:
                # Unexpected finish reason
                logger.error(f"Unexpected finish: {response.finish_reason}")
                break
        
        # Fallback if max iterations reached
        return "I apologize, I'm having trouble formulating a response."
    
    async def _execute_tool(self, tool_name: str, arguments: dict):
        """Execute a tool and return results."""
        
        if tool_name == "analyze_sentiment":
            return await self._tool_sentiment(
                arguments.get("message"),
                arguments.get("context")
            )
        
        elif tool_name == "identify_needs":
            return await self._tool_needs_analysis(
                arguments.get("message"),
                arguments.get("sentiment"),
                arguments.get("memories")
            )
        
        elif tool_name == "retrieve_memories":
            return await self._tool_retrieve_memories(
                arguments.get("query"),
                arguments.get("filters"),
                arguments.get("limit", 10)
            )
        
        # ... other tools
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        return """
You are a thoughtful AI companion focused on mental health support.

When responding to users:
1. Think carefully about their emotional state and needs
2. Use available tools to gather information
3. Reason about the best approach before responding
4. Provide authentic, psychologically-informed support

Available tools:
{tool_descriptions}

Think step-by-step, calling tools as needed to understand the user fully.
"""
```

## Advantages of Unified Thinking

### 1. **Coherent Reasoning**
```
Multi-LLM: Each component reasons in isolation
  Sentiment LLM: "User is sad"
  Needs LLM: "User needs belonging" (doesn't know sentiment)
  Strategy LLM: "Use validation" (doesn't know needs reasoning)

Unified: Single continuous thought process
  "User is sad [check sentiment] → intensity 0.8 → quite upset →
   what do they need? [check needs] → belonging → makes sense given
   sadness → what have they said before? [retrieve memories] →
   history of loneliness → I should validate and gently explore..."
```

### 2. **Dynamic Tool Selection**
The LLM decides **what** to check and **when**, not a pre-programmed decision engine:

```python
# Multi-LLM: Fixed decision tree
if crisis_keywords:
    nodes = ["crisis", "sentiment", "response"]
else:
    nodes = ["sentiment", "response"]

# Unified: LLM decides
<thinking>
Hmm, they mentioned "can't go on" - that's concerning.
Let me check for crisis indicators [CALLS crisis_tool]
→ Returns: moderate_risk=True

Okay, this is serious. I should understand the context better.
[CALLS retrieve_memories with filter=crisis_history]
→ Returns: [previous_crisis_3_months_ago]

They've struggled before. I need to be careful and supportive.
[CALLS recommend_strategy with crisis_context]
→ Returns: {approach: 'validation + resources'}
</thinking>
```

### 3. **Contextual Tool Calling**
Tools are called **only when needed** based on conversation:

```
Simple "How are you?" → No tools needed
Complex emotional disclosure → Multiple tools called
Crisis situation → Specific crisis tools prioritized
```

### 4. **Better Integration**
Tool results inform subsequent reasoning organically:

```
<thinking>
[Sentiment: negative, 0.8]
That's quite negative. Is this unusual for them?
[Retrieve memories → mostly neutral/positive recently]
Interesting - they're usually more upbeat. Something happened.
This warrants careful exploration...
</thinking>
```

## Disadvantages & Challenges

### 1. **Sequential Processing**
- Tools must be called one-by-one (no parallel execution)
- Higher latency for complex requests
- **Mitigation**: Cache tool results, optimize tool speed

### 2. **Token Cost**
- Extended thinking uses more tokens
- Each tool call adds to context
- **Mitigation**: Efficient tool outputs, context pruning

### 3. **Reliability**
- LLM might not call needed tools
- Could call wrong tools
- Might reason incorrectly
- **Mitigation**: Strong system prompts, guardrails, fallbacks

### 4. **Observability**
- Harder to debug (all in one big thought)
- Can't easily measure individual components
- **Mitigation**: Structured thinking output, logging

## Hybrid Approach: Best of Both Worlds

Combine unified thinking with node system:

```python
class HybridExecutor:
    """Hybrid: Unified thinking for decision-making, nodes for execution."""
    
    async def process_message(self, user_message: str):
        # Phase 1: Unified thinking to plan
        plan = await self.thinking_llm.plan_with_tools(
            message=user_message,
            available_tools=["analyze_context", "check_history", "assess_needs"]
        )
        # Plan: {
        #   sentiment_needed: True,
        #   needs_analysis: True,
        #   strategy: "reflective_listening",
        #   urgency: "medium"
        # }
        
        # Phase 2: Execute plan via node system
        broker = KnowledgeBroker()
        broker.add_knowledge("plan", plan)
        broker.add_knowledge("user_message", user_message)
        
        # Create nodes based on plan
        nodes = []
        if plan.sentiment_needed:
            nodes.append(SentimentAnalysisNode())
        if plan.needs_analysis:
            nodes.append(NeedsAnalysisNode())
        
        nodes.append(PrimaryResponseNode(strategy=plan.strategy))
        
        # Execute nodes
        for node in nodes:
            await queue_manager.enqueue(node)
        
        await queue_manager.execute_immediate(broker)
        
        return broker.get_knowledge("response")
```

## Implementation Strategy

### Option 1: Full Unified (Simple Cases)
For straightforward conversations:
```python
if is_simple_conversation(message):
    # Just use thinking LLM with minimal tools
    return await unified_executor.process_message(message)
```

### Option 2: Hybrid (Complex Cases)
For sophisticated processing:
```python
if is_complex_conversation(message):
    # Use thinking LLM to plan
    plan = await thinking_llm.create_plan(message)
    
    # Execute plan via nodes
    return await node_executor.execute_plan(plan)
```

### Option 3: Adaptive
Let the system decide:
```python
# Meta-decision: Which architecture to use?
approach = await meta_llm.decide_approach(
    message=message,
    conversation_state=state
)

if approach == "unified":
    return await unified_executor.process(message)
elif approach == "nodes":
    return await node_executor.process(message)
elif approach == "hybrid":
    return await hybrid_executor.process(message)
```

## Tool Design Principles

### 1. **Deterministic Tools**
Tools should return consistent results:
```python
# Good: Deterministic
def analyze_sentiment(message: str) -> SentimentAnalysis:
    return sentiment_model.analyze(message)

# Bad: Non-deterministic
def analyze_sentiment(message: str) -> SentimentAnalysis:
    return llm.ask("What's the sentiment?")  # LLM inside tool
```

### 2. **Fast Tools**
Tools should execute quickly (<100ms ideal):
```python
# Good: Direct computation
def retrieve_memories(query: str) -> list[Memory]:
    embedding = embed_service.encode(query)
    return vector_db.search(embedding, limit=5)

# Bad: Slow processing
def retrieve_memories(query: str) -> list[Memory]:
    all_memories = db.get_all()  # Could be huge
    return [m for m in all_memories if is_relevant(m, query)]
```

### 3. **Clear Outputs**
Tool results should be interpretable:
```python
# Good: Structured, clear
{
    "sentiment": "negative",
    "confidence": 0.85,
    "primary_emotion": "sadness",
    "intensity": 0.7
}

# Bad: Ambiguous
{
    "result": "kinda sad maybe",
    "score": 42
}
```

### 4. **Self-Contained**
Tools shouldn't have side effects (mostly):
```python
# Good: Read-only
def get_memories(query: str) -> list[Memory]:
    return memory_store.search(query)

# Acceptable: Clear side effect
def store_conversation(message: str, response: str):
    """Stores conversation. Side effect: writes to DB."""
    db.insert(message, response)

# Bad: Hidden side effect
def analyze_message(message: str) -> Analysis:
    db.insert(message)  # Hidden side effect!
    return Analysis(...)
```

## Real-World Example

### User Message
"I can't seem to focus on anything today. Everything feels pointless."

### Unified Thinking Process
```
<thinking>
This message has concerning elements. Let me assess:

1. Emotional state check:
   [CALLS: analyze_sentiment(message)]
   → Result: {
       sentiment: "negative",
       intensity: 0.75,
       emotions: ["hopelessness", "apathy"],
       concern_level: "moderate"
     }

2. This shows hopelessness and apathy - potential depression or crisis?
   [CALLS: check_crisis_indicators(message)]
   → Result: {
       crisis_score: 0.4,  # Moderate, not immediate crisis
       indicators: ["apathy", "lack_of_meaning"],
       requires_immediate_intervention: False
     }

3. Not immediate crisis, but worth understanding context:
   [CALLS: retrieve_memories(query="depression, motivation, focus")]
   → Result: [
       {text: "User mentioned feeling unmotivated 2 weeks ago", score: 0.9},
       {text: "User usually enjoys painting but hasn't mentioned it lately", score: 0.7}
     ]

4. Pattern emerging - declining motivation over 2 weeks. What needs are unmet?
   [CALLS: identify_needs(message, sentiment, memories)]
   → Result: {
       primary_needs: ["meaning", "autonomy"],
       secondary_needs: ["esteem"],
       need_urgency: 0.6
     }

5. Loss of meaning and autonomy. Given moderate concern level and pattern:
   [CALLS: recommend_strategy(needs, crisis_assessment, memories)]
   → Result: {
       strategy: "validation_and_exploration",
       tone: "warm_and_gentle",
       avoid: ["toxic_positivity", "minimizing"],
       approach: "acknowledge_feeling_then_gently_explore_context"
     }

6. I have enough information. Response plan:
   - Validate the feeling (not minimize)
   - Gently explore what might be contributing
   - Acknowledge the difficulty
   - Avoid forcing solutions
</thinking>

[Assistant Response]:
"That sounds really difficult - when everything feels pointless, it can be 
hard to find motivation for anything. You mentioned you haven't been able to 
focus either, which adds another layer of frustration.

I'm wondering, has something changed recently, or has this been building for 
a while? You don't have to go into details if you don't want to, but I'm here 
if you want to talk about it."
```

## When to Use Unified vs Nodes

### Use Unified Thinking When:
✅ Conversation requires nuanced decision-making  
✅ Context heavily influences what to check  
✅ Fewer than 3-4 tools likely needed  
✅ Reasoning quality > parallel speed  
✅ Using o1/o3 or other reasoning models  

### Use Node System When:
✅ Fixed workflow (always same steps)  
✅ Many independent analyses needed  
✅ Parallel execution beneficial  
✅ Component reusability important  
✅ Debugging/monitoring critical  

### Use Hybrid When:
✅ Complex multi-stage processing  
✅ Planning phase + execution phase distinct  
✅ Best of both approaches needed  

## Future: Agentic Thinking

Extended thinking could evolve into full agency:

```python
<thinking>
User wants to reconnect with old hobbies. Let me help proactively:

1. [CALLS: retrieve_memories(query="hobbies, interests")]
   → Painting mentioned 6 times, last 3 months ago

2. They used to paint! I should check if there's local art classes:
   [CALLS: search_local_resources(category="art_classes", location=user.city)]
   → Found 3 beginner-friendly classes

3. I should save this for later in case they're interested:
   [CALLS: create_reminder(
       content="Suggest art classes if user shows interest",
       trigger="next_conversation_about_hobbies"
   )]

4. For now, just gently remind them about painting...
</thinking>

"I remember you used to enjoy painting! Have you thought about picking that 
up again? Sometimes returning to old hobbies can help when things feel 
off-balance."
```

## Conclusion

Unified thinking with tool calling offers:
- **More coherent reasoning** (single thought process)
- **Dynamic adaptation** (LLM decides what to check)
- **Contextual efficiency** (only call needed tools)

But requires:
- Models with strong reasoning + tool calling
- Well-designed, fast tools
- Careful prompt engineering
- Guardrails against reasoning failures

The **hybrid approach** likely optimal:
- Use thinking LLM for **planning and decision-making**
- Use node system for **execution and parallel processing**
- Get benefits of both: coherent reasoning + efficient execution

As reasoning models improve (o1 → o3 → future), unified thinking becomes more viable. The node system we built provides flexibility to adopt either approach or switch between them based on conversation complexity.
