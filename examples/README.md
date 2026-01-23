# Example Scripts for LLM RAG Response Pipe

This directory contains working example scripts demonstrating various features of the LLM RAG Response Pipe.

## Prerequisites

1. **Start the pipeline server:**
   ```bash
   # Set your OpenRouter API key
   export OPENROUTER_API_KEY="your-api-key-here"
   
   # Start Qdrant (if using RAG)
   docker-compose up -d
   
   # Start the pipeline
   llm-rag-pipe remote
   ```

2. **Install dependencies:**
   ```bash
   pip install pyzmq
   ```

## Example Scripts

### 1. basic_client.py

**Purpose:** Simplest possible usage - send one request, get one response.

**What it demonstrates:**
- Basic ZMQ socket setup (DEALER + PULL)
- DialogueInput JSON format
- Sending requests and receiving responses

**Run it:**
```bash
python examples/basic_client.py
```

**Expected output:**
```
Connecting to LLM RAG Response Pipe...
Connected successfully.

Sending request: What is the capital of France?

✓ Acknowledgment: Request processed successfully | Sentiment: neutral

Waiting for response...

============================================================
Response:
============================================================
The capital of France is Paris.
============================================================
```

---

### 2. dialogue_client.py

**Purpose:** Demonstrate the full DialogueInput format including speaker identification and system prompt overrides.

**What it demonstrates:**
- Speaker identification field
- System prompt override for custom personas
- Multiple request examples with different configurations

**Run it:**
```bash
python examples/dialogue_client.py
```

**Key features:**
- Shows how to identify different speakers
- Demonstrates custom system prompts for different use cases
- Examples of technical, creative, and educational queries

---

### 3. conversation_demo.py

**Purpose:** Demonstrate multi-turn conversations with memory retention.

**What it demonstrates:**
- Multi-turn conversation flow
- How the pipeline remembers previous context
- RAG-based memory retrieval
- Context-aware responses across multiple exchanges

**Run it:**
```bash
python examples/conversation_demo.py
```

**What it tests:**
- Name recall ("What's my name?")
- Topic retention ("What am I learning about?")
- Complex context understanding (combining multiple pieces of information)

**Note:** This script pauses between messages to simulate natural conversation timing.

---

### 4. sentiment_aware_client.py

**Purpose:** Demonstrate sentiment analysis capabilities.

**What it demonstrates:**
- Positive, negative, and neutral sentiment classification
- Relevance scoring (how important the topic is)
- Chrono-relevance scoring (how long the topic stays relevant)
- How sentiment affects acknowledgments and memory storage

**Run it:**
```bash
python examples/sentiment_aware_client.py
```

**Example scenarios:**
- **High relevance, high chrono-relevance:** "My mother passed away" (long-lasting importance)
- **High relevance, low chrono-relevance:** "I need the bathroom urgently" (temporary but urgent)
- **Low relevance:** "What's the weather?" (casual, not important)

**Tip:** Check the acknowledgment messages to see sentiment classifications!

---

### 5. rag_retrieval_demo.py

**Purpose:** Comprehensive demonstration of RAG retrieval and semantic search.

**What it demonstrates:**
- How conversations are stored in Qdrant with embeddings
- Semantic search across conversation history
- Memory decay algorithm prioritizing recent/important memories
- Context retrieval from multiple stored conversations

**Run it:**
```bash
python examples/rag_retrieval_demo.py
```

**Three phases:**
1. **Building History:** Stores various pieces of information
2. **Testing Retrieval:** Queries that require retrieving stored context
3. **Semantic Search:** Tests semantic similarity (not just keyword matching)

**Examples of semantic search:**
- "What kind of pet do I have?" → finds "dog"
- "Where am I going on holiday?" → finds "vacation"
- "What's my occupation?" → finds "job"

**Advanced tip:** Run the pipeline with `--log-level DEBUG` to see:
- Qdrant similarity scores
- Memory decay calculations
- Number of documents retrieved
- Context selection decisions

---

### 6. test_pipeline.py

**Purpose:** Comprehensive end-to-end testing of all pipeline features.

**What it tests:**
1. ✅ Basic connectivity
2. ✅ DialogueInput format handling
3. ✅ Multi-turn conversation memory
4. ✅ Sentiment analysis
5. ✅ RAG retrieval
6. ✅ Error handling

**Run it:**
```bash
python examples/test_pipeline.py
```

**Output includes:**
- Pass/fail status for each test
- Detailed diagnostics
- Overall success rate
- Recommendations based on results

**Use case:** Run this script to verify your pipeline is working correctly after:
- Initial setup
- Configuration changes
- Upgrading dependencies
- Debugging issues

---

## Common Issues & Solutions

### Issue: "Connection refused" error

**Solution:**
```bash
# Make sure the pipeline is running first
llm-rag-pipe remote

# In another terminal, run the example
python examples/basic_client.py
```

### Issue: "No module named 'zmq'"

**Solution:**
```bash
pip install pyzmq
```

### Issue: Examples hang or timeout

**Possible causes:**
1. Pipeline is not running
2. Wrong endpoint configuration
3. Firewall blocking connections

**Solution:**
```bash
# Check if pipeline is listening
netstat -an | grep 5555

# Try with explicit endpoints
python examples/basic_client.py
```

### Issue: Sentiment not showing in acknowledgments

**Explanation:** Sentiment analysis may be disabled in your configuration.

**Solution:**
Check your config file or use CLI flag:
```bash
llm-rag-pipe remote --config-file config.json
# Ensure: "enable_sentiment_analysis": true
```

### Issue: RAG not retrieving context

**Possible causes:**
1. Qdrant not running
2. RAG disabled in configuration
3. Embeddings not being generated

**Solution:**
```bash
# Start Qdrant
docker-compose up -d

# Check RAG is enabled
llm-rag-pipe remote --rag-enabled
```

---

## Customization

### Changing Endpoints

All examples connect to default endpoints:
- Input: `tcp://localhost:5555`
- Output: `tcp://*:5556`

To change, edit the example files or create a wrapper script:

```python
from basic_client import main

# Modify connection strings in the script
# Or pass as environment variables in your application
```

### Adding Your Own Examples

Template for creating new example scripts:

```python
#!/usr/bin/env python3
import zmq
import json

def main():
    context = zmq.Context()
    
    # DEALER for sending requests
    dealer = context.socket(zmq.DEALER)
    dealer.connect("tcp://localhost:5555")
    
    # PULL for receiving responses
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5556")
    
    # Send request
    dialogue_input = {
        "content": "Your message here",
        "speaker": "user"
    }
    dealer.send_json(dialogue_input)
    
    # Receive response
    ack = dealer.recv_string()
    response = pull.recv_string()
    
    print(f"ACK: {ack}")
    print(f"Response: {response}")
    
    # Cleanup
    dealer.close()
    pull.close()
    context.term()

if __name__ == "__main__":
    main()
```

---

## Running Examples in Sequence

To see the full capabilities, run examples in this order:

```bash
# 1. Basic functionality
python examples/basic_client.py

# 2. Format demonstration
python examples/dialogue_client.py

# 3. Memory capabilities
python examples/conversation_demo.py

# 4. Sentiment features
python examples/sentiment_aware_client.py

# 5. RAG capabilities
python examples/rag_retrieval_demo.py

# 6. Comprehensive testing
python examples/test_pipeline.py
```

---

## Integration Examples

### Using in Your Application

```python
import zmq
import json

class PipelineClient:
    """Simple client wrapper for the pipeline."""
    
    def __init__(self):
        self.context = zmq.Context()
        self.dealer = self.context.socket(zmq.DEALER)
        self.pull = self.context.socket(zmq.PULL)
        
        self.dealer.connect("tcp://localhost:5555")
        self.pull.bind("tcp://*:5556")
    
    def send(self, message, speaker="user"):
        """Send message and get response."""
        dialogue = {
            "content": message,
            "speaker": speaker
        }
        
        self.dealer.send_json(dialogue)
        ack = self.dealer.recv_string()
        response = self.pull.recv_string()
        
        return response
    
    def close(self):
        """Clean up resources."""
        self.dealer.close()
        self.pull.close()
        self.context.term()

# Usage
client = PipelineClient()
response = client.send("Hello!")
print(response)
client.close()
```

---

## Next Steps

1. **Experiment with the examples** - Modify messages, speakers, and parameters
2. **Check the logs** - Run pipeline with `--log-level DEBUG` to see internals
3. **Read the documentation** - See `../docs/` for detailed feature documentation
4. **Build your application** - Use examples as templates for your use case

---

## Additional Resources

- **Main README:** `../README.md`
- **RAG Features:** `../docs/RAG_FEATURES.md`
- **Memory Decay:** `../docs/MEMORY_DECAY_ALGORITHM.md`
- **Configuration:** `../docs/QUICK_START_RAG.md`

---

## Getting Help

If you encounter issues:

1. Check the pipeline logs for error messages
2. Verify Qdrant is running: `docker ps | grep qdrant`
3. Test with the simplest example first (`basic_client.py`)
4. Run the test suite (`test_pipeline.py`) to identify specific issues
5. Review configuration in `example_config.json`

Happy coding! 🚀
