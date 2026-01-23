#!/usr/bin/env python3
"""RAG retrieval and memory decay demonstration for LLM RAG Response Pipe.

This example demonstrates:
- How RAG retrieves relevant context from conversation history
- Memory decay algorithm in action (recent vs old memories)
- Semantic search across stored conversations
- How context improves response quality
"""

import zmq
import time
import sys


def send_message(dealer, pull, content, speaker="user"):
    """Send a message and receive response.
    
    Args:
        dealer: ZMQ DEALER socket
        pull: ZMQ PULL socket
        content: Message content
        speaker: Speaker identifier
        
    Returns:
        tuple: (acknowledgment, response)
    """
    dialogue_input = {
        "content": content,
        "speaker": speaker
    }
    
    dealer.send_json(dialogue_input)
    ack = dealer.recv_string()
    response = pull.recv_string()
    
    return ack, response


def main():
    """Demonstrate RAG retrieval and memory features."""
    print("="*70)
    print("RAG Retrieval & Memory Decay Demo - LLM RAG Response Pipe")
    print("="*70)
    print("""
This demo demonstrates how the pipeline:
1. Stores conversations in Qdrant with embeddings
2. Retrieves relevant context using semantic search
3. Uses memory decay to prioritize recent/important memories
4. Provides contextually aware responses

We'll have a conversation, then query about past topics.
    """)
    
    context = zmq.Context()
    
    dealer = context.socket(zmq.DEALER)
    dealer.connect("tcp://localhost:5555")
    
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5556")
    
    print("✓ Connected to pipeline\n")
    
    try:
        print("="*70)
        print("Phase 1: Building Conversation History")
        print("="*70)
        
        # Build up conversation history with different topics
        history_messages = [
            ("My favorite programming language is Python.", "Establishing preference"),
            ("I work as a data scientist at a tech company.", "Career context"),
            ("I've been working on a machine learning project about image classification.", "Current project"),
            ("My dog's name is Max and he's a golden retriever.", "Personal info"),
            ("I'm planning to visit Japan next summer for vacation.", "Future plans"),
        ]
        
        for msg, description in history_messages:
            print(f"\n📝 {description}")
            print(f"   Message: {msg}")
            ack, response = send_message(dealer, pull, msg)
            print(f"   Response: {response[:100]}...")
            time.sleep(0.3)
        
        print(f"\n{'='*70}")
        print("Phase 2: Testing RAG Retrieval")
        print("="*70)
        print("\nNow we'll ask questions that require retrieving stored context...\n")
        
        time.sleep(1)
        
        # Test queries that should retrieve different parts of history
        test_queries = [
            ("What programming language do I prefer?", "Should retrieve: Python preference"),
            ("What's my dog's name and breed?", "Should retrieve: Max, golden retriever"),
            ("What project am I working on?", "Should retrieve: ML image classification"),
            ("Tell me about my job", "Should retrieve: Data scientist at tech company"),
            ("What are my travel plans?", "Should retrieve: Japan vacation"),
            ("Can you recommend Python libraries for my ML project?", "Should combine: Python + ML project context"),
        ]
        
        for query, expected in test_queries:
            print(f"{'─'*70}")
            print(f"🔍 Query: {query}")
            print(f"📋 Expected retrieval: {expected}")
            
            ack, response = send_message(dealer, pull, query)
            
            print(f"💬 Response: {response}")
            time.sleep(1.5)
        
        print(f"\n{'='*70}")
        print("Phase 3: Semantic Search Test")
        print("="*70)
        print("\nTesting semantic similarity (not just keyword matching)...\n")
        
        semantic_queries = [
            ("What kind of pet do I have?", "Should find: dog Max (semantic: pet → dog)"),
            ("Where am I going on holiday?", "Should find: Japan trip (semantic: holiday → vacation)"),
            ("What's my occupation?", "Should find: data scientist (semantic: occupation → job)"),
        ]
        
        for query, expected in semantic_queries:
            print(f"{'─'*70}")
            print(f"🔍 Query: {query}")
            print(f"📋 {expected}")
            
            ack, response = send_message(dealer, pull, query)
            print(f"💬 Response: {response}")
            time.sleep(1.5)
        
        # Summary
        print(f"\n{'='*70}")
        print("Demo completed!")
        print("="*70)
        print("""
💡 Key Observations:

1. RAG Storage: All conversations are automatically stored in Qdrant with embeddings
2. Semantic Search: The system finds relevant context even without exact keyword matches
3. Context Retrieval: Past conversations are retrieved to answer new questions
4. Memory Decay: Recent and high chrono-relevance events are prioritized
5. Multi-Context: The system can combine multiple pieces of historical context

🔬 Behind the Scenes:
- SQLite stores ordered conversation history (fast sequential access)
- Qdrant stores embeddings for semantic search (vector similarity)
- Memory decay algorithm weights recent and important memories higher
- Context interpreter reformulates retrieved documents for the primary LLM

📊 Check your pipeline logs (set log_level=DEBUG) to see:
- Qdrant similarity scores
- Memory decay calculations
- Number of documents retrieved
- Context selection decisions
        """)
        
    except zmq.ZMQError as e:
        print(f"\nError communicating with pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    
    finally:
        dealer.close()
        pull.close()
        context.term()


if __name__ == "__main__":
    main()
