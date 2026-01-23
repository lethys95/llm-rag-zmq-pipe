#!/usr/bin/env python3
"""Sentiment analysis demonstration for LLM RAG Response Pipe.

This example demonstrates:
- How sentiment analysis affects acknowledgments
- Different sentiment classifications (positive, negative, neutral)
- Relevance and chrono-relevance scoring
- How sentiment influences memory storage and retrieval
"""

import zmq
import time
import sys


def send_message(dealer, pull, content, speaker="user"):
    """Send a message and receive response with sentiment info.
    
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


def display_exchange(content, ack, response, description=""):
    """Display an exchange with sentiment information highlighted."""
    print(f"\n{'─'*70}")
    if description:
        print(f"📋 {description}")
        print(f"{'─'*70}")
    print(f"💬 Message: {content}")
    print(f"📊 ACK: {ack}")
    print(f"💭 Response: {response[:200]}{'...' if len(response) > 200 else ''}")


def main():
    """Demonstrate sentiment analysis features."""
    print("="*70)
    print("Sentiment Analysis Demo - LLM RAG Response Pipe")
    print("="*70)
    print("""
This demo shows how the pipeline analyzes sentiment in messages:
- Positive, negative, and neutral classifications
- Relevance scoring (how important the topic is)
- Chrono-relevance (how long the topic stays relevant)

The acknowledgment (ACK) includes sentiment information.
    """)
    
    context = zmq.Context()
    
    dealer = context.socket(zmq.DEALER)
    dealer.connect("tcp://localhost:5555")
    
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5556")
    
    print("✓ Connected to pipeline\n")
    
    try:
        # Example 1: Positive sentiment
        content = "I just got promoted at work! I'm so excited and happy!"
        ack, response = send_message(dealer, pull, content)
        display_exchange(
            content, ack, response,
            "Example 1: Positive Sentiment (high relevance, medium chrono-relevance)"
        )
        time.sleep(1)
        
        # Example 2: Negative sentiment with high chrono-relevance
        content = "My mother passed away last week. I'm still grieving."
        ack, response = send_message(dealer, pull, content)
        display_exchange(
            content, ack, response,
            "Example 2: Negative Sentiment (high relevance, HIGH chrono-relevance)"
        )
        time.sleep(1)
        
        # Example 3: Urgent but temporary (high relevance, low chrono-relevance)
        content = "I really need to use the bathroom right now, this is urgent!"
        ack, response = send_message(dealer, pull, content)
        display_exchange(
            content, ack, response,
            "Example 3: Urgent/Temporary (high relevance, LOW chrono-relevance)"
        )
        time.sleep(1)
        
        # Example 4: Neutral sentiment
        content = "What's the weather like today?"
        ack, response = send_message(dealer, pull, content)
        display_exchange(
            content, ack, response,
            "Example 4: Neutral Sentiment (low relevance)"
        )
        time.sleep(1)
        
        # Example 5: Stress/anxiety (medium-term relevance)
        content = "I have a final exam tomorrow and I'm really stressed about it."
        ack, response = send_message(dealer, pull, content)
        display_exchange(
            content, ack, response,
            "Example 5: Stress/Anxiety (medium relevance, low chrono-relevance)"
        )
        time.sleep(1)
        
        # Example 6: Test memory recall of high chrono-relevance event
        content = "How should I cope with what happened to my mother?"
        ack, response = send_message(dealer, pull, content)
        display_exchange(
            content, ack, response,
            "Example 6: Memory Recall (references high chrono-relevance event)"
        )
        
        # Summary
        print(f"\n{'='*70}")
        print("Demo completed!")
        print("="*70)
        print("""
💡 Key Takeaways:

1. Sentiment Analysis: The pipeline classifies messages as positive/negative/neutral
2. Relevance Scoring: Measures how important the topic is (0.0 to 1.0)
3. Chrono-Relevance: Measures how long the topic stays relevant
   - High: Long-term events (grief, major life changes)
   - Low: Temporary/urgent (bathroom, exam tomorrow)
   - Medium: Ongoing concerns (work stress, relationships)

4. Memory Impact: High chrono-relevance events are preserved longer in memory
5. Context Awareness: The assistant can recall and reference previous high-relevance topics

Check your pipeline logs to see detailed sentiment analysis scores!
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
