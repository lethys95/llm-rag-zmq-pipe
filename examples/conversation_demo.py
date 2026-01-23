#!/usr/bin/env python3
"""Multi-turn conversation demo with memory for LLM RAG Response Pipe.

This example demonstrates:
- Multi-turn conversations with context retention
- How the pipeline uses RAG to retrieve previous conversation history
- Memory persistence across multiple exchanges
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


def display_exchange(turn_num, user_msg, response, ack=None):
    """Display a conversation exchange in a formatted way."""
    print(f"\n{'─'*70}")
    print(f"Turn {turn_num}")
    print(f"{'─'*70}")
    print(f"🗣️  User: {user_msg}")
    if ack:
        print(f"📝  ACK: {ack}")
    print(f"🤖  Assistant: {response}")


def main():
    """Run a multi-turn conversation demonstrating memory and context."""
    print("="*70)
    print("Multi-Turn Conversation Demo - LLM RAG Response Pipe")
    print("="*70)
    print("\nThis demo shows how the pipeline remembers conversation history")
    print("and uses it to provide contextually aware responses.\n")
    
    context = zmq.Context()
    
    dealer = context.socket(zmq.DEALER)
    dealer.connect("tcp://localhost:5555")
    
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5556")
    
    print("✓ Connected to pipeline")
    print("\nStarting conversation...\n")
    
    try:
        # Conversation sequence demonstrating memory
        conversations = [
            "My name is Alice and I'm learning about machine learning.",
            "What are the main types of machine learning?",
            "Can you explain supervised learning in more detail?",
            "What's my name again?",  # Tests memory recall
            "What topic am I learning about?",  # Tests context retention
            "Could you recommend some beginner projects for what I'm studying?",  # Tests complex context understanding
        ]
        
        for i, message in enumerate(conversations, 1):
            # Small delay between messages to simulate natural conversation
            if i > 1:
                time.sleep(0.5)
            
            ack, response = send_message(dealer, pull, message, speaker="alice")
            display_exchange(i, message, response, ack)
        
        print(f"\n{'='*70}")
        print("Conversation completed!")
        print("="*70)
        print("\n💡 Notice how the assistant remembered:")
        print("   - Your name (Alice)")
        print("   - The topic (machine learning)")
        print("   - Previous discussion context (supervised learning)")
        print("\nThis demonstrates the RAG system retrieving and using")
        print("stored conversation history for contextually aware responses.")
        
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
