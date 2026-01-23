#!/usr/bin/env python3
"""DialogueInput format example for LLM RAG Response Pipe.

This example demonstrates the full DialogueInput format including:
- speaker identification
- system_prompt_override for custom personas
- proper JSON structure
"""

import zmq
import json
import sys


def send_dialogue(dealer, pull, content, speaker="user", system_prompt=None):
    """Send a single dialogue and receive response.
    
    Args:
        dealer: ZMQ DEALER socket
        pull: ZMQ PULL socket
        content: Message content
        speaker: Speaker identifier
        system_prompt: Optional system prompt override
        
    Returns:
        tuple: (acknowledgment, response)
    """
    dialogue_input = {
        "content": content,
        "speaker": speaker,
        "system_prompt_override": system_prompt
    }
    
    # Send request
    dealer.send_json(dialogue_input)
    
    # Receive acknowledgment and response
    ack = dealer.recv_string()
    response = pull.recv_string()
    
    return ack, response


def main():
    """Demonstrate DialogueInput format with various configurations."""
    print("="*70)
    print("DialogueInput Format Demo - LLM RAG Response Pipe")
    print("="*70 + "\n")
    
    context = zmq.Context()
    
    # DEALER socket to send requests
    dealer = context.socket(zmq.DEALER)
    dealer.connect("tcp://localhost:5555")
    
    # PULL socket to receive responses
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5556")
    
    print("✓ Connected to pipeline\n")
    
    try:
        # Example 1: Basic dialogue with speaker
        print("-" * 70)
        print("Example 1: Basic dialogue with speaker identification")
        print("-" * 70)
        content = "Hello! How are you today?"
        print(f"User (john): {content}")
        
        ack, response = send_dialogue(dealer, pull, content, speaker="john")
        print(f"ACK: {ack}")
        print(f"Response: {response}\n")
        
        # Example 2: System prompt override for custom persona
        print("-" * 70)
        print("Example 2: Custom persona with system_prompt_override")
        print("-" * 70)
        content = "Explain quantum entanglement"
        system_prompt = "You are a physics professor explaining concepts to undergraduate students. Be clear and use analogies."
        print(f"User: {content}")
        print(f"System Prompt: {system_prompt[:60]}...")
        
        ack, response = send_dialogue(
            dealer, pull, content, 
            speaker="user",
            system_prompt=system_prompt
        )
        print(f"ACK: {ack}")
        print(f"Response: {response}\n")
        
        # Example 3: Different speaker with technical query
        print("-" * 70)
        print("Example 3: Technical query from different speaker")
        print("-" * 70)
        content = "What's the difference between REST and GraphQL?"
        print(f"User (alice): {content}")
        
        ack, response = send_dialogue(dealer, pull, content, speaker="alice")
        print(f"ACK: {ack}")
        print(f"Response: {response}\n")
        
        # Example 4: Custom persona for creative task
        print("-" * 70)
        print("Example 4: Creative task with custom persona")
        print("-" * 70)
        content = "Write a haiku about artificial intelligence"
        system_prompt = "You are a creative poet. Answer poetically and philosophically."
        print(f"User: {content}")
        print(f"System Prompt: {system_prompt}")
        
        ack, response = send_dialogue(
            dealer, pull, content,
            speaker="user",
            system_prompt=system_prompt
        )
        print(f"ACK: {ack}")
        print(f"Response: {response}\n")
        
        print("="*70)
        print("Demo completed successfully!")
        print("="*70)
        
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
