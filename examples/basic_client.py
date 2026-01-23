#!/usr/bin/env python3
"""Basic client example for LLM RAG Response Pipe.

This is the simplest possible example of using the pipeline.
It sends a single request and receives the response.
"""

import zmq
import json
import sys


def main():
    """Send a basic request to the pipeline and receive response."""
    print("Connecting to LLM RAG Response Pipe...")
    
    context = zmq.Context()
    
    # DEALER socket to send requests
    dealer = context.socket(zmq.DEALER)
    dealer.connect("tcp://localhost:5555")
    
    # PULL socket to receive responses
    pull = context.socket(zmq.PULL)
    pull.bind("tcp://*:5556")
    
    print("Connected successfully.\n")
    
    # Create a DialogueInput message
    dialogue_input = {
        "content": "What is the capital of France?",
        "speaker": "user"
    }
    
    print(f"Sending request: {dialogue_input['content']}")
    
    try:
        # Send request
        dealer.send_json(dialogue_input)
        
        # Receive acknowledgment
        ack = dealer.recv_string(flags=zmq.NOBLOCK if False else 0)
        print(f"\n✓ Acknowledgment: {ack}\n")
        
        # Receive response
        print("Waiting for response...")
        response = pull.recv_string()
        
        print(f"\n{'='*60}")
        print("Response:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
        
    except zmq.ZMQError as e:
        print(f"Error communicating with pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    
    finally:
        dealer.close()
        pull.close()
        context.term()


if __name__ == "__main__":
    main()
