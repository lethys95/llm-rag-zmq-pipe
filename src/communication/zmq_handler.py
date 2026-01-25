"""ZMQ socket handler for ROUTER and DEALER patterns."""

import logging
import zmq
import json
import msgpack
from pydantic import ValidationError

from src.models import DialogueInput

logger = logging.getLogger(__name__)


class ZMQHandler:
    """Handler for ZMQ ROUTER and DEALER sockets.
    
    Manages the communication layer:
    - ROUTER socket: Receives prompts from STT clients, sends ACKs back
    - DEALER socket: Forwards responses to downstream pipeline, can receive error feedback
    
    Architecture:
    STT → [ROUTER] → This Service → [DEALER] → Downstream
         ← [ROUTER] ← (ACKs)        ← [DEALER] ← (Error feedback)
    """
    
    def __init__(self, input_endpoint: str, output_endpoint: str):
        """Initialize ZMQ handler.
        
        Args:
            input_endpoint: Endpoint to bind ROUTER socket (e.g., "tcp://*:5555")
            output_endpoint: Endpoint to connect DEALER socket (e.g., "tcp://localhost:5556")
        """
        self.input_endpoint = input_endpoint
        self.output_endpoint = output_endpoint
        
        self.context = zmq.Context()
        self.router_socket = None
        self.dealer_socket = None
        self.poller = None
        
        logger.info(f"ZMQ Handler initialized: input={input_endpoint}, output={output_endpoint}")
    
    def setup(self) -> None:
        """Set up ZMQ sockets and poller.
        
        Creates and configures the ROUTER and DEALER sockets.
        """
        # Create ROUTER socket for receiving requests from STT
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(self.input_endpoint)
        logger.info(f"ROUTER socket bound to {self.input_endpoint}")
        
        # Create DEALER socket for bidirectional communication with downstream
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.dealer_socket.connect(self.output_endpoint)
        logger.info(f"DEALER socket connected to {self.output_endpoint}")
        
        # Set up poller for non-blocking receive on both sockets
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        self.poller.register(self.dealer_socket, zmq.POLLIN)
        
        logger.info("ZMQ sockets and poller set up successfully")
    
    def receive_request(self, timeout: int = 1000) -> tuple[list[bytes], DialogueInput | None]:
        """Receive and parse a request from ROUTER socket.
        
        Args:
            timeout: Timeout in milliseconds for polling
            
        Returns:
            Tuple of (identity frames, DialogueInput object)
            Returns (None, None) if no message received within timeout
        """
        frames = self._poll_router_socket(timeout)
        if not frames:
            return None, None
        
        identity, message_bytes = self._extract_frames(frames)
        dialogue_input = self._parse_message(message_bytes)
        
        return identity, dialogue_input
    
    def _poll_router_socket(self, timeout: int) -> list[bytes] | None:
        """Poll ROUTER socket for incoming messages."""
        socks = dict(self.poller.poll(timeout))
        
        if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
            return self.router_socket.recv_multipart()
        
        return None
    
    def _extract_frames(self, frames: list[bytes]) -> tuple[list[bytes], bytes]:
        """Extract identity and message from ROUTER multipart frames.
        
        ROUTER frames: [identity_part1, identity_part2, ..., empty_delimiter, message]
        """
        return frames[:-1], frames[-1]
    
    def _parse_message(self, message_bytes: bytes) -> DialogueInput | None:
        """Parse message bytes into DialogueInput.
        
        Attempts deserialization in order: msgpack → JSON
        Handles both STT TranscriptionResponse and direct DialogueInput formats.
        """
        message_data = self._deserialize_message(message_bytes)
        if not message_data:
            return None
        
        return self._extract_dialogue_input(message_data)
    
    def _deserialize_message(self, message_bytes: bytes) -> dict | None:
        """Deserialize message bytes trying msgpack first, then JSON."""
        try:
            return msgpack.unpackb(message_bytes, raw=False)
        except (msgpack.exceptions.UnpackException, TypeError):
            pass
        
        try:
            message_str = message_bytes.decode('utf-8')
            return json.loads(message_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to deserialize message: {e}")
            return None
    
    def _extract_dialogue_input(self, message_data: dict) -> DialogueInput | None:
        """Extract DialogueInput from message data.
        
        Handles two formats:
        1. STT TranscriptionResponse: {text, status, speaker?, error_details?}
        2. Direct DialogueInput: {content, speaker, ...}
        """
        if self._is_stt_response(message_data):
            return self._handle_stt_response(message_data)
        
        return self._create_dialogue_input(message_data)
    
    def _is_stt_response(self, data: dict) -> bool:
        """Check if message is from STT service based on schema."""
        return isinstance(data, dict) and 'text' in data and 'status' in data
    
    def _handle_stt_response(self, data: dict) -> DialogueInput | None:
        """Convert STT TranscriptionResponse to DialogueInput."""
        if data.get('status') != 'success':
            error_details = data.get('error_details', 'Unknown error')
            logger.error(f"STT transcription failed: {error_details}")
            return None
        
        text = data.get('text', '')
        speaker = data.get('speaker', 'User')
        
        dialogue_input = DialogueInput(content=text, speaker=speaker)
        logger.debug(f"Received STT from '{speaker}': {text[:100]}...")
        return dialogue_input
    
    def _create_dialogue_input(self, data: dict) -> DialogueInput | None:
        """Create DialogueInput from dictionary data."""
        try:
            dialogue_input = DialogueInput(**data)
            logger.debug(f"Received message from '{dialogue_input.speaker}': {dialogue_input.content[:100]}...")
            return dialogue_input
        except ValidationError as e:
            logger.error(f"Invalid DialogueInput data: {e}")
            return None
    
    def send_acknowledgment(self, identity: list[bytes], status: str, message: str) -> None:
        """Send acknowledgment back to the requester via ROUTER.
        
        Args:
            identity: Identity frames from the original request
            status: Status string ("success" or "error")
            message: Acknowledgment message
        """
        ack_message = f"[{status.upper()}] {message}"
        ack_bytes = ack_message.encode('utf-8')
        
        # Send back through ROUTER: identity frames + message
        self.router_socket.send_multipart(identity + [ack_bytes])
        
        logger.debug(f"Sent acknowledgment: {status}")
    
    def forward_response(self, response: str) -> None:
        """Forward the LLM response to downstream pipeline via DEALER.
        
        Args:
            response: The generated response to forward
        """
        print(response)
        response_bytes = response.encode('utf-8')
        self.dealer_socket.send(response_bytes)
        
        logger.debug(f"Forwarded response via DEALER: {response[:100]}...")
    
    def check_downstream_feedback(self, timeout: int = 0) -> str | None:
        """Check for feedback/errors from downstream via DEALER (non-blocking).
        
        This allows the service to be informed if downstream components
        report incompatibility or errors with the sent data.
        
        Args:
            timeout: Timeout in milliseconds for polling (0 = non-blocking)
            
        Returns:
            Feedback message from downstream, or None if no message
        """
        try:
            socks = dict(self.poller.poll(timeout))
            
            if self.dealer_socket in socks and socks[self.dealer_socket] == zmq.POLLIN:
                # DEALER receives messages directly
                message_bytes = self.dealer_socket.recv(zmq.NOBLOCK)
                message = message_bytes.decode('utf-8')
                
                logger.warning(f"Received feedback from downstream: {message}")
                return message
                
        except zmq.Again:
            # No message available
            pass
        except Exception as e:
            logger.error(f"Error checking downstream feedback: {e}", exc_info=True)
        
        return None
    
    def close(self) -> None:
        """Close ZMQ sockets and terminate context.
        
        Should be called when shutting down the server.
        """
        logger.info("Closing ZMQ handler")
        
        if self.router_socket:
            self.router_socket.close()
        
        if self.dealer_socket:
            self.dealer_socket.close()
        
        if self.context:
            self.context.term()
        
        logger.info("ZMQ handler closed")
