"""Node for receiving and parsing ZMQ messages."""

import logging
import json
import zmq
import msgpack
from pydantic import ValidationError

from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_connection_manager import ZMQConnectionManager
from src.models import DialogueInput

logger = logging.getLogger(__name__)


class ReceiveMessageNode(BaseNode):
    """Node that receives and parses incoming ZMQ messages.
    
    This node polls the ROUTER socket for incoming messages, parses them
    (supporting both msgpack and JSON formats), extracts DialogueInput,
    and stores the identity frames and parsed data in the KnowledgeBroker.
    
    Stored in broker:
        - 'zmq_identity': Identity frames for routing replies
        - 'message_content': User's message content
        - 'message_speaker': Speaker identifier
        - 'system_prompt_override': Optional system prompt override
        - 'dialogue_input': Complete DialogueInput object
    """
    
    def __init__(
        self,
        name: str = "receive_message",
        timeout_ms: int = 1000,
        **kwargs
    ):
        """Initialize the receive message node.
        
        Args:
            name: Node name (default: "receive_message")
            timeout_ms: Polling timeout in milliseconds (default: 1000)
            **kwargs: Additional BaseNode arguments
        """
        super().__init__(name=name, priority=0, **kwargs)
        self.timeout_ms = timeout_ms
        self.zmq_manager = ZMQConnectionManager.get_instance()
    
    async def execute(self, broker) -> NodeResult:
        """Receive and parse a message from the ROUTER socket.
        
        Args:
            broker: Knowledge broker for storing parsed data
            
        Returns:
            NodeResult with SUCCESS if message received and parsed,
            SKIPPED if no message within timeout,
            FAILED if parsing error
        """
        try:
            poller = self.zmq_manager.get_poller()
            router_socket = self.zmq_manager.get_router_socket()
            
            socks = dict(poller.poll(self.timeout_ms))
            
            if router_socket not in socks or socks[router_socket] != zmq.POLLIN:
                return NodeResult(
                    status=NodeStatus.SKIPPED,
                    metadata={"reason": "no_message_within_timeout"}
                )
            
            frames = router_socket.recv_multipart()
            identity, message_bytes = self._extract_frames(frames)
            
            dialogue_input = self._parse_message(message_bytes)
            
            if dialogue_input is None:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Failed to parse message into DialogueInput"
                )
            
            data = {
                'zmq_identity': identity,
                'message_content': dialogue_input.content,
                'message_speaker': dialogue_input.speaker,
                'system_prompt_override': dialogue_input.system_prompt_override,
                'dialogue_input': dialogue_input,
            }
            
            logger.info(
                f"Received message from '{dialogue_input.speaker}': "
                f"{dialogue_input.content[:100]}..."
            )
            
            return NodeResult(status=NodeStatus.SUCCESS, data=data)
        
        except Exception as e:
            logger.error(f"Error receiving message: {e}", exc_info=True)
            return NodeResult(
                status=NodeStatus.FAILED,
                error=str(e)
            )
    
    def _extract_frames(self, frames: list[bytes]) -> tuple[list[bytes], bytes]:
        """Extract identity and message from ROUTER multipart frames.
        
        ROUTER frames: [identity_part1, identity_part2, ..., empty_delimiter, message]
        
        Args:
            frames: Multipart message frames
            
        Returns:
            Tuple of (identity_frames, message_bytes)
        """
        return frames[:-1], frames[-1]
    
    def _parse_message(self, message_bytes: bytes) -> DialogueInput | None:
        """Parse message bytes into DialogueInput.
        
        Attempts deserialization in order: msgpack → JSON
        Handles both STT TranscriptionResponse and direct DialogueInput formats.
        
        Args:
            message_bytes: Raw message bytes
            
        Returns:
            DialogueInput object or None if parsing failed
        """
        message_data = self._deserialize_message(message_bytes)
        if not message_data:
            return None
        
        return self._extract_dialogue_input(message_data)
    
    def _deserialize_message(self, message_bytes: bytes) -> dict | None:
        """Deserialize message bytes trying msgpack first, then JSON.
        
        Args:
            message_bytes: Raw message bytes
            
        Returns:
            Deserialized dictionary or None if both fail
        """
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
        
        Args:
            message_data: Deserialized message dictionary
            
        Returns:
            DialogueInput object or None if extraction failed
        """
        if self._is_stt_response(message_data):
            return self._handle_stt_response(message_data)
        
        return self._create_dialogue_input(message_data)
    
    def _is_stt_response(self, data: dict) -> bool:
        """Check if message is from STT service based on schema.
        
        Args:
            data: Message dictionary
            
        Returns:
            True if appears to be STT response format
        """
        return isinstance(data, dict) and 'text' in data and 'status' in data
    
    def _handle_stt_response(self, data: dict) -> DialogueInput | None:
        """Convert STT TranscriptionResponse to DialogueInput.
        
        Args:
            data: STT response dictionary
            
        Returns:
            DialogueInput object or None if status indicates failure
        """
        if data.get('status') != 'success':
            error_details = data.get('error_details', 'Unknown error')
            logger.error(f"STT transcription failed: {error_details}")
            return None
        
        text = data.get('text', '')
        speaker = data.get('speaker', 'User')
        
        dialogue_input = DialogueInput(content=text, speaker=speaker)
        logger.debug(f"Converted STT response from '{speaker}': {text[:100]}...")
        return dialogue_input
    
    def _create_dialogue_input(self, data: dict) -> DialogueInput | None:
        """Create DialogueInput from dictionary data.
        
        Args:
            data: DialogueInput dictionary
            
        Returns:
            DialogueInput object or None if validation failed
        """
        try:
            dialogue_input = DialogueInput(**data)
            logger.debug(
                f"Created DialogueInput from '{dialogue_input.speaker}': "
                f"{dialogue_input.content[:100]}..."
            )
            return dialogue_input
        except ValidationError as e:
            logger.error(f"Invalid DialogueInput data: {e}")
            return None
