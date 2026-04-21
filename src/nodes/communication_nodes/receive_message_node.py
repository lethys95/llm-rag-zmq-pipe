"""Node for receiving and parsing ZMQ messages."""

import logging
import json
import zmq
import msgpack
from pydantic import ValidationError

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_handler import ZMQHandler
from src.models import DialogueInput

logger = logging.getLogger(__name__)


@register_node
class ReceiveMessageNode(BaseNode):
    """Receives and parses incoming ZMQ messages into the broker."""

    _timeout_ms = 30_000

    def __init__(self, zmq_handler: ZMQHandler) -> None:
        super().__init__()
        self._zmq_handler = zmq_handler

    def get_description(self) -> str:
        return "Poll the ROUTER socket for an incoming message and populate the broker with DialogueInput and ZMQ identity."

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Receive and parse a message from the ROUTER socket.

        Args:
            broker: Knowledge broker for storing parsed data

        Returns:
            NodeResult with SUCCESS if message received and parsed,
            SKIPPED if no message within timeout,
            FAILED if parsing error
        """
        try:
            poller = self._zmq_handler.get_poller()
            router_socket = self._zmq_handler.get_router_socket()

            socks = dict(poller.poll(self._timeout_ms))

            if router_socket not in socks or socks[router_socket] != zmq.POLLIN:
                return NodeResult(
                    status=NodeStatus.SKIPPED,
                    metadata={"reason": "no_message_within_timeout"},
                )

            frames = router_socket.recv_multipart()
            identity, message_bytes = self._extract_frames(frames)

            dialogue_input = self._parse_message(message_bytes)

            if dialogue_input is None:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Failed to parse message into DialogueInput",
                )

            data = {
                "zmq_identity": identity,
                "message_content": dialogue_input.content,
                "message_speaker": dialogue_input.speaker,
                "system_prompt_override": dialogue_input.system_prompt_override,
                "dialogue_input": dialogue_input,
            }

            logger.info(
                "Received message from '%s': %.100s...",
                dialogue_input.speaker,
                dialogue_input.content,
            )

            return NodeResult(status=NodeStatus.SUCCESS, data=data)

        except Exception as e:
            logger.error("Error receiving message: %s", e, exc_info=True)
            return NodeResult(status=NodeStatus.FAILED, error=str(e))

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
            message_str = message_bytes.decode("utf-8")
            return json.loads(message_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to deserialize message: %s", e)
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
        return isinstance(data, dict) and "text" in data and "status" in data

    def _handle_stt_response(self, data: dict) -> DialogueInput | None:
        """Convert STT TranscriptionResponse to DialogueInput.

        Args:
            data: STT response dictionary

        Returns:
            DialogueInput object or None if status indicates failure
        """
        if data.get("status") != "success":
            error_details = data.get("error_details", "Unknown error")
            logger.error("STT transcription failed: %s", error_details)
            return None

        text = data.get("text", "")
        speaker = data.get("speaker", "User")
        system_prompt_override = data.get("system_prompt_override")

        dialogue_input = DialogueInput(
            content=text, speaker=speaker, system_prompt_override=system_prompt_override
        )
        logger.debug("Converted STT response from '%s': %.100s...", speaker, text)
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
                "Created DialogueInput from '%s': %.100s...",
                dialogue_input.speaker,
                dialogue_input.content,
            )
            return dialogue_input
        except ValidationError as e:
            logger.error("Invalid DialogueInput data: %s", e)
            return None
