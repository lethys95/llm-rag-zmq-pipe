"""ZMQ socket handler for ROUTER and DEALER patterns."""

import json
import logging
from collections import deque
from enum import StrEnum
from typing import cast

import msgpack  # type: ignore[import-untyped]
import zmq
from pydantic import ValidationError
from zmq import Poller

from src.config.settings import Settings
from src.models import DialogueInput

logger = logging.getLogger(__name__)


class MessageTopic(StrEnum):
    """Topics for routing multipart messages."""

    STT = "stt"
    DIALOGUE = "dialogue"


class ZMQHandler:
    """Handler for ZMQ ROUTER and DEALER sockets.

    Manages the communication layer:
    - ROUTER socket: Receives prompts from STT clients, sends ACKs back
    - DEALER socket: Forwards responses to downstream pipeline, can receive error feedback

    Architecture:
    STT → [ROUTER] → This Service → [DEALER] → Downstream
         ← [ROUTER] ← (ACKs)        ← [DEALER] ← (Error feedback)
    """

    def __init__(self, settings: Settings, max_queue_size: int = 1000):
        """Initialize ZMQ handler.

        Args:
            settings: Application settings
            max_queue_size: Maximum size for outgoing message queue
        """
        self.max_queue_size = max_queue_size
        self.outgoing_queue: deque[bytes] = deque(maxlen=max_queue_size)

        self.context = zmq.Context()

        # Create ROUTER socket for receiving requests from STT
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(settings.zmq_input_endpoint)
        logger.info("ROUTER socket bound to %s", settings.zmq_input_endpoint)

        # Create DEALER socket for bidirectional communication with downstream
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.dealer_socket.connect(settings.zmq_output_endpoint)
        logger.info("DEALER socket connected to %s", settings.zmq_output_endpoint)

        self.poller = self._setup_poller()

        logger.info(
            "ZMQ Handler initialized: input=%s, output=%s",
            settings.zmq_input_endpoint,
            settings.zmq_output_endpoint,
        )

    def _setup_poller(self) -> Poller:
        # Set up poller for non-blocking receive on both sockets
        poller = zmq.Poller()
        poller.register(self.router_socket, zmq.POLLIN)
        poller.register(self.dealer_socket, zmq.POLLIN)
        return poller

    def get_poller(self) -> zmq.Poller:
        """Get the ZMQ poller for non-blocking operations.

        Returns:
            ZMQ Poller
        """
        return self.poller

    def get_router_socket(self) -> zmq.Socket:  # type: ignore[no-untyped-call]
        """Get the ROUTER socket for receiving messages.

        Returns:
            ZMQ ROUTER socket
        """
        return self.router_socket

    def get_dealer_socket(self) -> zmq.Socket:  # type: ignore[no-untyped-call]
        """Get the DEALER socket for sending messages.

        Returns:
            ZMQ DEALER socket
        """
        return self.dealer_socket

    def queue_outgoing(self, message: bytes) -> bool:
        """Queue an outgoing message for transmission via DEALER.

        If the queue is full, the oldest message is dropped.

        Args:
            message: Message bytes to send

        Returns:
            True if queued successfully, False if queue was full
        """
        if len(self.outgoing_queue) >= self.max_queue_size:
            logger.warning(
                "Outgoing queue full, dropping oldest message (queue_size=%s)",
                self.max_queue_size,
            )
            self.outgoing_queue.popleft()

        self.outgoing_queue.append(message)
        logger.debug(
            "Queued outgoing message (queue_size=%s)", len(self.outgoing_queue)
        )
        return True

    def flush_queue(self, max_messages: int | None = None) -> int:
        """Flush queued messages to DEALER socket (non-blocking).

        Args:
            max_messages: Maximum number of messages to send (None = all)

        Returns:
            Number of messages successfully sent
        """
        if not self.outgoing_queue:
            return 0

        sent_count = 0
        messages_to_send = max_messages or len(self.outgoing_queue)

        for _ in range(min(messages_to_send, len(self.outgoing_queue))):
            message = None
            try:
                message = self.outgoing_queue.popleft()
                self.dealer_socket.send(message, zmq.NOBLOCK)
                sent_count += 1
                logger.debug("Flushed message from queue (%s sent)", sent_count)

            except zmq.Again:
                if message:
                    self.outgoing_queue.appendleft(message)
                    logger.debug("DEALER socket busy, message re-queued")
                else:
                    logger.error("DEALER socket busy, and message was None which is weird.")
                break

            except Exception as e:
                logger.error("Error flushing message from queue: %s", e, exc_info=True)

        return sent_count

    def send_immediate(self, message: bytes) -> bool:
        """Send message immediately via DEALER (non-blocking).

        If send fails, message is queued automatically.

        Args:
            message: Message bytes to send

        Returns:
            True if sent immediately, False if queued
        """
        try:
            self.dealer_socket.send(message, zmq.NOBLOCK)
            logger.debug("Message sent immediately via DEALER")
            return True

        except zmq.Again:
            logger.debug("DEALER socket busy, queueing message")
            self.queue_outgoing(message)
            return False

        except Exception as e:
            logger.error("Error sending message via DEALER: %s", e, exc_info=True)
            return False

    def receive_request(
        self, timeout: int = 1000
    ) -> tuple[list[bytes] | None, DialogueInput | None]:
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

        extracted = self._extract_frames(frames)
        if extracted is None:
            return None, None

        identity, topic, message_bytes = extracted
        dialogue_input = self._parse_message(topic, message_bytes)

        return identity, dialogue_input

    def _poll_router_socket(self, timeout: int) -> list[bytes] | None:
        """Poll ROUTER socket for incoming messages."""
        socks = dict(self.poller.poll(timeout))

        if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
            return self.router_socket.recv_multipart()

        return None

    def _extract_frames(
        self, frames: list[bytes]
    ) -> tuple[list[bytes], MessageTopic, bytes] | None:
        """Extract identity, topic, and message from ROUTER multipart frames.

        ROUTER frames: [identity_part1, ..., topic, message]
        First frame after identity is the topic, rest is the message.

        Returns None if the frame structure is invalid or the topic is unknown.
        """
        if len(frames) < 2:
            logger.warning("Received malformed ZMQ message: expected at least 2 frames, got %d", len(frames))
            return None

        try:
            topic_str = frames[-2].decode("utf-8")
            topic = MessageTopic(topic_str)
        except (UnicodeDecodeError, ValueError):
            logger.warning("Received unknown or undecodable ZMQ topic: %r", frames[-2])
            return None

        identity = frames[:-2]
        message = frames[-1]
        return identity, topic, message

    def _parse_message(
        self, topic: MessageTopic, message_bytes: bytes
    ) -> DialogueInput | None:
        """Parse message bytes into DialogueInput based on topic.

        Args:
            topic: The message topic determining how to parse
            message_bytes: Raw message bytes

        Returns:
            DialogueInput or None if parsing fails
        """
        message_data = self._deserialize_message(message_bytes)
        if not message_data:
            return None

        match topic:
            case MessageTopic.STT:
                return self._handle_stt_response(message_data)
            case MessageTopic.DIALOGUE:
                return self._create_dialogue_input(message_data)
            case _:
                logger.warning("Unknown message topic: %s", topic)
                return None

    def _deserialize_message(self, message_bytes: bytes) -> dict[str, object] | None:
        """Deserialize message bytes trying msgpack first, then JSON."""
        try:
            result = msgpack.unpackb(  # type: ignore[no-untyped-call]
                message_bytes, raw=False
            )
            return cast(dict[str, object], result)
        except (msgpack.exceptions.UnpackException, TypeError):
            pass

        try:
            message_str = message_bytes.decode("utf-8")
            return json.loads(message_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to deserialize message: %s", e)
            return None

    def _handle_stt_response(self, data: dict[str, object]) -> DialogueInput | None:
        """Convert STT TranscriptionResponse to DialogueInput."""
        if data.get("status") != "success":
            error_details = data.get("error_details", "Unknown error")
            logger.error("STT transcription failed: %s", error_details)
            return None

        text = str(data.get("text", ""))
        speaker = str(data.get("speaker", "User"))
        system_prompt = data.get("system_prompt")
        system_prompt_override = (
            str(system_prompt) if system_prompt is not None else None
        )
        voice_id = data.get("voice_id")

        dialogue_input = DialogueInput(
            content=text,
            speaker=speaker,
            system_prompt_override=system_prompt_override,
            voice_id=str(voice_id) if voice_id is not None else None,
        )
        logger.debug("Received STT from '%s': %.100s...", speaker, text)
        return dialogue_input

    def _create_dialogue_input(self, data: dict[str, object]) -> DialogueInput | None:
        """Create DialogueInput from dictionary data."""
        try:
            dialogue_input = DialogueInput.model_validate(data)
            logger.debug(
                "Received message from '%s': %.100s...",
                dialogue_input.speaker,
                dialogue_input.content,
            )
            return dialogue_input
        except ValidationError as e:
            logger.error("Invalid DialogueInput data: %s", e)
            return None

    def send_acknowledgment(
        self, identity: list[bytes], status: str, message: str
    ) -> None:
        """Send acknowledgment back to the requester via ROUTER.

        Args:
            identity: Identity frames from the original request
            status: Status string ("success" or "error")
            message: Acknowledgment message
        """
        ack_message = f"[{status.upper()}] {message}"
        ack_bytes = ack_message.encode("utf-8")

        # Send back through ROUTER: identity frames + message
        self.router_socket.send_multipart(  # type: ignore[no-untyped-call]
            identity + [ack_bytes]
        )

        logger.debug("Sent acknowledgment: %s", status)

    def forward_response(self, response: str, voice_id: str | None = None) -> bool:
        """Forward the LLM response to downstream pipeline via DEALER.

        If the DEALER socket is busy, the message is queued and flushed
        immediately after — ensuring queued messages are not abandoned.

        Args:
            response: The generated response to forward
            voice_id: Optional TTS voice ID passed through from the originating client

        Returns:
            True if sent immediately, False if queued then flushed
        """
        payload: dict = {
            "type": "synthesize",
            "text": response,
        }
        if voice_id is not None:
            payload["voice_id"] = voice_id

        response_bytes = msgpack.packb(payload)
        sent_immediate = self.send_immediate(response_bytes)

        if not sent_immediate:
            logger.debug("Response queued for later transmission")
            self.flush_queue()

        logger.debug("Forwarded response via DEALER: %.100s...", response)
        return sent_immediate

    def check_downstream_feedback(self, timeout: int = 0) -> str | None:
        """Drain TTS streaming response frames from the DEALER socket.

        TTS sends back metadata → audio chunks → complete. We drain all available
        frames to prevent the socket buffer filling up, and surface any error frames.

        Args:
            timeout: Timeout in milliseconds for initial poll (0 = non-blocking)

        Returns:
            Error message if TTS reported an error, otherwise None
        """
        try:
            socks = dict(self.poller.poll(timeout))
            if self.dealer_socket not in socks or socks[self.dealer_socket] != zmq.POLLIN:
                return None

            while True:
                try:
                    frames = self.dealer_socket.recv_multipart(zmq.NOBLOCK)
                    msg_type = frames[0] if frames else b""

                    if msg_type == b"error":
                        data = frames[1] if len(frames) > 1 else b""
                        try:
                            error_msg = msgpack.unpackb(data, raw=False).get("error", "unknown TTS error")
                        except Exception:
                            error_msg = data.decode("utf-8", errors="replace")
                        logger.warning("TTS error response: %s", error_msg)
                        return error_msg

                    elif msg_type == b"complete":
                        logger.debug("TTS synthesis complete")
                        return None

                    # metadata and audio frames: drain silently

                except zmq.Again:
                    break

        except Exception as e:
            logger.error("Error draining downstream response: %s", e, exc_info=True)

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

    def __repr__(self) -> str:
        """String representation of the handler."""
        return (
            f"<ZMQHandler "
            f"queue_size={len(self.outgoing_queue)} "
            f"max_queue={self.max_queue_size}>"
        )
