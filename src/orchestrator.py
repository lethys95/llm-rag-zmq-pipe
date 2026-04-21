"""Node-based orchestrator for LLM RAG response system."""

from dataclasses import dataclass
import logging

from src.communication.zmq_handler import ZMQHandler
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class Orchestrator:
    """Node-based orchestrator for LLM RAG response system.

    This orchestrator replaces the old PipelineServer with a flexible node-based
    architecture that dynamically selects and executes processing nodes based on
    incoming message and context.

    Architecture:
        1. Receive message via ZMQ
        2. Use DecisionEngine to select nodes
        3. Execute nodes via TaskQueueManager
        4. Send ACK and forward response
        5. Background nodes continue async
    """

    settings: Settings
    _zmq: ZMQHandler = ZMQHandler()
    _running: bool = False

    def run(self) -> None:
        """Run the orchestrator main loop.

        Listens for incoming ZMQ messages, processes them, and sends responses.
        This method blocks until stop() is called.
        """
        self._running = True
        logger.info("Orchestrator started, listening for messages...")
        self._run_main_loop()

        logger.info("Orchestrator stopped")

    def _run_main_loop(self) -> None:
        while self._running:
            try:
                identity, dialogue_input = self._zmq.receive_request(timeout=1000)

                if identity is None or dialogue_input is None:
                    continue

                logger.info(
                    "Received message from '%s': %.100s...",
                    dialogue_input.speaker,
                    dialogue_input.content,
                )

                response = f"Echo: {dialogue_input.content}"

                self._zmq.send_acknowledgment(
                    identity, "success", "Message received and processed"
                )

                self._zmq.forward_response(response)

                logger.debug("Message processed and response forwarded")

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self._running = False

    def stop(self) -> None:
        """Signal the orchestrator to stop running."""
        self._running = False
        logger.info("Stop signal received")

    def close(self) -> None:
        """Close the orchestrator and cleanup resources."""
        self.stop()
        self._zmq.close()
        logger.info("Orchestrator closed")
