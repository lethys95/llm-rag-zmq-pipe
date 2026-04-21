"""Node for forwarding responses via ZMQ DEALER."""

import logging

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_handler import ZMQHandler

logger = logging.getLogger(__name__)


@register_node
class ForwardResponseNode(BaseNode):
    """Forwards the final response to the downstream pipeline via the DEALER socket."""

    def __init__(self, zmq_handler: ZMQHandler) -> None:
        super().__init__()
        self._zmq_handler = zmq_handler

    def get_description(self) -> str:
        return "Send the primary response to the downstream pipeline (TTS) via the DEALER socket."

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Forward response via DEALER socket.

        Args:
            broker: Knowledge broker for reading response

        Returns:
            NodeResult with SUCCESS if forwarded, FAILED otherwise
        """
        try:
            response = broker.primary_response

            if response is None:
                logger.warning("No primary_response found in broker, cannot forward")
                return NodeResult(
                    status=NodeStatus.FAILED, error="Missing primary_response in broker"
                )

            sent_immediate = self._zmq_handler.forward_response(response)

            return NodeResult(
                status=NodeStatus.SUCCESS,
                metadata={"response_forwarded": True, "sent_immediate": sent_immediate},
            )

        except Exception as e:
            logger.error("Error forwarding response: %s", e, exc_info=True)
            return NodeResult(status=NodeStatus.FAILED, error=str(e))
