"""Node for sending acknowledgments via ZMQ ROUTER."""

import logging

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_handler import ZMQHandler

logger = logging.getLogger(__name__)


@register_node
class SendAcknowledgmentNode(BaseNode):
    """Sends an acknowledgment back to the requester via the ROUTER socket."""

    def __init__(self, zmq_handler: ZMQHandler) -> None:
        super().__init__()
        self._zmq_handler = zmq_handler

    def get_description(self) -> str:
        return "Send an ACK message back to the original requester via the ROUTER socket."

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Send acknowledgment message via ROUTER socket.

        Args:
            broker: Knowledge broker for reading identity and message data

        Returns:
            NodeResult with SUCCESS if ACK sent, FAILED otherwise
        """
        try:
            identity = broker.zmq_identity
            status = broker.ack_status or "success"
            message = broker.ack_message or "Request processed"

            if identity is None:
                logger.warning("No zmq_identity found in broker, cannot send ACK")
                return NodeResult(
                    status=NodeStatus.FAILED, error="Missing zmq_identity in broker"
                )

            self._zmq_handler.send_acknowledgment(identity, status, message)

            logger.debug("Sent acknowledgment: %s", status)

            return NodeResult(
                status=NodeStatus.SUCCESS,
                metadata={"ack_status": status, "ack_sent": True},
            )

        except Exception as e:
            logger.error("Error sending acknowledgment: %s", e, exc_info=True)
            return NodeResult(status=NodeStatus.FAILED, error=str(e))
