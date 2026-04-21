"""Node for checking feedback from downstream pipeline."""

import logging

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_handler import ZMQHandler

logger = logging.getLogger(__name__)


@register_node
class CheckFeedbackNode(BaseNode):
    """Node that checks for feedback from downstream pipeline.

    This node performs a non-blocking poll of the DEALER socket to check
    if the downstream pipeline has sent any error messages or feedback.
    This is typically used to detect incompatibility issues.

    Stored in broker (if feedback received):
        - 'downstream_feedback': Feedback message from downstream
    """

    _zmq_handler = ZMQHandler()

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Check for feedback from downstream via DEALER socket.

        Args:
            broker: Knowledge broker for storing feedback if received

        Returns:
            NodeResult with SUCCESS if feedback received,
            SKIPPED if no feedback,
            FAILED on error
        """
        try:
            message = self._zmq_handler.check_downstream_feedback()

            if message is None:
                return NodeResult(
                    status=NodeStatus.SKIPPED,
                    metadata={"reason": "no_feedback_available"},
                )

            data = {"downstream_feedback": message}

            return NodeResult(
                status=NodeStatus.SUCCESS,
                data=data,
                metadata={"feedback_received": True},
            )

        except Exception as e:
            logger.error("Error checking downstream feedback: %s", e, exc_info=True)
            return NodeResult(status=NodeStatus.FAILED, error=str(e))
