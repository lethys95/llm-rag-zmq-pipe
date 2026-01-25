"""Node for checking feedback from downstream pipeline."""

import logging
import zmq

from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_connection_manager import ZMQConnectionManager

logger = logging.getLogger(__name__)


class CheckFeedbackNode(BaseNode):
    """Node that checks for feedback from downstream pipeline.
    
    This node performs a non-blocking poll of the DEALER socket to check
    if the downstream pipeline has sent any error messages or feedback.
    This is typically used to detect incompatibility issues.
    
    Stored in broker (if feedback received):
        - 'downstream_feedback': Feedback message from downstream
    """
    
    def __init__(
        self,
        name: str = "check_feedback",
        priority: int = 15,
        timeout_ms: int = 0,
        **kwargs
    ):
        """Initialize the check feedback node.
        
        Args:
            name: Node name (default: "check_feedback")
            priority: Execution priority (default: 15, after forwarding)
            timeout_ms: Polling timeout in milliseconds (default: 0, non-blocking)
            **kwargs: Additional BaseNode arguments
        """
        super().__init__(name=name, priority=priority, queue_type="background", **kwargs)
        self.timeout_ms = timeout_ms
        self.zmq_manager = ZMQConnectionManager.get_instance()
    
    async def execute(self, broker) -> NodeResult:
        """Check for feedback from downstream via DEALER socket.
        
        Args:
            broker: Knowledge broker for storing feedback if received
            
        Returns:
            NodeResult with SUCCESS if feedback received,
            SKIPPED if no feedback,
            FAILED on error
        """
        try:
            poller = self.zmq_manager.get_poller()
            dealer_socket = self.zmq_manager.get_dealer_socket()
            
            socks = dict(poller.poll(self.timeout_ms))
            
            if dealer_socket not in socks or socks[dealer_socket] != zmq.POLLIN:
                return NodeResult(
                    status=NodeStatus.SKIPPED,
                    metadata={"reason": "no_feedback_available"}
                )
            
            message_bytes = dealer_socket.recv(zmq.NOBLOCK)
            message = message_bytes.decode('utf-8')
            
            logger.warning(f"Received feedback from downstream: {message}")
            
            data = {'downstream_feedback': message}
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                data=data,
                metadata={'feedback_received': True}
            )
        
        except zmq.Again:
            return NodeResult(
                status=NodeStatus.SKIPPED,
                metadata={"reason": "no_message_available"}
            )
        
        except Exception as e:
            logger.error(f"Error checking downstream feedback: {e}", exc_info=True)
            return NodeResult(
                status=NodeStatus.FAILED,
                error=str(e)
            )
