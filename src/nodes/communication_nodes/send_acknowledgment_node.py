"""Node for sending acknowledgments via ZMQ ROUTER."""

import logging

from src.nodes.core.base import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_connection_manager import ZMQConnectionManager

logger = logging.getLogger(__name__)


class SendAcknowledgmentNode(BaseNode):
    """Node that sends acknowledgment messages back to requesters.
    
    This node reads the identity frames and acknowledgment data from the
    KnowledgeBroker and sends a formatted ACK message back to the original
    requester via the ROUTER socket.
    
    Expected in broker:
        - zmq_identity: Identity frames for routing
        - ack_status: Status string ("success" or "error")
        - ack_message: Acknowledgment message content
    """
    
    def __init__(
        self,
        name: str = "send_acknowledgment",
        priority: int = 10,
        **kwargs
    ):
        """Initialize the send acknowledgment node.
        
        Args:
            name: Node name (default: "send_acknowledgment")
            priority: Execution priority (default: 10, low priority)
            **kwargs: Additional BaseNode arguments
        """
        super().__init__(name=name, priority=priority, **kwargs)
        self.zmq_manager = ZMQConnectionManager.get_instance()
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Send acknowledgment message via ROUTER socket.
        
        Args:
            broker: Knowledge broker for reading identity and message data
            
        Returns:
            NodeResult with SUCCESS if ACK sent, FAILED otherwise
        """
        try:
            identity = broker.zmq_identity
            status = broker.ack_status or 'success'
            message = broker.ack_message or 'Request processed'
            
            if identity is None:
                logger.warning("No zmq_identity found in broker, cannot send ACK")
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error="Missing zmq_identity in broker"
                )
            
            ack_message = f"[{status.upper()}] {message}"
            ack_bytes = ack_message.encode('utf-8')
            
            router_socket = self.zmq_manager.get_router_socket()
            router_socket.send_multipart(identity + [ack_bytes])
            
            logger.debug(f"Sent acknowledgment: {status}")
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                metadata={'ack_status': status, 'ack_sent': True}
            )
        
        except Exception as e:
            logger.error(f"Error sending acknowledgment: {e}", exc_info=True)
            return NodeResult(
                status=NodeStatus.FAILED,
                error=str(e)
            )
