"""Node for forwarding responses via ZMQ DEALER."""

import logging

from src.nodes.core.base import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus
from src.communication.zmq_connection_manager import ZMQConnectionManager

logger = logging.getLogger(__name__)


class ForwardResponseNode(BaseNode):
    """Node that forwards the final response to downstream pipeline.
    
    This node reads the final response from the KnowledgeBroker and
    sends it via the DEALER socket to the downstream pipeline. It also
    prints the response to stdout.
    
    Expected in broker:
        - primary_response: The final response text to forward
    """
    
    def __init__(
        self,
        name: str = "forward_response",
        priority: int = 11,
        **kwargs
    ):
        """Initialize the forward response node.
        
        Args:
            name: Node name (default: "forward_response")
            priority: Execution priority (default: 11, after ACK)
            **kwargs: Additional BaseNode arguments
        """
        super().__init__(name=name, priority=priority, **kwargs)
        self.zmq_manager = ZMQConnectionManager.get_instance()
    
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
                    status=NodeStatus.FAILED,
                    error="Missing primary_response in broker"
                )
            
            print(response)
            
            response_bytes = response.encode('utf-8')
            sent_immediate = self.zmq_manager.send_immediate(response_bytes)
            
            if not sent_immediate:
                logger.debug("Response queued for later transmission")
            
            logger.debug(f"Forwarded response via DEALER: {response[:100]}...")
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                metadata={'response_forwarded': True, 'sent_immediate': sent_immediate}
            )
        
        except Exception as e:
            logger.error(f"Error forwarding response: {e}", exc_info=True)
            return NodeResult(
                status=NodeStatus.FAILED,
                error=str(e)
            )
