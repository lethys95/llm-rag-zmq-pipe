"""ZMQ connection manager for persistent ROUTER/DEALER sockets."""

import logging
import zmq
from collections import deque

logger = logging.getLogger(__name__)


class ZMQConnectionManager:
    """Manages persistent ZMQ ROUTER and DEALER socket connections.
    
    This singleton class maintains long-lived socket connections and provides
    an internal queue for buffering outgoing messages when the DEALER socket
    is busy. It handles non-blocking message transmission with graceful queuing.
    
    Attributes:
        input_endpoint: ZMQ ROUTER bind endpoint (e.g., "tcp://*:5555")
        output_endpoint: ZMQ DEALER connect endpoint (e.g., "tcp://localhost:5556")
        max_queue_size: Maximum size for outgoing message queue
        context: ZMQ context
        router_socket: ROUTER socket for receiving messages
        dealer_socket: DEALER socket for sending messages
        outgoing_queue: Queue for buffering outgoing messages
        poller: ZMQ poller for non-blocking operations
    """
    
    _instance: "ZMQConnectionManager | None" = None
    
    def __init__(
        self,
        input_endpoint: str,
        output_endpoint: str,
        max_queue_size: int = 1000
    ):
        """Initialize the ZMQ connection manager.
        
        Args:
            input_endpoint: Endpoint to bind ROUTER socket
            output_endpoint: Endpoint to connect DEALER socket
            max_queue_size: Maximum size for outgoing message queue
        """
        self.input_endpoint = input_endpoint
        self.output_endpoint = output_endpoint
        self.max_queue_size = max_queue_size
        
        self.context: zmq.Context | None = None
        self.router_socket: zmq.Socket | None = None
        self.dealer_socket: zmq.Socket | None = None
        self.outgoing_queue: deque = deque(maxlen=max_queue_size)
        self.poller: zmq.Poller | None = None
        
        logger.info(
            f"ZMQ Connection Manager initialized: "
            f"input={input_endpoint}, output={output_endpoint}"
        )
    
    @classmethod
    def get_instance(
        cls,
        input_endpoint: str | None = None,
        output_endpoint: str | None = None,
        max_queue_size: int = 1000
    ) -> "ZMQConnectionManager":
        """Get singleton instance of the connection manager.
        
        Args:
            input_endpoint: Required on first call
            output_endpoint: Required on first call
            max_queue_size: Maximum queue size (default: 1000)
            
        Returns:
            ZMQConnectionManager singleton instance
        """
        if cls._instance is None:
            if input_endpoint is None or output_endpoint is None:
                raise ValueError(
                    "input_endpoint and output_endpoint required for first instantiation"
                )
            cls._instance = cls(input_endpoint, output_endpoint, max_queue_size)
        return cls._instance
    
    def setup(self) -> None:
        """Set up ZMQ sockets and poller."""
        self.context = zmq.Context()
        
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(self.input_endpoint)
        logger.info(f"ROUTER socket bound to {self.input_endpoint}")
        
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.dealer_socket.connect(self.output_endpoint)
        logger.info(f"DEALER socket connected to {self.output_endpoint}")
        
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        self.poller.register(self.dealer_socket, zmq.POLLIN)
        
        logger.info("ZMQ sockets and poller set up successfully")
    
    def get_router_socket(self) -> zmq.Socket:
        """Get the ROUTER socket for receiving messages.
        
        Returns:
            ZMQ ROUTER socket
            
        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if self.router_socket is None:
            raise RuntimeError("Router socket not initialized. Call setup() first.")
        return self.router_socket
    
    def get_dealer_socket(self) -> zmq.Socket:
        """Get the DEALER socket for sending messages.
        
        Returns:
            ZMQ DEALER socket
            
        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if self.dealer_socket is None:
            raise RuntimeError("Dealer socket not initialized. Call setup() first.")
        return self.dealer_socket
    
    def get_poller(self) -> zmq.Poller:
        """Get the ZMQ poller for non-blocking operations.
        
        Returns:
            ZMQ Poller
            
        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if self.poller is None:
            raise RuntimeError("Poller not initialized. Call setup() first.")
        return self.poller
    
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
                "Outgoing queue full, dropping oldest message "
                f"(queue_size={self.max_queue_size})"
            )
            self.outgoing_queue.popleft()
        
        self.outgoing_queue.append(message)
        logger.debug(f"Queued outgoing message (queue_size={len(self.outgoing_queue)})")
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
            try:
                message = self.outgoing_queue.popleft()
                self.dealer_socket.send(message, zmq.NOBLOCK)
                sent_count += 1
                logger.debug(f"Flushed message from queue ({sent_count} sent)")
            
            except zmq.Again:
                self.outgoing_queue.appendleft(message)
                logger.debug("DEALER socket busy, message re-queued")
                break
            
            except Exception as e:
                logger.error(f"Error flushing message from queue: {e}", exc_info=True)
        
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
            logger.error(f"Error sending message via DEALER: {e}", exc_info=True)
            return False
    
    def close(self) -> None:
        """Close ZMQ sockets and terminate context."""
        logger.info("Closing ZMQ connection manager")
        
        if self.router_socket:
            self.router_socket.close()
        
        if self.dealer_socket:
            self.dealer_socket.close()
        
        if self.context:
            self.context.term()
        
        logger.info("ZMQ connection manager closed")
    
    def __repr__(self) -> str:
        """String representation of the connection manager."""
        return (
            f"<ZMQConnectionManager "
            f"queue_size={len(self.outgoing_queue)} "
            f"max_queue={self.max_queue_size}>"
        )
