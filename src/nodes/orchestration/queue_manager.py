"""Task queue manager for priority-based node execution."""

import asyncio
import logging
import time
from dataclasses import dataclass

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker


@dataclass
class QueueStatus:
    immediate_queue_size: int
    background_queue_size: int
    completed_nodes: int
    failed_nodes: int
    waiting_nodes: int


logger = logging.getLogger(__name__)


class TaskQueueManager:
    """Manages execution of nodes with priorities and dependencies.

    The queue manager handles:
    - Priority-based ordering (lower priority number = executes first)
    - Dependency resolution (wait for required nodes to complete)
    - Immediate vs background execution
    - Error handling and graceful degradation

    Attributes:
        immediate_queue: Queue for user-facing nodes (blocks response)
        background_queue: Queue for async processing (doesn't block)
        completed_nodes: Set of successfully completed node names
        failed_nodes: Set of failed node names
        waiting_nodes: List of nodes waiting for dependencies
    """

    def __init__(self):
        """Initialize the task queue manager."""
        self.immediate_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.background_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_nodes: set[str] = set()
        self.failed_nodes: set[str] = set()
        self.waiting_nodes: list[BaseNode] = []

        logger.debug("Task queue manager initialized")

    async def enqueue(self, node: BaseNode) -> None:
        """Add a node to the appropriate queue.

        Args:
            node: The node instance to enqueue
        """
        queue = (
            self.immediate_queue
            if node.queue_type == "immediate"
            else self.background_queue
        )

        # PriorityQueue uses tuple (priority, item) for ordering
        # We use (priority, id) to ensure stable ordering when priorities match
        await queue.put((node.priority, id(node), node))

        logger.debug(
            f"Enqueued node '{node.name}' to {node.queue_type} queue "
            f"(priority={node.priority})"
        )

    async def execute_immediate(self, broker: KnowledgeBroker) -> None:
        """Execute all immediate nodes, blocking until complete.

        This processes nodes in priority order, respecting dependencies.
        Nodes that fail don't block other nodes from executing.

        Args:
            broker: Knowledge broker for nodes to read/write context
        """
        logger.info("Starting immediate node execution")

        while not self.immediate_queue.empty() or self.waiting_nodes:
            # Process waiting nodes first (those with unmet dependencies)
            await self._process_waiting_nodes(broker)

            # Try to get next node from queue
            if self.immediate_queue.empty():
                # All remaining nodes must be waiting on dependencies
                if self.waiting_nodes:
                    logger.warning(
                        f"{len(self.waiting_nodes)} nodes stuck waiting for "
                        f"dependencies that may never complete"
                    )
                break

            try:
                priority, node_id, node = await asyncio.wait_for(
                    self.immediate_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            # Check dependencies
            if not node.validate_dependencies(self.completed_nodes):
                # Dependencies not met, add to waiting list
                self.waiting_nodes.append(node)
                logger.debug(f"Node '{node.name}' waiting for dependencies")
                continue

            # Execute the node
            await self._execute_node(node, broker)

        logger.info(
            f"Immediate execution complete: "
            f"{len(self.completed_nodes)} completed, "
            f"{len(self.failed_nodes)} failed"
        )

    async def execute_background(self, broker: KnowledgeBroker) -> None:
        """Execute background nodes without blocking.

        This is meant to be called with asyncio.create_task() to run
        in the background while the response is being returned.

        Args:
            broker: Knowledge broker for nodes to read/write context
        """
        if self.background_queue.empty():
            logger.debug("No background nodes to execute")
            return

        logger.info("Starting background node execution")
        node_count = 0

        while not self.background_queue.empty():
            try:
                priority, node_id, node = await asyncio.wait_for(
                    self.background_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            # Background nodes execute regardless of dependencies
            # (they're meant for async processing)
            await self._execute_node(node, broker)
            node_count += 1

        logger.info(f"Background execution complete: {node_count} nodes processed")

    async def _process_waiting_nodes(self, broker: KnowledgeBroker) -> None:
        """Check waiting nodes and execute those whose dependencies are now met.

        Args:
            broker: Knowledge broker for nodes to read/write context
        """
        if not self.waiting_nodes:
            return

        still_waiting = []

        for node in self.waiting_nodes:
            if node.validate_dependencies(self.completed_nodes):
                # Dependencies now met, execute it
                await self._execute_node(node, broker)
            else:
                # Still waiting
                still_waiting.append(node)

        self.waiting_nodes = still_waiting

    async def _execute_node(self, node: BaseNode, broker: KnowledgeBroker) -> None:
        """Execute a single node and handle its result.

        Args:
            node: The node to execute
            broker: Knowledge broker for context
        """
        logger.info(f"Executing node: '{node.name}'")

        # Check if node should run (conditional logic)
        if not node.should_run(broker):
            logger.info(f"Node '{node.name}' skipped (should_run returned False)")
            broker.record_node_execution(node.name, "skipped")
            return

        start_time = time.time()

        try:
            # Execute with optional timeout
            if node.timeout:
                result = await asyncio.wait_for(
                    node.execute(broker), timeout=node.timeout
                )
            else:
                result = await node.execute(broker)

            duration = time.time() - start_time

            # Handle result
            if result.is_success():
                self.completed_nodes.add(node.name)
                broker.record_node_execution(node.name, "success", duration)

                # Data is handled by nodes directly setting broker attributes

                # Enqueue any next nodes
                for next_node in result.next_nodes:
                    await self.enqueue(next_node)

                logger.info(
                    f"Node '{node.name}' completed successfully in {duration:.3f}s"
                )

            elif result.is_failed():
                self.failed_nodes.add(node.name)
                broker.record_node_execution(node.name, "failed", duration)

                logger.error(
                    f"Node '{node.name}' failed in {duration:.3f}s: {result.error}"
                )

            elif result.is_skipped():
                broker.record_node_execution(node.name, "skipped", duration)
                logger.info(f"Node '{node.name}' skipped in {duration:.3f}s")

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.failed_nodes.add(node.name)
            broker.record_node_execution(node.name, "failed", duration)

            logger.error(
                f"Node '{node.name}' timed out after {duration:.3f}s "
                f"(timeout={node.timeout}s)"
            )

        except Exception as e:
            duration = time.time() - start_time
            self.failed_nodes.add(node.name)
            broker.record_node_execution(node.name, "failed", duration)

            logger.error(
                f"Node '{node.name}' raised exception in {duration:.3f}s: {e}",
                exc_info=True,
            )

    def reset(self) -> None:
        """Reset the queue manager for a new request.

        Clears all queues and tracking sets.
        """
        self.immediate_queue = asyncio.PriorityQueue()
        self.background_queue = asyncio.PriorityQueue()
        self.completed_nodes.clear()
        self.failed_nodes.clear()
        self.waiting_nodes.clear()

        logger.debug("Task queue manager reset")

    def get_status(self) -> QueueStatus:
        """Get current status of the queue manager.

        Returns:
            QueueStatus dataclass with queue sizes and execution status
        """
        return QueueStatus(
            immediate_queue_size=self.immediate_queue.qsize(),
            background_queue_size=self.background_queue.qsize(),
            completed_nodes=len(self.completed_nodes),
            failed_nodes=len(self.failed_nodes),
            waiting_nodes=len(self.waiting_nodes),
        )
