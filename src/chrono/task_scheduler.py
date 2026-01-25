"""Task scheduler for managing background and scheduled tasks."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task with metadata."""
    
    name: str
    callback: Callable
    interval: timedelta | None = None
    next_run: datetime | None = None
    enabled: bool = True
    last_run: datetime | None = None
    run_count: int = 0


class TaskScheduler:
    """Singleton task scheduler for managing background tasks.
    
    This scheduler handles:
    - Periodic task execution (detox sessions, memory consolidation)
    - One-time task scheduling
    - Task lifecycle management
    - Observer pattern for task completion notifications
    """
    
    _instance: "TaskScheduler" | None = None
    
    def __new__(cls):
        """Create or get singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize task scheduler."""
        self.tasks: dict[str, ScheduledTask] = {}
        self.running = False
        self.observers: list[Callable[[str, object]]] = []
        self._task_lock = asyncio.Lock()
        
        logger.info("Task scheduler initialized")
    
    def register_task(
        self,
        name: str,
        callback: Callable,
        interval: timedelta | None = None,
        initial_delay: timedelta | None = None
    ) -> None:
        """Register a new task.
        
        Args:
            name: Unique task identifier
            callback: Async function to execute
            interval: Time between executions (None for one-time)
            initial_delay: Delay before first execution
        """
        if name in self.tasks:
            logger.warning(f"Task '{name}' already registered, updating")
        
        next_run = datetime.now() + (initial_delay or timedelta(0))
        
        self.tasks[name] = ScheduledTask(
            name=name,
            callback=callback,
            interval=interval,
            next_run=next_run,
            enabled=True
        )
        
        logger.info(f"Registered task '{name}' with interval {interval}")
    
    def unregister_task(self, name: str) -> None:
        """Unregister a task.
        
        Args:
            name: Task identifier to remove
        """
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Unregistered task '{name}'")
    
    def enable_task(self, name: str) -> None:
        """Enable a task.
        
        Args:
            name: Task identifier to enable
        """
        if name in self.tasks:
            self.tasks[name].enabled = True
            logger.debug(f"Enabled task '{name}'")
    
    def disable_task(self, name: str) -> None:
        """Disable a task.
        
        Args:
            name: Task identifier to disable
        """
        if name in self.tasks:
            self.tasks[name].enabled = False
            logger.debug(f"Disabled task '{name}'")
    
    def add_observer(self, observer: Callable[[str, object]]) -> None:
        """Add an observer for task completion notifications.
        
        Args:
            observer: Function to call when task completes
        """
        self.observers.append(observer)
        logger.debug(f"Added observer, total: {len(self.observers)}")
    
    def remove_observer(self, observer: Callable[[str, object]]) -> None:
        """Remove an observer.
        
        Args:
            observer: Observer function to remove
        """
        if observer in self.observers:
            self.observers.remove(observer)
            logger.debug(f"Removed observer, total: {len(self.observers)}")
    
    async def start(self) -> None:
        """Start the task scheduler."""
        if self.running:
            logger.warning("Task scheduler already running")
            return
        
        self.running = True
        logger.info("Starting task scheduler")
        
        # Start background task runner
        asyncio.create_task(self._run_scheduler())
    
    async def stop(self) -> None:
        """Stop the task scheduler."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping task scheduler")
    
    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self.running:
            try:
                await self._check_and_run_tasks()
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in scheduler: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _check_and_run_tasks(self) -> None:
        """Check and run tasks that are due."""
        now = datetime.now()
        
        async with self._task_lock:
            for name, task in self.tasks.items():
                if not task.enabled:
                    continue
                
                if task.next_run is None or task.next_run <= now:
                    await self._execute_task(task)
                    
                    # Schedule next run if interval is set
                    if task.interval:
                        task.next_run = now + task.interval
                    else:
                        task.next_run = None  # One-time task, don't reschedule
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a single task and notify observers.
        
        Args:
            task: The task to execute
        """
        task.last_run = datetime.now()
        task.run_count += 1
        
        logger.info(
            f"Executing task '{task.name}' (run #{task.run_count})"
        )
        
        try:
            result = await task.callback()
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer(task.name, result)
                except Exception as e:
                    logger.error(
                        f"Observer error for task '{task.name}': {e}",
                        exc_info=True
                    )
            
            logger.debug(f"Task '{task.name}' completed successfully")
            
        except Exception as e:
            logger.error(
                f"Task '{task.name}' failed: {e}",
                exc_info=True
            )
    
    def get_task_status(self, name: str) -> dict[str, object] | None:
        """Get status of a task.
        
        Args:
            name: Task identifier
            
        Returns:
            Dictionary with task status information
        """
        if name not in self.tasks:
            return None
        
        task = self.tasks[name]
        
        return {
            "name": task.name,
            "enabled": task.enabled,
            "interval": str(task.interval) if task.interval else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "run_count": task.run_count
        }
    
    def get_all_task_status(self) -> dict[str, dict[str, object]]:
        """Get status of all registered tasks.
        
        Returns:
            Dictionary mapping task names to their status
        """
        return {
            name: self.get_task_status(name)
            for name in self.tasks.keys()
        }
