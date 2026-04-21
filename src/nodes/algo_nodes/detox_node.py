"""Detox scheduler for triggering detox sessions during idle time."""

import logging

from src.nodes.core.base_node import BaseNode

logger = logging.getLogger(__name__)

# This needs to be redone completely.
# Detox scheduling just means taking the detox session and putting it on the task schedule stack.

class DetoxSessionNode(BaseNode):
    """Node that orchestrates a complete detox session.

    This node runs the full detox protocol:
    1. Identify topics discussed
    2. Run nudging algorithm for each topic
    3. Store companion's recalibrated positions
    4. Generate guidance for next conversation
    """