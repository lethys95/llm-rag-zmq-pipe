from __future__ import annotations

import re

_HANDLER_CLASSES: set[type] = set()


def register_handler(cls: type) -> type:
    _HANDLER_CLASSES.add(cls)
    return cls


def get_registered_handler_classes() -> frozenset[type]:
    return frozenset(_HANDLER_CLASSES)


def handler_key(cls: type) -> str:
    """Convert a handler class name to its snake_case dep key.

    EmotionalStateHandler → emotional_state_handler
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
