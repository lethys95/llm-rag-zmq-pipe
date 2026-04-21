from __future__ import annotations

_NODE_CLASSES: set[type] = set()


def register_node(cls: type) -> type:
    _NODE_CLASSES.add(cls)
    return cls


def get_registered_classes() -> frozenset[type]:
    return frozenset(_NODE_CLASSES)
