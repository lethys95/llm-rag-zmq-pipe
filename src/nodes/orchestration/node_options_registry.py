import logging
from textwrap import dedent
from typing import Self
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker

logger = logging.getLogger(__name__)


class NodeOptionsRegistry:
    _instance: Self | None = None
    _knowledge_broker: KnowledgeBroker | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._registered_nodes: dict[str, type[BaseNode]] = {}
        self._knowledge_broker = KnowledgeBroker()

    @classmethod
    def get_instance(cls) -> Self:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_option_names(self) -> set[str]:
        return {type_.__name__ for type_ in self._registered_nodes.values()}

    def _get_options_system_prompt_menu_single_item(self, type_: type[BaseNode]) -> str:
        type_instance = type_(type_.__name__)

        return dedent(f"""
        {type_.__name__}
            - {type_instance.get_description()}
        """)

    def get_option_system_prompt_menu(self) -> str:
        return "\n".join(
            [
                self._get_options_system_prompt_menu_single_item(registered_node)
                for registered_node in self._registered_nodes.values()
            ]
        )

    async def execute(self, node_type: str | type[BaseNode]) -> NodeResult | None:
        if isinstance(node_type, str):
            name = node_type
            t_node = self._registered_nodes.get(node_type)
        else:
            name = node_type.__name__
            t_node = self._registered_nodes.get(name)
        if not t_node:
            logger.error(
                f"Error in {self.__class__.__name__}, could not find node type of {node_type}"
            )
            return None
        if not self._knowledge_broker:
            logger.error(
                f"Error in {self.__class__.__name__}, knowledge broker was somehow none?"
            )
            return None

        return await t_node(name).execute(self._knowledge_broker)
