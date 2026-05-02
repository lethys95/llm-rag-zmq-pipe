import asyncio

from src.handlers.memory_advisor import MemoryAdvisorHandler
from src.models.analysis import MemoryEvaluation
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MemoryAdvisorNode(BaseNode):
    dependencies: list[str] = ["MemoryRetrievalNode"]
    min_criticality: float = 0.2

    def __init__(self, memory_advisor_handler: MemoryAdvisorHandler) -> None:
        super().__init__()
        self.handler = memory_advisor_handler

    def get_description(self) -> str:
        return (
            "Synthesises memory documents into natural language guidance for the primary LLM and "
            "appends an AdvisorOutput to broker.advisor_outputs. "
            "The primary LLM never sees raw Qdrant documents — only this advisor's synthesis. "
            "The advice may include: things the user has shared previously, patterns the companion "
            "has observed over time, emotional episodes from prior turns that are relevant now, "
            "preferences and facts extracted from earlier messages, and any unresolved threads "
            "worth acknowledging. Potency reflects how many meaningful memories were found "
            "(0.0=none retrieved, 1.0=rich relevant history). "
            "Skip if broker.retrieved_documents is empty."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        evaluated = broker.evaluated_memories
        if not evaluated and broker.retrieved_documents:
            # MemoryEvaluationNode didn't run — build stub evaluations from decay scores
            evaluated = [
                (doc, MemoryEvaluation(
                    relevance=doc.score,
                    chrono_relevance=float(doc.metadata.get("chrono_relevance", 0.5)),
                    reasoning="(not evaluated)",
                ))
                for doc in broker.retrieved_documents
            ]

        output = await asyncio.to_thread(
            self.handler.advise,
            message=broker.dialogue_input.content,
            evaluated_memories=evaluated,
        )

        broker.advisor_outputs.append(output)
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"advisor": output.advisor, "potency": output.potency},
        )
