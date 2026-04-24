from src.handlers.memory_advisor import MemoryAdvisorHandler
from src.models.analysis import MemoryEvaluation
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MemoryAdvisorNode(BaseNode):

    def __init__(self, memory_advisor_handler: MemoryAdvisorHandler) -> None:
        super().__init__()
        self.handler = memory_advisor_handler

    def get_description(self) -> str:
        return (
            "Synthesise retrieved and evaluated memories into natural language guidance "
            "for the primary LLM. Run after MemoryEvaluationNode. Produces an AdvisorOutput "
            "with advice and potency — the primary LLM never sees raw memory documents."
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

        output = self.handler.advise(
            message=broker.dialogue_input.content,
            evaluated_memories=evaluated,
        )

        broker.advisor_outputs.append(output)
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"advisor": output.advisor, "potency": output.potency},
        )
