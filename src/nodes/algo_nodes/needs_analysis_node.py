"""Needs analysis node using Maslow's hierarchy."""

import json
import logging
from textwrap import dedent

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.analysis import NeedsAnalysis

logger = logging.getLogger(__name__)


@register_node
class NeedsAnalysisNode(BaseNode):
    """Node that analyzes user's psychological needs using Maslow's hierarchy.

    This node uses LLM analysis to identify unmet psychological needs
    based on conversation context, sentiment, and retrieved memories.
    """

    SYSTEM_PROMPT = dedent("""
        You are a needs analysis assistant. Your job is to analyze user's conversation
        and identify their psychological needs based on Maslow's hierarchy of needs.

        Maslow's Hierarchy:
        1. Physiological Needs: Basic survival needs (food, water, sleep, health)
        2. Safety Needs: Security, stability, freedom from fear
        3. Belonging Needs: Love, friendship, intimacy, family, sense of connection
        4. Esteem Needs: Respect, self-esteem, status, recognition, freedom
        5. Autonomy Needs: Control over one's life, choices, independence
        6. Meaning Needs: Purpose, morality, creativity, spirituality, self-actualization
        7. Growth Needs: Learning, self-improvement, personal development

        For each need level, provide:
        - A score from 0.0 (not relevant) to 1.0 (very important)
        - Brief reasoning for your score

        Then provide:
        1. Primary Needs: Top 2-3 need levels (most urgent)
        2. Unmet Needs: Need levels with score < 0.5
        3. Need Urgency: Overall urgency (0.0 = not urgent, 1.0 = crisis)
        4. Need Persistence: How long will this need persist (0.0 = fleeting, 1.0 = long-term)
        5. Context Summary: Brief explanation of user's situation
        6. Suggested Approach: How to address these needs (e.g., "reflective_listening", "socratic_questioning", "practical_problem_solving")

        Respond in JSON format:
        {
            "physiological": 0.0-1.0,
            "safety": 0.0-1.0,
            "belonging": 0.0-1.0,
            "esteem": 0.0-1.0,
            "autonomy": 0.0-1.0,
            "meaning": 0.0-1.0,
            "growth": 0.0-1.0,
            "primary_needs": ["belonging", "meaning"],
            "unmet_needs": ["belonging"],
            "need_urgency": 0.0-1.0,
            "need_persistence": 0.0-1.0,
            "context_summary": "Brief explanation",
            "suggested_approach": "reflective_listening"
        }""")

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Analyze user needs from context.

        Args:
            broker: Knowledge broker with current context

        Returns:
            NodeResult with needs analysis
        """
        # Get inputs from broker
        dialogue_input = broker.dialogue_input
        sentiment = broker.sentiment_analysis
        memories = broker.get("evaluated_memories", [])
        trust = broker.get("trust_analysis", None)

        # Build prompt for LLM
        prompt = self._build_analysis_prompt(dialogue_input, sentiment, memories, trust)

        try:
            # Call LLM to analyze needs
            response = self.llm.generate(prompt)

            # Parse response into NeedsAnalysis
            needs = self._parse_needs_response(response)

            # Store in broker
            broker.needs_analysis = needs

            logger.info(
                f"Needs analysis completed. Primary needs: {needs.primary_needs}, "
                f"urgency: {needs.need_urgency:.2f}"
            )

            return NodeResult(status=NodeStatus.SUCCESS, data={"needs": needs})

        except Exception as e:
            logger.error(f"Needs analysis failed: {e}", exc_info=True)
            return NodeResult(status=NodeStatus.FAILED, error=str(e))

    def _build_analysis_prompt(self, dialogue_input, sentiment, memories, trust) -> str:
        """Build prompt for needs analysis.

        Args:
            dialogue_input: User's input message
            sentiment: Sentiment analysis
            memories: Evaluated memories
            trust: Trust analysis

        Returns:
            Full prompt for LLM
        """
        parts = [f"User Message: {dialogue_input.content}"]

        if sentiment:
            parts.append(f"Current Sentiment: {sentiment.sentiment}")
            if sentiment.emotional_tone:
                parts.append(f"Emotional Tone: {sentiment.emotional_tone}")
            if sentiment.key_topics:
                topics = ", ".join(sentiment.key_topics)
                parts.append(f"Topics: {topics}")

        if trust:
            if trust.score is not None:
                parts.append(f"Trust Level: {trust.score:.2f}")
            if trust.relationship_stage:
                parts.append(f"Relationship Stage: {trust.relationship_stage}")

        if memories:
            parts.append(f"Relevant Context: {len(memories)} memories retrieved")
            for i, (doc, evaluation) in enumerate(memories[:3], 1):
                summary = evaluation.get("summary") if evaluation else doc.content
                parts.append(f"Memory {i}: {summary}")

        context_str = "\n\n".join(parts)

        return f"{self.SYSTEM_PROMPT}\n\n{context_str}"

    def _parse_needs_response(self, response: str) -> NeedsAnalysis:
        """Parse LLM response into NeedsAnalysis.

        Args:
            response: Raw LLM response

        Returns:
            Parsed NeedsAnalysis object
        """
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                # Create NeedsAnalysis with defaults
                return NeedsAnalysis(
                    memory_owner="user",
                    physiological=data.get("physiological", 0.0),
                    safety=data.get("safety", 0.0),
                    belonging=data.get("belonging", 0.0),
                    esteem=data.get("esteem", 0.0),
                    autonomy=data.get("autonomy", 0.0),
                    meaning=data.get("meaning", 0.0),
                    growth=data.get("growth", 0.0),
                    primary_needs=data.get("primary_needs", []),
                    unmet_needs=data.get("unmet_needs", []),
                    need_urgency=data.get("need_urgency", 0.0),
                    need_persistence=data.get("need_persistence", 0.0),
                    context_summary=data.get("context_summary", ""),
                    suggested_approach=data.get("suggested_approach", ""),
                )

            logger.warning("Could not parse JSON from needs analysis response")
            # Return default needs analysis
            return self._default_needs_analysis()

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse needs analysis response: {e}")
            return self._default_needs_analysis()

    def _default_needs_analysis(self) -> NeedsAnalysis:
        """Return default needs analysis when parsing fails.

        Returns:
            Default NeedsAnalysis object
        """
        return NeedsAnalysis(
            memory_owner="user",
            physiological=0.0,
            safety=0.0,
            belonging=0.0,
            esteem=0.0,
            autonomy=0.0,
            meaning=0.0,
            growth=0.0,
            primary_needs=[],
            unmet_needs=[],
            need_urgency=0.0,
            need_persistence=0.0,
            context_summary="",
            suggested_approach="",
        )
