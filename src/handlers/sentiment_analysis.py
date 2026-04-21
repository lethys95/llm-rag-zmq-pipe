"""Sentiment analysis handler using composition."""

import logging
import json
import time
from textwrap import dedent
from datetime import datetime
from pydantic import ValidationError

from src.llm.base import BaseLLM
from src.rag.base import BaseRAG
from src.rag.embeddings import EmbeddingService
from src.models.sentiment import SentimentAnalysis

logger = logging.getLogger(__name__)


class SentimentAnalysisHandler:
    """Handler for sentiment analysis using a small LLM.

    This handler composes a BaseLLM provider and RAG system to perform
    sentiment analysis on user messages. It uses composition over inheritance
    to remain flexible and focused on its single responsibility.

    Includes retry logic to handle unpredictable AI model behavior.
    """

    SYSTEM_PROMPT = dedent(
        """
        You are a sentiment analysis assistant. Your ONLY job is to analyze the sentiment of user messages and return a JSON response.

        You must ALWAYS respond with valid JSON in the following format:
        {
          "sentiment": "positive|negative|neutral",
          "confidence": 0.0-1.0,
          "emotional_tone": "happy|sad|angry|confused|anxious|calm|etc",
          "relevance": 0.0-1.0,
          "chrono_relevance": 0.0-1.0,
          "context_summary": "brief description of the specific situation",
          "key_topics": ["topic1", "topic2", ...]
        }

        Rules:
        - sentiment: Must be exactly "positive", "negative", or "neutral"
        - confidence: A number between 0.0 and 1.0 indicating how confident you are
        - emotional_tone: Optional. The primary emotional tone detected
        - relevance: Optional. A float 0.0-1.0 representing general impact/importance
          Example: "death of a family member" = 0.9 (very high relevance)
          Example: "thinking about lunch" = 0.3 (low relevance)
        - chrono_relevance: Optional. A float 0.0-1.0 representing how long this stays relevant over time
          Example: "need to use bathroom urgently" = 0.1 (not relevant after action is taken)
          Example: "death of a family member" = 0.95 (stays relevant for a very long time)
          Example: "preparing for exam tomorrow" = 0.2 (only relevant until exam is over)
        - context_summary: Optional. A brief 1-2 sentence description of the specific situation
          Example: "User's mother passed away recently" or "User needs to use the bathroom urgently"
          This should capture WHO, WHAT specifically, not just generic topics
        - key_topics: Optional. List of main topics or themes in the message (for querying/indexing)

        IMPORTANT: Respond ONLY with valid JSON. No explanations, no additional text. No nothing. Anything else than JSON in that SPECIFIC format means that you have failed your task."""
    )

    def __init__(
        self,
        llm_provider: BaseLLM,
        rag_provider: BaseRAG,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize the sentiment analysis handler.

        Args:
            llm_provider: The LLM provider to use for analysis (composed, not inherited)
            rag_provider: The RAG provider for storing/retrieving sentiment data
            max_retries: Maximum number of retry attempts on failure (default: 3)
            retry_delay: Delay in seconds between retries (default: 0.5)
            embedding_service: Optional EmbeddingService instance (will create singleton if None)
        """
        self.llm = llm_provider
        self.rag = rag_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.embedding_service = embedding_service or EmbeddingService.get_instance()

        logger.info(
            "Sentiment analysis handler initialized (max_retries=%s, retry_delay=%ss)",
            max_retries,
            retry_delay,
        )

    def analyze(self, message: str, speaker: str) -> SentimentAnalysis | None:
        """Analyze the sentiment of a message with retry logic.

        Retries on failures to handle unpredictable AI model behavior.
        Each failure is logged for debugging.

        Args:
            message: The user message to analyze
            speaker: Identifier for who is speaking (memory owner)

        Returns:
            SentimentAnalysis object if successful, None if all attempts fail
        """
        logger.debug("Analyzing sentiment for '%s' message: %s...", speaker, message[:100])

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            sentiment = self._attempt_analysis(message, speaker, attempt)

            if sentiment:
                self._log_success(sentiment, attempt)
                self._store_in_rag(message, sentiment)
                return sentiment
            else:
                last_error = self._handle_failure(attempt, message)

        # All attempts failed
        self._log_final_failure(last_error)
        return None

    def _attempt_analysis(
        self, message: str, speaker: str, attempt: int
    ) -> SentimentAnalysis | None:
        """Attempt a single sentiment analysis.

        Args:
            message: The user message to analyze
            speaker: Identifier for who is speaking
            attempt: Current attempt number

        Returns:
            SentimentAnalysis object if successful, None otherwise
        """
        try:
            logger.debug("Sentiment analysis attempt %s/%s", attempt, self.max_retries)
            response = self.llm.generate(self._build_prompt(message))

            return self._parse_response(response, speaker)

        except json.JSONDecodeError as e:
            logger.error(
                "Attempt %s/%s: JSON parsing error: %s. Raw response causing error: %s",
                attempt,
                self.max_retries,
                e,
                response,
            )
            return None

        except Exception as e:
            logger.error(
                "Attempt %s/%s: Unexpected error: %s",
                attempt,
                self.max_retries,
                e,
                exc_info=True,
            )
            return None

    def _handle_failure(self, attempt: int, message: str) -> str:
        """Handle a failed analysis attempt.

        Args:
            attempt: Current attempt number
            message: The message that was being analyzed

        Returns:
            Error message describing the failure
        """
        error_msg = (
            "Sentiment analysis validation failed for message: %s..." % message[:50]
        )
        logger.warning("Attempt %s/%s: %s", attempt, self.max_retries, error_msg)

        if attempt < self.max_retries:
            logger.debug("Retrying in %ss...", self.retry_delay)
            time.sleep(self.retry_delay)

        return error_msg

    def _log_success(self, sentiment: SentimentAnalysis, attempt: int) -> None:
        """Log a successful sentiment analysis.

        Args:
            sentiment: The successful sentiment analysis result
            attempt: The attempt number that succeeded
        """
        if attempt > 1:
            logger.info(
                "Sentiment analysis successful on attempt %s/%s: %s (confidence: %s)",
                attempt,
                self.max_retries,
                sentiment.sentiment,
                sentiment.confidence,
            )
        else:
            logger.info(
                "Sentiment analysis successful: %s (confidence: %s)",
                sentiment.sentiment,
                sentiment.confidence,
            )

    def _log_final_failure(self, last_error: str | None) -> None:
        """Log final failure after all retry attempts exhausted.

        Args:
            last_error: The last error message encountered
        """
        logger.error(
            "Sentiment analysis failed after %s attempts. Last error: %s",
            self.max_retries,
            last_error,
        )

    def _build_prompt(self, message: str) -> str:
        """Build the full prompt for sentiment analysis.

        Args:
            message: The user message to analyze

        Returns:
            Full prompt with system instructions and user message
        """
        return dedent(f"""
            {self.SYSTEM_PROMPT}

            User message to analyze:
            {message}

            JSON response:""")

    def _parse_response(self, response: str, speaker: str) -> SentimentAnalysis | None:
        """Parse LLM response into SentimentAnalysis object.

        Args:
            response: Raw response from LLM
            speaker: The memory owner to add to the sentiment analysis

        Returns:
            SentimentAnalysis object if parsing successful, None otherwise
        """
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            data["memory_owner"] = speaker
            sentiment = SentimentAnalysis(**data)

            logger.debug("Successfully parsed sentiment: %s", sentiment.model_dump())
            return sentiment

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON response: %s. Raw response: %s",
                e,
                response,
            )
            return None

        except ValidationError as e:
            logger.error("Invalid sentiment data structure: %s", e)
            return None

        except (TypeError, KeyError) as e:
            logger.error("Invalid sentiment data: %s", e)
            return None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain other content.

        Args:
            text: Text that might contain JSON

        Returns:
            Extracted JSON string

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            json.loads(json_str)
            return json_str

        return text

    def _store_in_rag(self, message: str, sentiment: SentimentAnalysis) -> None:
        """Store sentiment analysis results in RAG system.

        Stores the message with its sentiment metadata for future retrieval.
        Uses the RAG provider's store method with embeddings.

        Args:
            message: Original user message
            sentiment: Analyzed sentiment data
        """
        try:
            point_id = self.rag.store(
                text=message,
                embedding=self.embedding_service.encode(message),
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "memory_owner": sentiment.memory_owner,
                    "sentiment": sentiment.sentiment,
                    "confidence": sentiment.confidence,
                    "emotional_tone": sentiment.emotional_tone,
                    "relevance": sentiment.relevance,
                    "chrono_relevance": sentiment.chrono_relevance,
                    "context_summary": sentiment.context_summary,
                    "key_topics": sentiment.key_topics or [],
                },
            )

            logger.debug(
                "Stored sentiment in RAG system: Point ID: %s, Message: %s..., Sentiment: %s (%s)",
                point_id,
                message[:50],
                sentiment.sentiment,
                sentiment.confidence,
            )

        except Exception as e:
            logger.error("Error storing sentiment in RAG: %s", e, exc_info=True)
