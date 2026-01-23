"""Unit tests for sentiment analysis models."""

import pytest

from src.models.sentiment import SentimentAnalysis, DialogueInput


@pytest.mark.unit
class TestDialogueInput:
    """Tests for DialogueInput Pydantic model."""
    
    def test_create_dialogue_input(self):
        """Test creating a dialogue input."""
        dialogue = DialogueInput(
            content="Hello, how are you?",
            speaker="user"
        )
        
        assert dialogue.content == "Hello, how are you?"
        assert dialogue.speaker == "user"
        assert dialogue.system_prompt_override is None
        
    def test_dialogue_input_with_system_prompt(self):
        """Test creating dialogue input with system prompt override."""
        dialogue = DialogueInput(
            content="Explain quantum physics",
            speaker="user",
            system_prompt_override="You are a physics professor"
        )
        
        assert dialogue.system_prompt_override == "You are a physics professor"


@pytest.mark.unit
class TestSentimentAnalysis:
    """Tests for SentimentAnalysis Pydantic model."""
    
    def test_create_positive_sentiment(self):
        """Test creating a positive sentiment analysis."""
        sentiment = SentimentAnalysis(
            sentiment="positive",
            confidence=0.95,
            memory_owner="user",
            emotional_tone="happy"
        )
        
        assert sentiment.sentiment == "positive"
        assert sentiment.confidence == 0.95
        assert sentiment.memory_owner == "user"
        assert sentiment.emotional_tone == "happy"
        
    def test_create_negative_sentiment(self):
        """Test creating a negative sentiment analysis."""
        sentiment = SentimentAnalysis(
            sentiment="negative",
            confidence=0.85,
            memory_owner="user",
            emotional_tone="frustrated"
        )
        
        assert sentiment.sentiment == "negative"
        assert sentiment.confidence == 0.85
        
    def test_create_neutral_sentiment(self):
        """Test creating a neutral sentiment analysis."""
        sentiment = SentimentAnalysis(
            sentiment="neutral",
            confidence=0.60,
            memory_owner="user"
        )
        
        assert sentiment.sentiment == "neutral"
        assert sentiment.confidence == 0.60
        
    def test_sentiment_validation_valid(self):
        """Test that valid sentiments are accepted."""
        for sent_val in ["positive", "negative", "neutral"]:
            sentiment = SentimentAnalysis(
                sentiment=sent_val,
                confidence=0.9,
                memory_owner="user"
            )
            assert sentiment.sentiment == sent_val
        
    def test_sentiment_validation_invalid(self):
        """Test that invalid sentiments are rejected."""
        with pytest.raises(ValueError):
            SentimentAnalysis(
                sentiment="invalid",
                confidence=0.9,
                memory_owner="user"
            )
            
    def test_confidence_validation_valid(self):
        """Test that valid confidence scores are accepted."""
        # Test boundaries
        sentiment_min = SentimentAnalysis(sentiment="neutral", confidence=0.0, memory_owner="user")
        sentiment_max = SentimentAnalysis(sentiment="positive", confidence=1.0, memory_owner="user")
        sentiment_mid = SentimentAnalysis(sentiment="neutral", confidence=0.5, memory_owner="user")
        
        assert sentiment_min.confidence == 0.0
        assert sentiment_max.confidence == 1.0
        assert sentiment_mid.confidence == 0.5
        
    def test_confidence_validation_invalid(self):
        """Test that invalid confidence scores are rejected."""
        with pytest.raises(ValueError):
            SentimentAnalysis(sentiment="positive", confidence=-0.1, memory_owner="user")
            
        with pytest.raises(ValueError):
            SentimentAnalysis(sentiment="positive", confidence=1.1, memory_owner="user")
            
    def test_sentiment_serialization(self):
        """Test serializing sentiment to dict."""
        sentiment = SentimentAnalysis(
            sentiment="positive",
            confidence=0.95,
            memory_owner="user",
            emotional_tone="happy",
            key_topics=["greeting", "mood"]
        )
        
        data = sentiment.model_dump()
        
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.95
        assert data["memory_owner"] == "user"
        assert data["emotional_tone"] == "happy"
        assert data["key_topics"] == ["greeting", "mood"]
        
    def test_sentiment_deserialization(self):
        """Test deserializing sentiment from dict."""
        data = {
            "sentiment": "negative",
            "confidence": 0.75,
            "memory_owner": "user",
            "emotional_tone": "sad"
        }
        
        sentiment = SentimentAnalysis.model_validate(data)
        
        assert sentiment.sentiment == "negative"
        assert sentiment.confidence == 0.75
        assert sentiment.memory_owner == "user"
        
    def test_sentiment_json_serialization(self):
        """Test JSON serialization round-trip."""
        original = SentimentAnalysis(
            sentiment="neutral",
            confidence=0.5,
            memory_owner="user",
            context_summary="Neutral statement"
        )
        
        json_str = original.model_dump_json()
        reconstructed = SentimentAnalysis.model_validate_json(json_str)
        
        assert reconstructed.sentiment == original.sentiment
        assert reconstructed.confidence == original.confidence
        assert reconstructed.memory_owner == original.memory_owner
        
    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing memory_owner
        with pytest.raises(ValueError):
            SentimentAnalysis(sentiment="positive", confidence=0.5)
            
        # Missing confidence
        with pytest.raises(ValueError):
            SentimentAnalysis(sentiment="positive", memory_owner="user")
            
        # Missing sentiment
        with pytest.raises(ValueError):
            SentimentAnalysis(confidence=0.5, memory_owner="user")
