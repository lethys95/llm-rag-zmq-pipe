"""Integration tests for configuration system."""

import json
import tempfile
from pathlib import Path

import pytest

from src.config.loader import load_config
from src.config.settings import Settings


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""
    
    def test_full_config_cascade(self, monkeypatch: pytest.MonkeyPatch):
        """Test complete configuration cascade with file, env, and CLI."""
        # Create config file
        config_data = {
            "input_endpoint": "tcp://*:1111",
            "output_endpoint": "tcp://localhost:1112",
            "temperature": 0.8,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = Path(f.name)
        
        try:
            # Set environment variables (should override file)
            monkeypatch.setenv("LLM_RAG_PIPE_INPUT_ADDRESS", "tcp://*:2222")
            
            # CLI args (should override both)
            cli_args = {"temperature": 0.5}
            
            settings = load_config(cli_args=cli_args, config_file=config_file)
            
            # CLI should win for temperature
            assert settings.temperature == 0.5
            # Env should win for input_endpoint
            assert settings.input_endpoint == "tcp://*:2222"
            # File should win for output_endpoint
            assert settings.output_endpoint == "tcp://localhost:1112"
            
        finally:
            config_file.unlink()
            
    def test_config_validation_integration(self):
        """Test that invalid configurations are properly rejected."""
        cli_args = {
            "rag_type": "invalid_type",
        }
        
        with pytest.raises(ValueError):
            load_config(cli_args=cli_args)
            
    def test_config_with_all_llm_types(self):
        """Test configuration with different LLM provider types."""
        # Test OpenRouter
        cli_args = {
            "primary_llm_provider": "openrouter",
            "primary_openrouter_model": "anthropic/claude-3.5-sonnet",
        }
        
        settings = load_config(cli_args=cli_args)
        assert settings.primary_llm.provider == "openrouter"
        assert settings.primary_llm.openrouter_model == "anthropic/claude-3.5-sonnet"
        
        # Test local
        cli_args = {
            "primary_llm_provider": "llama_local",
            "primary_model_path": "/path/to/model.gguf",
        }
        
        settings = load_config(cli_args=cli_args)
        assert settings.primary_llm.provider == "llama_local"
        assert settings.primary_llm.model_path == "/path/to/model.gguf"
        
    def test_nested_config_structure(self):
        """Test that flat config properly converts to nested Settings."""
        cli_args = {
            "qdrant_collection_name": "test_collection",
            "qdrant_url": "http://localhost:6333",
            "conversation_db_path": "./test.db",
            "memory_half_life_days": 60.0,
        }
        
        settings = load_config(cli_args=cli_args)
        
        # Check nested structures are properly created
        assert settings.qdrant.collection_name == "test_collection"
        assert settings.qdrant.url == "http://localhost:6333"
        assert settings.conversation_store.db_path == "./test.db"
        assert settings.memory_decay.half_life_days == 60.0
        
    def test_config_with_feature_toggles(self):
        """Test configuration with various feature toggles."""
        cli_args = {
            "rag_enabled": False,
            "enable_sentiment_analysis": False,
            "enable_context_interpreter": True,
        }
        
        settings = load_config(cli_args=cli_args)
        
        assert settings.rag_enabled is False
        assert settings.enable_sentiment_analysis is False
        assert settings.enable_context_interpreter is True
