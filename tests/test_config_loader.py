"""Unit tests for configuration loading and cascading precedence."""

import json
import tempfile
from pathlib import Path

import pytest

from src.config.loader import (
    load_config,
    _load_config_file,
    _load_env_variables,
    _filter_none_values,
    _create_settings_from_flat_config,
)
from src.config.settings import Settings


@pytest.mark.unit
class TestConfigLoader:
    """Tests for the main load_config function."""
    
    def test_load_default_config(self):
        """Test loading with only default values."""
        settings = load_config()
        
        assert settings.input_endpoint == "tcp://*:5555"
        assert settings.output_endpoint == "tcp://localhost:20501"
        assert settings.rag_enabled is True
        assert settings.enable_sentiment_analysis is True
        
    def test_load_config_from_file(self, temp_config_file: Path):
        """Test loading configuration from a file."""
        settings = load_config(config_file=temp_config_file)
        
        assert settings.input_endpoint == "tcp://*:6666"
        assert settings.output_endpoint == "tcp://localhost:6667"
        assert settings.primary_llm.provider == "openrouter"
        
    def test_load_config_with_env_vars(self, mock_env_vars: dict[str, str]):
        """Test loading configuration with environment variables."""
        settings = load_config()
        
        assert settings.input_endpoint == "tcp://*:7777"
        assert settings.output_endpoint == "tcp://localhost:7778"
        
    def test_load_config_with_cli_args(self):
        """Test loading configuration with CLI arguments."""
        cli_args = {
            "input_endpoint": "tcp://*:9999",
            "output_endpoint": "tcp://localhost:9998",
            "temperature": 0.5,
        }
        
        settings = load_config(cli_args=cli_args)
        
        assert settings.input_endpoint == "tcp://*:9999"
        assert settings.output_endpoint == "tcp://localhost:9998"
        assert settings.temperature == 0.5
        
    def test_cascading_precedence(self, temp_config_file: Path, mock_env_vars: dict[str, str]):
        """Test that CLI args override env vars which override file config."""
        cli_args = {"input_endpoint": "tcp://*:1111"}
        
        settings = load_config(cli_args=cli_args, config_file=temp_config_file)
        
        # CLI should win
        assert settings.input_endpoint == "tcp://*:1111"
        # Env var should win over file
        assert settings.output_endpoint == "tcp://localhost:7778"
        
    def test_none_values_ignored_in_cli_args(self):
        """Test that None values in CLI args don't override lower precedence."""
        cli_args = {
            "input_endpoint": None,
            "temperature": 0.5,
        }
        
        settings = load_config(cli_args=cli_args)
        
        # None value should not override default
        assert settings.input_endpoint == "tcp://*:5555"
        # Non-None value should override
        assert settings.temperature == 0.5
        
    def test_load_config_file_not_found(self):
        """Test that loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(config_file=Path("/nonexistent/config.json"))
            
    def test_load_config_invalid_json(self):
        """Test that loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            path = Path(f.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(config_file=path)
        finally:
            path.unlink()


@pytest.mark.unit
class TestLoadConfigFile:
    """Tests for file loading functionality."""
    
    def test_load_valid_config_file(self, temp_config_file: Path):
        """Test loading a valid configuration file."""
        config = _load_config_file(temp_config_file)
        
        assert config["input_endpoint"] == "tcp://*:6666"
        assert config["primary_llm_provider"] == "openrouter"
        assert config["rag_enabled"] is True
        
    def test_load_missing_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            _load_config_file(Path("/nonexistent/file.json"))


@pytest.mark.unit
class TestLoadEnvVariables:
    """Tests for environment variable loading."""
    
    def test_load_env_variables_present(self, mock_env_vars: dict[str, str]):
        """Test loading when environment variables are set."""
        config = _load_env_variables()
        
        assert config["input_endpoint"] == "tcp://*:7777"
        assert config["output_endpoint"] == "tcp://localhost:7778"
        
    def test_load_env_variables_absent(self, monkeypatch: pytest.MonkeyPatch):
        """Test loading when environment variables are not set."""
        monkeypatch.delenv("LLM_RAG_PIPE_INPUT_ADDRESS", raising=False)
        monkeypatch.delenv("TTS_INPUT_ADDRESS", raising=False)
        
        config = _load_env_variables()
        
        assert config == {}


@pytest.mark.unit
class TestFilterNoneValues:
    """Tests for None value filtering."""
    
    def test_filter_none_values(self):
        """Test that None values are filtered out."""
        input_dict = {
            "key1": "value1",
            "key2": None,
            "key3": 0,
            "key4": False,
            "key5": None,
        }
        
        result = _filter_none_values(input_dict)
        
        assert result == {
            "key1": "value1",
            "key3": 0,
            "key4": False,
        }
        
    def test_filter_empty_dict(self):
        """Test filtering an empty dictionary."""
        result = _filter_none_values({})
        assert result == {}


@pytest.mark.unit
class TestCreateSettingsFromFlatConfig:
    """Tests for Settings object creation from flat config."""
    
    def test_create_settings_basic(self):
        """Test creating Settings from a flat configuration dict."""
        from src.config.defaults import DEFAULT_CONFIG
        
        settings = _create_settings_from_flat_config(DEFAULT_CONFIG)
        
        assert isinstance(settings, Settings)
        assert settings.input_endpoint == DEFAULT_CONFIG["input_endpoint"]
        assert settings.primary_llm.provider == DEFAULT_CONFIG["primary_llm_provider"]
        assert settings.qdrant.collection_name == DEFAULT_CONFIG["qdrant_collection_name"]
        
    def test_create_settings_with_overrides(self):
        """Test creating Settings with custom values."""
        from src.config.defaults import DEFAULT_CONFIG
        
        custom_config = DEFAULT_CONFIG.copy()
        custom_config["input_endpoint"] = "tcp://*:9999"
        custom_config["temperature"] = 0.3
        custom_config["rag_enabled"] = False
        
        settings = _create_settings_from_flat_config(custom_config)
        
        assert settings.input_endpoint == "tcp://*:9999"
        assert settings.temperature == 0.3
        assert settings.rag_enabled is False
