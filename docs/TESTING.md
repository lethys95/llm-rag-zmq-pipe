# Testing Guide

This document provides comprehensive information about the testing infrastructure for the LLM RAG Response Pipe project.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Coverage](#coverage)
- [Continuous Integration](#continuous-integration)

---

## Overview

The project uses **pytest** as the testing framework with several plugins for enhanced functionality:

- `pytest` - Core testing framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `pytest-timeout` - Test timeout management
- `responses` - HTTP request mocking

## Test Structure

```
tests/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared fixtures and configuration
├── test_config_loader.py                # Configuration loading tests
├── test_models_sentiment.py             # Sentiment model tests
├── test_rag_algorithms.py               # RAG algorithm tests
├── test_storage_conversation_store.py   # Conversation storage tests
└── test_integration_config.py           # Integration tests
```

### Fixtures (conftest.py)

Common test fixtures are defined in `conftest.py`:

- `sample_llm_config` - Sample LLM configuration
- `sample_qdrant_config` - Sample Qdrant configuration
- `sample_conversation_store_config` - Sample conversation store config
- `sample_memory_decay_config` - Sample memory decay config
- `sample_settings` - Complete Settings instance
- `temp_config_file` - Temporary config file
- `mock_env_vars` - Mock environment variables
- `temp_db_path` - Temporary database path

## Running Tests

### Install Development Dependencies

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Tests requiring Qdrant
pytest -m requires_qdrant

# Tests requiring LLM access
pytest -m requires_llm

# Exclude slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
pytest tests/test_config_loader.py
pytest tests/test_models_sentiment.py
pytest tests/test_rag_algorithms.py
```

### Run Specific Test Functions

```bash
pytest tests/test_config_loader.py::TestConfigLoader::test_load_default_config
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Show Print Statements

```bash
pytest -s
```

## Test Categories

Tests are organized using pytest markers:

### `@pytest.mark.unit`
Unit tests for individual components in isolation.

**Examples:**
- Configuration loading
- Sentiment model validation
- Memory decay algorithms
- Conversation storage operations

**Characteristics:**
- Fast execution
- No external dependencies
- Use mocks/stubs for dependencies
- Test single functions/classes

### `@pytest.mark.integration`
Integration tests for component interactions.

**Examples:**
- Complete configuration cascade
- End-to-end data flow
- Multiple components working together

**Characteristics:**
- Slower than unit tests
- May use real dependencies
- Test realistic scenarios

### `@pytest.mark.slow`
Tests that take significant time to complete.

**Examples:**
- Large dataset processing
- Complex RAG operations
- Performance benchmarks

### `@pytest.mark.requires_qdrant`
Tests requiring a running Qdrant instance.

**Example:**
```python
@pytest.mark.requires_qdrant
def test_qdrant_integration():
    # Test requiring Qdrant
    pass
```

**Setup:**
```bash
docker-compose up -d qdrant
```

### `@pytest.mark.requires_llm`
Tests requiring LLM model access (local or API).

**Example:**
```python
@pytest.mark.requires_llm
def test_llm_generation():
    # Test requiring actual LLM
    pass
```

## Writing Tests

### Test File Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Basic Test Structure

```python
"""Description of what this test module covers."""

import pytest
from src.module import Component


@pytest.mark.unit
class TestComponent:
    """Tests for Component class."""
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        component = Component()
        result = component.do_something()
        
        assert result == expected_value
        
    def test_error_handling(self):
        """Test that errors are properly handled."""
        component = Component()
        
        with pytest.raises(ValueError):
            component.do_invalid_thing()
```

### Using Fixtures

```python
def test_with_config(sample_settings):
    """Test using the sample_settings fixture."""
    assert sample_settings.input_endpoint == "tcp://*:5555"
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (0.0, 1.0),
    (0.5, 0.707),
    (1.0, 0.5),
])
def test_decay_calculation(input, expected):
    """Test decay with various inputs."""
    result = calculate_decay(input)
    assert abs(result - expected) < 0.01
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test asynchronous operation."""
    result = await some_async_function()
    assert result is not None
```

### Using Mocks

```python
from unittest.mock import Mock, patch

def test_with_mock(mocker):
    """Test using pytest-mock."""
    mock_client = mocker.Mock()
    mock_client.query.return_value = {"results": []}
    
    result = process_with_client(mock_client)
    
    mock_client.query.assert_called_once()
```

## Coverage

### Generate Coverage Reports

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

### Coverage Configuration

Coverage settings are in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "",
    "@abstractmethod",
]
```

### Coverage Goals

- **Overall:** >80%
- **Core modules:** >90%
- **Utilities:** >70%

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

### Do's ✅

- Write descriptive test names
- Test both success and error cases
- Use fixtures for common setup
- Keep tests independent and isolated
- Test edge cases and boundary conditions
- Mock external dependencies
- Use appropriate markers
- Write docstrings for tests
- Aim for fast test execution

### Don'ts ❌

- Don't test external libraries
- Don't make tests depend on each other
- Don't use hardcoded paths (use fixtures)
- Don't skip proper cleanup
- Don't test implementation details
- Don't write overly complex tests
- Don't ignore test warnings

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Ensure package is installed in development mode
pip install -e .
```

### Coverage Not Working

```bash
# Reinstall with dev dependencies
pip install -e ".[dev]"
```

### Slow Test Execution

```bash
# Run only fast tests
pytest -m "not slow"

# Run tests in parallel (install pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

### Database Locked Errors

Use `:memory:` for SQLite tests or ensure proper cleanup:

```python
@pytest.fixture
def temp_store():
    store = ConversationStore(db_path=":memory:")
    yield store
    store.close()
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)

---

For questions or issues with testing, please open an issue on GitHub.
