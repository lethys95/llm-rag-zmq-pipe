"""Command-line interface for LLM RAG Response Pipe."""

import click
import sys
from pathlib import Path
from typing import Optional

from .config.loader import load_config
from .pipeline.server import PipelineServer
from .utils.logger import setup_logging


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """LLM RAG Response Pipe - ZMQ-based pipeline for LLM and RAG processing."""
    pass


@cli.command()
@click.option(
    '--input-endpoint',
    type=str,
    help='ZMQ ROUTER bind endpoint for receiving requests (e.g., tcp://*:5555)'
)
@click.option(
    '--output-endpoint',
    type=str,
    help='ZMQ PUSH connect endpoint for forwarding responses (e.g., tcp://localhost:5556)'
)
@click.option(
    '--openrouter-model',
    type=str,
    help='OpenRouter model identifier (e.g., z.ai/glm-4.7)'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True, path_type=Path),
    help='Path to JSON configuration file'
)
@click.option(
    '--temperature',
    type=float,
    help='Sampling temperature (0.0 to 2.0)'
)
@click.option(
    '--max-tokens',
    type=int,
    help='Maximum tokens to generate'
)
@click.option(
    '--top-p',
    type=float,
    help='Top-p sampling parameter'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
    help='Logging level'
)
@click.option(
    '--rag-enabled/--no-rag',
    default=None,
    help='Enable or disable RAG functionality'
)
def remote(
    input_endpoint: Optional[str],
    output_endpoint: Optional[str],
    openrouter_model: Optional[str],
    config_file: Optional[Path],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    log_level: Optional[str],
    rag_enabled: Optional[bool],
) -> None:
    """Run the LLM RAG Response Pipe server with a remote LLM provider (OpenRouter).
    
    This command starts the pipeline server using a remote LLM API like OpenRouter.
    The server:
    - Receives prompts via ZMQ ROUTER socket
    - Retrieves context using RAG (if enabled)
    - Generates responses using remote LLM API
    - Sends acknowledgment to requester
    - Forwards responses via ZMQ PUSH socket
    
    Example usage:
    
        # Using OpenRouter with default settings
        llm-rag-pipe remote --input-endpoint tcp://*:5555 --output-endpoint tcp://localhost:5556
    
        # Specify a particular model
        llm-rag-pipe remote --openrouter-model z-ai/glm-4.7
    
        # Using config file with CLI overrides
        llm-rag-pipe remote --config-file config.json --temperature 0.9
    """
    # Build CLI args dictionary
    cli_args = {
        'input_endpoint': input_endpoint,
        'output_endpoint': output_endpoint,
        'primary_llm_provider': 'openrouter',  # Force OpenRouter provider
        'primary_openrouter_model': openrouter_model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'log_level': log_level,
        'rag_enabled': rag_enabled,
    }
    
    _run_server(cli_args, config_file)


@cli.command()
@click.option(
    '--input-endpoint',
    type=str,
    help='ZMQ ROUTER bind endpoint for receiving requests (e.g., tcp://*:5555)'
)
@click.option(
    '--output-endpoint',
    type=str,
    help='ZMQ PUSH connect endpoint for forwarding responses (e.g., tcp://localhost:5556)'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, path_type=Path),
    help='Path to local model file (required for local llama provider)'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True, path_type=Path),
    help='Path to JSON configuration file'
)
@click.option(
    '--n-ctx',
    type=int,
    help='Context window size for llama model'
)
@click.option(
    '--n-threads',
    type=int,
    help='Number of CPU threads for llama model'
)
@click.option(
    '--n-gpu-layers',
    type=int,
    help='Number of GPU layers for llama model (-1 for all)'
)
@click.option(
    '--temperature',
    type=float,
    help='Sampling temperature (0.0 to 2.0)'
)
@click.option(
    '--max-tokens',
    type=int,
    help='Maximum tokens to generate'
)
@click.option(
    '--top-p',
    type=float,
    help='Top-p sampling parameter'
)
@click.option(
    '--top-k',
    type=int,
    help='Top-k sampling parameter'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
    help='Logging level'
)
@click.option(
    '--rag-enabled/--no-rag',
    default=None,
    help='Enable or disable RAG functionality'
)
def local(
    input_endpoint: Optional[str],
    output_endpoint: Optional[str],
    model_path: Optional[Path],
    config_file: Optional[Path],
    n_ctx: Optional[int],
    n_threads: Optional[int],
    n_gpu_layers: Optional[int],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    top_k: Optional[int],
    log_level: Optional[str],
    rag_enabled: Optional[bool],
) -> None:
    """Run the LLM RAG Response Pipe server with a local LLM model (llama.cpp).
    
    This command starts the pipeline server using a local LLM model.
    The server:
    - Receives prompts via ZMQ ROUTER socket
    - Retrieves context using RAG (if enabled)
    - Generates responses using local LLM
    - Sends acknowledgment to requester
    - Forwards responses via ZMQ PUSH socket
    
    Example usage:
    
        # Using local llama model
        llm-rag-pipe local --model-path /path/to/model.gguf --input-endpoint tcp://*:5555
    
        # With GPU acceleration
        llm-rag-pipe local --model-path model.gguf --n-gpu-layers -1
    
        # Using config file with CLI overrides
        llm-rag-pipe local --config-file config.json --temperature 0.7 --n-threads 8
    """
    # Build CLI args dictionary
    cli_args = {
        'input_endpoint': input_endpoint,
        'output_endpoint': output_endpoint,
        'primary_llm_provider': 'llama',  # Force llama provider
        'primary_model_path': str(model_path) if model_path else None,
        'n_ctx': n_ctx,
        'n_threads': n_threads,
        'n_gpu_layers': n_gpu_layers,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'top_k': top_k,
        'log_level': log_level,
        'rag_enabled': rag_enabled,
    }
    
    _run_server(cli_args, config_file)


def _run_server(cli_args: dict, config_file: Optional[Path]) -> None:
    """Common server execution logic shared by remote and local commands."""
    try:
        # Load configuration with cascading precedence
        settings = load_config(cli_args=cli_args, config_file=config_file)
        
        # Set up logging
        setup_logging(settings.log_level)
        
        # Create and run pipeline server
        server = PipelineServer(settings)
        server.run()
        
    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    
    except FileNotFoundError as e:
        click.echo(f"File not found: {e}", err=True)
        sys.exit(1)
    
    except ImportError as e:
        click.echo(f"Import error: {e}", err=True)
        if 'llama' in cli_args.get('primary_llm_provider', '').lower():
            click.echo("\nHint: If you're trying to use the 'llama' provider,", err=True)
            click.echo("make sure to run './setup.sh' to install llama-cpp-python with CUDA support.", err=True)
        sys.exit(1)
    
    except KeyboardInterrupt:
        click.echo("\nShutdown requested by user")
        sys.exit(0)
    
    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        sys.exit(1)
