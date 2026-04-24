"""Command-line interface for LLM RAG Response Pipe."""

import logging
import sys
import tempfile
import threading
import time
from pathlib import Path

import click
import msgpack
import zmq

from .config.settings import Settings
from .orchestrator import Orchestrator
from .utils.logger import setup_logging


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """LLM RAG Response Pipe - ZMQ-based pipeline for LLM and RAG processing."""


@cli.command()
@click.option(
    "--input-endpoint",
    type=str,
    help="ZMQ ROUTER bind endpoint for receiving requests (e.g., tcp://*:5555)",
)
@click.option(
    "--output-endpoint",
    type=str,
    help="ZMQ PUSH connect endpoint for forwarding responses (e.g., tcp://localhost:5556)",
)
@click.option(
    "--openrouter-model",
    type=str,
    help="OpenRouter model identifier (e.g., z.ai/glm-4.7)",
)
@click.option("--temperature", type=float, help="Sampling temperature (0.0 to 2.0)")
@click.option("--max-tokens", type=int, help="Maximum tokens to generate")
@click.option("--top-p", type=float, help="Top-p sampling parameter")
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Logging level",
)
@click.option(
    "--rag-enabled/--no-rag", default=None, help="Enable or disable RAG functionality"
)
def remote(
    input_endpoint: str | None,
    output_endpoint: str | None,
    openrouter_model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    log_level: str | None,
    rag_enabled: bool | None,
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
    """
    settings = Settings()
    settings.primary_llm.provider = "openrouter"

    if input_endpoint is not None:
        settings.zmq_input_endpoint = input_endpoint
    if output_endpoint is not None:
        settings.zmq_output_endpoint = output_endpoint
    if openrouter_model is not None:
        settings.primary_llm.openrouter_model = openrouter_model
    if temperature is not None:
        settings.temperature = temperature
    if max_tokens is not None:
        settings.max_tokens = max_tokens
    if top_p is not None:
        settings.top_p = top_p
    if rag_enabled is not None:
        settings.rag_enabled = rag_enabled

    _run_server(settings, log_level)


@cli.command()
@click.option(
    "--input-endpoint",
    type=str,
    help="ZMQ ROUTER bind endpoint for receiving requests (e.g., tcp://*:5555)",
)
@click.option(
    "--output-endpoint",
    type=str,
    help="ZMQ PUSH connect endpoint for forwarding responses (e.g., tcp://localhost:5556)",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to local model file (required for local llama provider)",
)
@click.option("--n-ctx", type=int, help="Context window size for llama model")
@click.option("--n-threads", type=int, help="Number of CPU threads for llama model")
@click.option(
    "--n-gpu-layers", type=int, help="Number of GPU layers for llama model (-1 for all)"
)
@click.option("--temperature", type=float, help="Sampling temperature (0.0 to 2.0)")
@click.option("--max-tokens", type=int, help="Maximum tokens to generate")
@click.option("--top-p", type=float, help="Top-p sampling parameter")
@click.option("--top-k", type=int, help="Top-k sampling parameter")
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Logging level",
)
@click.option(
    "--rag-enabled/--no-rag", default=None, help="Enable or disable RAG functionality"
)
def local(
    input_endpoint: str | None,
    output_endpoint: str | None,
    model_path: Path | None,
    n_ctx: int | None,
    n_threads: int | None,
    n_gpu_layers: int | None,
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    top_k: int | None,
    log_level: str | None,
    rag_enabled: bool | None,
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
    """
    settings = Settings()
    settings.primary_llm.provider = "llama"

    if input_endpoint is not None:
        settings.zmq_input_endpoint = input_endpoint
    if output_endpoint is not None:
        settings.zmq_output_endpoint = output_endpoint
    if model_path is not None:
        settings.primary_llm.model_path = str(model_path)
    if n_ctx is not None:
        settings.n_ctx = n_ctx
    if n_threads is not None:
        settings.n_threads = n_threads
    if n_gpu_layers is not None:
        settings.n_gpu_layers = n_gpu_layers
    if temperature is not None:
        settings.temperature = temperature
    if max_tokens is not None:
        settings.max_tokens = max_tokens
    if top_p is not None:
        settings.top_p = top_p
    if top_k is not None:
        settings.top_k = top_k
    if rag_enabled is not None:
        settings.rag_enabled = rag_enabled

    _run_server(settings, log_level)


_TEST_INPUT_ENDPOINT = "tcp://127.0.0.1:15555"
_TEST_OUTPUT_ENDPOINT = "tcp://127.0.0.1:15556"
_TEST_BIND_INPUT = "tcp://*:15555"
_TEST_RESPONSE_TIMEOUT_MS = 90_000


@cli.command("test-run")
@click.option(
    "--message", "-m",
    default="Hey, how are you doing today?",
    help="Message to send through the pipeline",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level",
)
def test_run(message: str, log_level: str) -> None:
    """Send a single message through the full pipeline and print the response.

    Runs the orchestrator in-process with ephemeral in-memory storage,
    sends the message over ZMQ, waits for the response, then exits.
    No data is persisted — Qdrant is in-memory, SQLite uses a temp file.
    """
    setup_logging(getattr(logging, log_level.upper()))
    logger = logging.getLogger(__name__)

    # Build settings with ephemeral storage and test-only ZMQ ports
    settings = Settings()
    settings.qdrant.url = None
    settings.qdrant.path = None   # triggers in-memory QdrantClient(":memory:")

    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db.close()
    settings.conversation_store.db_path = tmp_db.name

    # Patch ZMQHandler's class-level settings before the singleton is created
    from .communication.zmq_handler import ZMQHandler
    ZMQHandler._settings.zmq_input_endpoint = _TEST_BIND_INPUT
    ZMQHandler._settings.zmq_output_endpoint = _TEST_OUTPUT_ENDPOINT

    # Bind the output receiver BEFORE the orchestrator starts so the
    # orchestrator DEALER can connect to it immediately on startup
    ctx = zmq.Context()
    output_socket = ctx.socket(zmq.ROUTER)
    output_socket.bind(_TEST_OUTPUT_ENDPOINT)

    orchestrator = Orchestrator(settings=settings)
    orch_thread = threading.Thread(target=orchestrator.run, daemon=True)
    orch_thread.start()

    # Give the ROUTER socket time to bind
    time.sleep(0.5)

    # Connect sender DEALER to orchestrator input
    sender = ctx.socket(zmq.DEALER)
    sender.connect(_TEST_INPUT_ENDPOINT)

    payload = msgpack.packb({"content": message, "speaker": "test_user"}, use_bin_type=True)
    sender.send_multipart([b"dialogue", payload])
    click.echo(f"\n→ Sent: {message!r}\n")

    # Drain the ACK that the orchestrator sends back via ROUTER
    ack_poller = zmq.Poller()
    ack_poller.register(sender, zmq.POLLIN)
    if dict(ack_poller.poll(3000)).get(sender):
        ack = sender.recv_multipart()
        logger.debug("ACK received: %s", ack)

    # Wait for the response forwarded to the output socket
    out_poller = zmq.Poller()
    out_poller.register(output_socket, zmq.POLLIN)
    ready = dict(out_poller.poll(_TEST_RESPONSE_TIMEOUT_MS))

    if ready.get(output_socket):
        frames = output_socket.recv_multipart()
        # ROUTER receives [identity, ..., message] — response is the last frame
        response = frames[-1].decode("utf-8")
        click.echo(f"← Response:\n{response}\n")
    else:
        click.echo("✗ No response received within timeout.", err=True)

    orchestrator.stop()
    sender.close()
    output_socket.close()
    ctx.term()


def _run_server(settings: Settings, log_level: str | None) -> None:
    """Common server execution logic shared by remote and local commands."""
    try:
        settings.validate()

        if log_level is not None:
            setup_logging(getattr(logging, log_level.upper()))
        else:
            setup_logging(settings.log_level)

        orchestrator = Orchestrator(settings=settings)
        orchestrator.run()

    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)

    except FileNotFoundError as e:
        click.echo(f"File not found: {e}", err=True)
        sys.exit(1)

    except ImportError as e:
        click.echo(f"Import error: {e}", err=True)
        if settings.primary_llm.provider in ["llama", "llama_local"]:
            click.echo(
                "\nHint: If you're trying to use the 'llama' provider,", err=True
            )
            click.echo(
                "make sure to run './setup.sh' to install llama-cpp-python with CUDA support.",
                err=True,
            )
        sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\nShutdown requested by user")
        sys.exit(0)

    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        sys.exit(1)
