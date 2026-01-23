"""Local LLM provider using llama-cpp-python."""

import logging

from .base import BaseLLM

logger = logging.getLogger(__name__)

# Conditional import - llama-cpp-python requires special installation
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    logger.warning(
        "llama-cpp-python is not installed. "
        "Run ./setup.sh to install with CUDA support."
    )


class LlamaLocalLLM(BaseLLM):
    """Local LLM provider using llama-cpp-python.
    
    This provider loads and runs models locally using the llama.cpp library
    with optional GPU acceleration.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
    ):
        """Initialize the local LLM provider.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Raises:
            ImportError: If llama-cpp-python is not installed
            ValueError: If model_path is invalid
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please run ./setup.sh to install it with CUDA support."
            )
        
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Configuration: n_ctx={n_ctx}, n_threads={n_threads}, "
                   f"n_gpu_layers={n_gpu_layers}")
        
        self._model = self._load_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
        
        logger.info("Model loaded successfully")
    
    def _load_model(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
    ) -> Llama:
        """Load the llama model.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            n_gpu_layers: Number of layers to offload to GPU
            
        Returns:
            Initialized Llama instance
            
        Raises:
            ValueError: If model cannot be loaded
        """
        try:
            return Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the local LLM.
        
        Args:
            prompt: The input prompt to generate a response for
            
        Returns:
            The generated response as a string
            
        Raises:
            Exception: If generation fails
        """
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        try:
            response = self._model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                echo=False,
            )
            
            # Extract the generated text from the response
            generated_text = response['choices'][0]['text']
            
            logger.debug(f"Generated response: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def close(self) -> None:
        """Clean up resources and close the model.
        
        This implementation doesn't require explicit cleanup,
        as the model will be garbage collected.
        """
        logger.info("Closing local LLM provider")
        self._model = None
