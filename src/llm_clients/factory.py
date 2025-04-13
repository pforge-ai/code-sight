# src/llm_clients/factory.py
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Optional

# Import the base class and concrete client implementations
from .base import LLMClient
from .ollama_client import OllamaClient 
from .deepseek_client import DeepSeekClient
# Import config only if needed for validation inside factory, usually not required
# from ..utils import config_loader as cfg

logger = logging.getLogger(__name__)

# Cache for client instances (Singleton pattern per provider)
_client_instances: Dict[str, LLMClient] = {}

def get_llm_client(provider: str) -> Optional[LLMClient]:
    """
    Factory function to get an instance of the appropriate LLMClient based on the provider name.
    Implements a simple singleton pattern to reuse client instances.

    Args:
        provider (str): The name of the LLM provider (e.g., 'ollama', 'deepseek').
                        Should be lowercase.

    Returns:
        Optional[LLMClient]: An instance of the requested LLMClient,
                             or None if the provider is unknown or initialization fails.

    Raises:
        ValueError: If a known provider fails during initialization (e.g., missing API key).
    """
    provider = provider.lower() # Normalize provider name

    # Check cache first
    if provider in _client_instances:
        logger.debug(f"Returning cached instance for provider: {provider}")
        return _client_instances[provider]

    # If not in cache, create a new instance
    client_instance: Optional[LLMClient] = None
    logger.info(f"Creating new LLM client instance for provider: {provider}")
    try:
        if provider == 'ollama':
            client_instance = OllamaClient()
        elif provider == 'deepseek':
            # DeepSeekClient __init__ raises ValueError if API key is missing
            client_instance = DeepSeekClient()
        # Add elif blocks here for other future providers
        # elif provider == 'another_provider':
        #     client_instance = AnotherProviderClient()
        else:
            logger.error(f"Unknown LLM provider requested: {provider}")
            return None # Return None for unknown providers

        # Store the successfully created instance in the cache
        if client_instance:
            _client_instances[provider] = client_instance
        return client_instance

    except ValueError as ve:
         # Catch specific errors like missing API keys during initialization
         logger.error(f"Failed to initialize client for provider '{provider}': {ve}")
         raise # Re-raise the error to signal critical failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating client for provider '{provider}': {e}", exc_info=True)
        return None # Return None for unexpected errors


def get_client_for_identifier(identifier: str) -> Optional[LLMClient]:
    """
    Helper function to get the appropriate client based on a model identifier string.

    Args:
        identifier (str): The model identifier string (e.g., "ollama::nomic-embed-text").

    Returns:
        Optional[LLMClient]: The client instance, or None if parsing/creation fails.
    """
    from ..utils import config_loader as cfg # Import config loader here to avoid circular dependency at module level
    provider, _ = cfg.parse_model_identifier(identifier)
    if not provider:
        logger.error(f"Could not determine provider from identifier: {identifier}")
        return None
    try:
        return get_llm_client(provider)
    except ValueError:
         # Initialization failed (e.g., missing key), error already logged by get_llm_client
         return None


# Example of clearing the cache if needed (e.g., for testing)
def clear_client_cache():
    """Clears the cached LLM client instances."""
    global _client_instances
    logger.debug("Clearing LLM client instance cache.")
    _client_instances = {}

