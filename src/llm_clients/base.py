# src/llm_clients/base.py
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class LLMClient(ABC):
    """
    Abstract Base Class for Large Language Model clients.
    Defines the common interface for interacting with different LLM services.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding vector for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            Optional[List[float]]: The embedding vector as a list of floats,
                                   or None if embedding fails.
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generates a response to a given prompt, potentially using provided context.
        This is primarily used for RAG-style generation.

        Args:
            prompt (str): The main prompt or question.
            context (Optional[str]): Additional context retrieved from documents/code
                                     to help answer the prompt.

        Returns:
            str: The generated response string. Returns an error message string
                 if generation fails.
        """
        pass

    @abstractmethod
    def summarize_chunk(self, chunk_id: str, chunk_type: str, code_snippet: str, filepath: str) -> Optional[str]:
        """
        Generates a concise summary for a given code chunk.

        Args:
            chunk_id (str): Unique identifier for the code chunk.
            chunk_type (str): Type of the code chunk (e.g., 'function', 'class', 'module').
            code_snippet (str): The actual code content of the chunk.
            filepath (str): The path to the file containing the chunk.

        Returns:
            Optional[str]: The generated summary string, or None if summarization fails.
                           May return a placeholder like "[摘要生成失败]" on error.
        """
        pass

    @abstractmethod
    def extract_relationships(self, chunk_id: str, chunk_type: str, code_snippet: str, filepath: str, summary: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """
        (Experimental) Extracts potential relationships (like dependencies, calls, concepts)
        from a given code chunk.

        Args:
            chunk_id (str): Unique identifier for the code chunk.
            chunk_type (str): Type of the code chunk.
            code_snippet (str): The actual code content.
            filepath (str): The path to the file containing the chunk.
            summary (Optional[str]): An optional pre-generated summary to provide context.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of extracted relationships, where each
                                            relationship is a dictionary (e.g.,
                                            {'target': 'db_utils', 'type': 'DEPENDS_ON'}).
                                            Returns None if extraction fails or is not supported.
        """
        pass

    def _handle_error(self, operation: str, error: Exception, chunk_id: Optional[str] = None) -> None:
        """
        Helper method for logging errors consistently.
        Subclasses can override this for more specific error handling.
        """
        if chunk_id:
            logger.error(f"Error during {operation} for chunk {chunk_id}: {error}", exc_info=True)
        else:
            logger.error(f"Error during {operation}: {error}", exc_info=True)

    def __repr__(self) -> str:
        """Provide a basic representation of the client."""
        return f"<{self.__class__.__name__}>"

