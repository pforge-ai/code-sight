# src/llm_clients/ollama_client.py
# -*- coding: utf-8 -*-

import ollama
import logging
import time
import json
import numpy as np
from typing import List, Dict, Optional, Any

# Import base class and configuration
from .base import LLMClient
from ..utils import config_loader as cfg

logger = logging.getLogger(__name__)

class OllamaClient(LLMClient):
    """
    LLMClient implementation for interacting with an Ollama server.
    """

    def __init__(self):
        # Initialize the Ollama client library instance
        # The library typically reads OLLAMA_HOST from environment variables,
        # but we can specify host if needed, though the library handles it.
        # self.client = ollama.Client(host=cfg.OLLAMA_BASE_URL) # If direct control is needed
        logger.info(f"Initializing OllamaClient. Using base URL: {cfg.OLLAMA_BASE_URL}")
        # Test connection or list models (optional)
        try:
            models = ollama.list()
            logger.info(f"Connected to Ollama. Available models: {[m['name'] for m in models]}")
            # Check if required models are available
            required_models = {
                cfg.OLLAMA_EMBEDDING_MODEL,
                cfg.OLLAMA_GENERATION_MODEL,
                cfg.OLLAMA_SUMMARIZATION_MODEL,
                cfg.OLLAMA_RELATIONSHIP_MODEL
            }
            available_names = {m['name'] for m in models}
            missing_models = required_models - available_names
            if missing_models:
                 logger.warning(f"The following configured Ollama models are not available: {missing_models}. Operations using them might fail.")

        except Exception as e:
            logger.error(f"Failed to connect or list models from Ollama at {cfg.OLLAMA_BASE_URL}: {e}", exc_info=True)
            # Depending on severity, could raise an error here

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding vector using the configured Ollama embedding model."""
        if not text or text.isspace():
            logger.warning("Attempted to embed empty or whitespace-only text.")
            return None
        try:
            start_time = time.time()
            response = ollama.embeddings(model=cfg.OLLAMA_EMBEDDING_MODEL, prompt=text)
            duration = time.time() - start_time
            embedding = response.get('embedding')

            if embedding:
                # Validate dimension
                if len(embedding) != cfg.EMBEDDING_DIM:
                    logger.warning(f"Ollama embedding dimension mismatch! Model '{cfg.OLLAMA_EMBEDDING_MODEL}' returned {len(embedding)} dimensions, but expected {cfg.EMBEDDING_DIM}. Check your EMBEDDING_DIM config.")
                    # Decide whether to return the mismatched embedding or None
                    # Returning it might cause issues with FAISS index dimension.
                    # return embedding # Option 1: Return anyway (potential downstream error)
                    return None # Option 2: Return None to indicate failure
                logger.debug(f"Generated embedding for text (len {len(text)}) in {duration:.2f}s using {cfg.OLLAMA_EMBEDDING_MODEL}.")
                return embedding
            else:
                logger.error(f"Ollama embedding API did not return 'embedding' key for model {cfg.OLLAMA_EMBEDDING_MODEL}.")
                return None
        except Exception as e:
            self._handle_error(f"Ollama embedding generation with model {cfg.OLLAMA_EMBEDDING_MODEL}", e)
            return None

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generates a response using the configured Ollama generation model (via chat)."""
        final_prompt = prompt
        if context:
             # Simple context integration, can be refined
             final_prompt = f"""Based on the following context:
--- CONTEXT START ---
{context}
--- CONTEXT END ---

Answer the following question: {prompt}
"""
        logger.debug(f"Sending chat request to Ollama model {cfg.OLLAMA_GENERATION_MODEL}...")
        start_time = time.time()
        try:
            # Use ollama.chat for conversational responses
            response = ollama.chat(
                model=cfg.OLLAMA_GENERATION_MODEL,
                messages=[
                    # Optional: Add a system prompt if needed
                    # {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': final_prompt}
                ]
                # Add other options like temperature if needed: options={'temperature': 0.7}
            )
            duration = time.time() - start_time
            generated_text = response.get('message', {}).get('content', '').strip()
            if generated_text:
                logger.info(f"Generated response (len {len(generated_text)}) in {duration:.2f}s using {cfg.OLLAMA_GENERATION_MODEL}.")
                return generated_text
            else:
                 logger.error(f"Ollama chat API did not return valid content for model {cfg.OLLAMA_GENERATION_MODEL}.")
                 return "[生成回答时出错：未收到有效内容]"

        except Exception as e:
            self._handle_error(f"Ollama chat generation with model {cfg.OLLAMA_GENERATION_MODEL}", e)
            return f"[生成回答时出错: {e}]"

    def summarize_chunk(self, chunk_id: str, chunk_type: str, code_snippet: str, filepath: str) -> Optional[str]:
        """Generates a summary using the configured Ollama summarization model."""
        if not code_snippet or code_snippet.isspace():
            logger.warning(f"Skipping summarization for empty code chunk: {chunk_id}")
            return None

        # Construct prompt based on chunk type (similar to original preprocessing)
        if chunk_type == 'module':
            prompt = f"""请根据以下 Python 模块的顶层代码和导入语句，用一句话简要总结这个模块 '{filepath}' 的主要作用和目的：
```python
{code_snippet[:3000]}
```
模块的主要作用是：""" # Limit context size
        elif chunk_type in ['function', 'class']:
            prompt = f"""请用简洁的自然语言描述以下 Python {chunk_type} '{chunk_id.split('::')[-1]}' 的主要功能和目的：
```python
{code_snippet[:3000]}
```
这个 {chunk_type} 的主要功能是：""" # Limit context size
        else:
            logger.warning(f"Unsupported chunk type for summarization: {chunk_type} for {chunk_id}")
            return None

        logger.debug(f"Requesting summary for {chunk_id} using Ollama model {cfg.OLLAMA_SUMMARIZATION_MODEL}...")
        start_time = time.time()
        try:
            # Use ollama.generate for single-turn summarization
            response = ollama.generate(
                model=cfg.OLLAMA_SUMMARIZATION_MODEL,
                prompt=prompt,
                # options={"num_predict": 100} # Optional: limit response length
            )
            duration = time.time() - start_time
            summary = response.get('response', '').strip()

            if not summary or "无法" in summary or "抱歉" in summary or len(summary) < 5:
                logger.warning(f"Potentially invalid summary received for {chunk_id} from {cfg.OLLAMA_SUMMARIZATION_MODEL}: '{summary}'")
                return f"[摘要生成失败或无效]" # Return placeholder for failure

            logger.debug(f"Summary received for {chunk_id} in {duration:.2f}s.")
            return summary
        except Exception as e:
            self._handle_error(f"Ollama summarization with model {cfg.OLLAMA_SUMMARIZATION_MODEL}", e, chunk_id=chunk_id)
            return None # Indicate failure with None

    def extract_relationships(self, chunk_id: str, chunk_type: str, code_snippet: str, filepath: str, summary: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """Extracts relationships using the configured Ollama relationship model."""
        if not code_snippet or code_snippet.isspace():
            logger.warning(f"Skipping relationship extraction for empty code chunk: {chunk_id}")
            return None

        # Construct prompt (similar to original preprocessing)
        prompt_context = f"文件路径: {filepath}\n类型: {chunk_type}\nID: {chunk_id}\n"
        if summary and "[失败]" not in summary:
            prompt_context += f"已知功能摘要: {summary}\n"
        prompt_context += f"程序代码片段:\n```python\n{code_snippet[:3500]}\n```\n" # Limit context size

        prompt = f"""{prompt_context}
请分析以上 Python 程序代码片段，识别其主要架构角色和关键依赖关系。请仅返回一个 JSON 对象，该对象包含一个名为 'relationships' 的列表。列表中的每个对象应包含 'target' (依赖的模块、类别名或通用概念，如 'database', 'file_io', 'api_call') 和 'type' (关系类型，如 'DEPENDS_ON', 'CALLS', 'IMPLEMENTS', 'CONFIGURES', 'RELATES_TO_CONCEPT')。如果找到关系，请务必包含 'description' 键值对来描述该关系。

如果未找到明确的关系，请返回：
{{ "relationships": [] }}

JSON 对象:
"""

        logger.debug(f"Requesting relationships for {chunk_id} using Ollama model {cfg.OLLAMA_RELATIONSHIP_MODEL}...")
        start_time = time.time()
        try:
            # Use ollama.generate with format="json"
            response = ollama.generate(
                model=cfg.OLLAMA_RELATIONSHIP_MODEL,
                prompt=prompt,
                format="json", # Request JSON output format
                # options={"num_predict": 300} # Optional: Adjust response length limit
            )
            duration = time.time() - start_time
            response_text = response.get('response', '').strip()
            logger.debug(f"Relationship response received for {chunk_id} in {duration:.2f}s: {response_text[:150]}...")

            # Robust JSON parsing
            try:
                # Ollama with format=json should return just the JSON string
                data = json.loads(response_text)
                relationships = data.get('relationships')

                # Basic validation
                if isinstance(relationships, list):
                     # Further validation: ensure items are dicts with required keys
                     valid_rels = []
                     for r in relationships:
                          if isinstance(r, dict) and 'target' in r and 'type' in r:
                               valid_rels.append(r)
                          else:
                               logger.warning(f"Invalid relationship item format in response for {chunk_id}: {r}")
                     logger.debug(f"Successfully parsed {len(valid_rels)} relationships for {chunk_id}.")
                     return valid_rels
                elif relationships is None:
                     logger.warning(f"Parsed JSON for {chunk_id}, but 'relationships' key is missing.")
                     return None
                else:
                     logger.warning(f"Parsed JSON for {chunk_id}, but 'relationships' is not a list: {type(relationships)}")
                     return None

            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to decode JSON relationship response for {chunk_id} from {cfg.OLLAMA_RELATIONSHIP_MODEL}: {json_err}. Response text: {response_text}")
                return None
            except Exception as parse_err:
                 logger.error(f"Error parsing relationship JSON response for {chunk_id}: {parse_err}", exc_info=True)
                 return None

        except Exception as e:
            self._handle_error(f"Ollama relationship extraction with model {cfg.OLLAMA_RELATIONSHIP_MODEL}", e, chunk_id=chunk_id)
            return None # Indicate failure

