# src/llm_clients/deepseek_client.py
# -*- coding: utf-8 -*-

import requests # Use requests library for HTTP calls
import logging
import time
import json
from typing import List, Dict, Optional, Any

# Import base class and configuration
from .base import LLMClient
from ..utils import config_loader as cfg
from ..utils.json_utils import extract_json_string

logger = logging.getLogger(__name__)

# Constants for DeepSeek API
DEEPSEEK_API_TIMEOUT = 60 # Timeout for API requests in seconds

class DeepSeekClient(LLMClient):
    """
    LLMClient implementation for interacting with the DeepSeek API.
    Note: DeepSeek primarily provides chat models. Embedding functionality
          may not be available or supported via their public API.
    """

    def __init__(self):
        if not cfg.DEEPSEEK_API_KEY:
            logger.error("DeepSeek API key is not configured in .env (DEEPSEEK_API_KEY). DeepSeekClient cannot function.")
            # Optionally raise an error to prevent instantiation
            raise ValueError("DeepSeek API key is missing.")
        self.api_key = cfg.DEEPSEEK_API_KEY
        self.base_url = cfg.DEEPSEEK_BASE_URL.rstrip('/') # Ensure no trailing slash
        logger.info(f"Initializing DeepSeekClient. Using base URL: {self.base_url}")
        # Note: No simple 'list models' equivalent like Ollama usually

    def _get_headers(self) -> Dict[str, str]:
        """Returns the standard headers for DeepSeek API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding vector using the DeepSeek embedding model.
        NOTE: As of current knowledge, DeepSeek does not provide a public embedding API.
              This method will log a warning and return None.
        """
        logger.warning("DeepSeekClient.get_embedding called, but DeepSeek does not currently offer a public embedding API. Returning None.")
        # If DeepSeek ever releases an embedding API, implement the logic here,
        # similar to the previous version but using the correct endpoint and model ID.
        # Remember to check cfg.DEEPSEEK_EMBEDDING_MODEL configuration.
        return None # Explicitly return None as the service is unavailable

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generates a response using the DeepSeek chat model."""
        endpoint = f"{self.base_url}/chat/completions"
        messages = []
        # Optional: Add a system prompt
        # messages.append({"role": "system", "content": "You are a helpful code analysis assistant."})

        if context:
            # Combine context and prompt for the user message
            user_content = f"""Based on the following context:
--- CONTEXT START ---
{context}
--- CONTEXT END ---

Answer the following question: {prompt}
"""
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": cfg.DEEPSEEK_CHAT_MODEL, # Use the configured chat model
            "messages": messages,
            # Add other parameters like temperature, max_tokens if needed
            # "temperature": 0.7,
            # "max_tokens": 1000,
        }

        logger.debug(f"Sending chat request to DeepSeek model {cfg.DEEPSEEK_CHAT_MODEL} at {endpoint}...")
        start_time = time.time()
        try:
            response = requests.post(endpoint, headers=self._get_headers(), json=payload, timeout=DEEPSEEK_API_TIMEOUT)
            response.raise_for_status()
            duration = time.time() - start_time

            data = response.json()
            if "choices" in data and data["choices"] and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
                generated_text = data["choices"][0]["message"]["content"].strip()
                logger.info(f"Generated DeepSeek response (len {len(generated_text)}) in {duration:.2f}s using {cfg.DEEPSEEK_CHAT_MODEL}.")
                return generated_text
            else:
                logger.error(f"DeepSeek chat API response missing expected data structure. Response: {data}")
                return "[生成回答时出错：API响应格式无效]"

        except requests.exceptions.RequestException as e:
             # Log more details from the response if available
             error_details = ""
             if e.response is not None:
                  try:
                       error_details = e.response.json()
                  except json.JSONDecodeError:
                       error_details = e.response.text
             logger.error(f"Error during DeepSeek chat request to {endpoint}: {e}. Details: {error_details}", exc_info=False) # Set exc_info to False as details are logged
             return f"[生成回答时出错: {e}]"
        except Exception as e:
            self._handle_error(f"DeepSeek chat generation with model {cfg.DEEPSEEK_CHAT_MODEL}", e)
            return f"[生成回答时出错: {e}]"


    def summarize_chunk(self, chunk_id: str, chunk_type: str, code_snippet: str, filepath: str) -> Optional[str]:
        """Generates a summary using the DeepSeek chat model."""
        # Construct prompt based on chunk type
        if chunk_type == 'module':
            prompt = f"""请根据以下 Python 模块的顶层代码和导入语句，用一句话简要总结这个模块 '{filepath}' 的主要作用和目的：
```python
{code_snippet[:3000]}
```
模块的主要作用是："""
        elif chunk_type in ['function', 'class']:
            prompt = f"""请用简洁的自然语言描述以下 Python {chunk_type} '{chunk_id.split('::')[-1]}' 的主要功能和目的：
```python
{code_snippet[:3000]}
```
这个 {chunk_type} 的主要功能是："""
        else:
            logger.warning(f"Unsupported chunk type for summarization: {chunk_type} for {chunk_id}")
            return None

        # Use the generate_response method, treating summarization as a generation task
        # We pass the specific summarization model configured for DeepSeek if needed,
        # but here we assume the main chat model is used for summarization too.
        # If you configure a specific DeepSeek summarization model, you'd use that here.
        logger.debug(f"Requesting summary for {chunk_id} using DeepSeek model {cfg.DEEPSEEK_CHAT_MODEL}...")
        summary = self.generate_response(prompt=prompt) # Context is not typically needed here

        # Check if the response indicates an error
        if summary.startswith("[生成回答时出错"):
            logger.warning(f"Summarization failed for {chunk_id} using DeepSeek. Error: {summary}")
            return None # Indicate failure

        # Basic validation of the summary content
        if not summary or "无法" in summary or "抱歉" in summary or len(summary) < 5:
             logger.warning(f"Potentially invalid summary received for {chunk_id} from DeepSeek: '{summary}'")
             return f"[摘要生成失败或无效]" # Return placeholder

        return summary


    def extract_relationships(self, chunk_id: str, chunk_type: str, code_snippet: str, filepath: str, summary: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """Extracts relationships using the DeepSeek chat model, requesting JSON format."""
        if not code_snippet or code_snippet.isspace():
            logger.warning(f"Skipping relationship extraction for empty code chunk: {chunk_id}")
            return None

        # Construct prompt requesting JSON
        prompt_context = f"文件路径: {filepath}\n类型: {chunk_type}\nID: {chunk_id}\n"
        if summary and "[失败]" not in summary:
            prompt_context += f"已知功能摘要: {summary}\n"
        prompt_context += f"程序代码片段:\n```python\n{code_snippet[:3500]}\n```\n"

        prompt = f"""{prompt_context}
请分析以上 Python 程序代码片段，识别其主要架构角色和关键依赖关系。请严格按照 JSON 格式返回结果，该 JSON 对象必须包含一个名为 'relationships' 的列表。列表中的每个对象应包含 'target' (字符串类型，表示依赖的模块、类别名或通用概念) 和 'type' (字符串类型，表示关系类型，如 'DEPENDS_ON', 'CALLS', 'IMPLEMENTS', 'CONFIGURES', 'RELATES_TO_CONCEPT')。如果找到关系，请务必包含 'description' (字符串类型) 键值对来描述该关系。

如果未找到明确的关系，请返回：
{{ "relationships": [] }}

请确保你的整个输出就是一个合法的 JSON 对象，不要包含任何额外的解释或标记。
JSON 对象:
"""

        endpoint = f"{self.base_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": cfg.DEEPSEEK_CHAT_MODEL, # Use the configured chat model for this task too
            "messages": messages,
            # Attempt to use response_format for JSON mode (OpenAI compatible)
            "response_format": { "type": "json_object" }
        }

        logger.debug(f"Requesting relationships (JSON mode) for {chunk_id} using DeepSeek model {cfg.DEEPSEEK_CHAT_MODEL}...")
        start_time = time.time()
        try:
            response = requests.post(endpoint, headers=self._get_headers(), json=payload, timeout=DEEPSEEK_API_TIMEOUT)
            response.raise_for_status()
            duration = time.time() - start_time

            data = response.json()
            if "choices" in data and data["choices"] and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
                response_text = data["choices"][0]["message"]["content"].strip()
                logger.debug(f"Relationship response received for {chunk_id} in {duration:.2f}s: {response_text[:150]}...")

                # Parse the JSON string returned in the content
                try:
                    
                    rel_data = json.loads(extract_json_string(response_text))
                    relationships = rel_data.get('relationships')

                    # Basic validation
                    if isinstance(relationships, list):
                        valid_rels = []
                        for r in relationships:
                            if isinstance(r, dict) and 'target' in r and 'type' in r:
                                valid_rels.append(r)
                            else:
                                logger.warning(f"Invalid relationship item format in DeepSeek response for {chunk_id}: {r}")
                        logger.debug(f"Successfully parsed {len(valid_rels)} relationships for {chunk_id} from DeepSeek.")
                        return valid_rels
                    else:
                        logger.warning(f"Parsed JSON from DeepSeek for {chunk_id}, but 'relationships' is not a list or missing.")
                        return None

                except json.JSONDecodeError as json_err:
                    logger.warning(f"Failed to decode JSON relationship response for {chunk_id} from DeepSeek: {json_err}. Response text: {response_text}")
                    return None
                except Exception as parse_err:
                    logger.error(f"Error parsing relationship JSON from DeepSeek for {chunk_id}: {parse_err}", exc_info=True)
                    return None
            else:
                logger.error(f"DeepSeek chat API response (JSON mode) missing expected data structure. Response: {data}")
                return None

        except requests.exceptions.RequestException as e:
            error_details = ""
            if e.response is not None:
                 try: error_details = e.response.json()
                 except json.JSONDecodeError: error_details = e.response.text
            logger.error(f"Error during DeepSeek relationship request to {endpoint}: {e}. Details: {error_details}", exc_info=False)
            return None
        except Exception as e:
            self._handle_error(f"DeepSeek relationship extraction with model {cfg.DEEPSEEK_CHAT_MODEL}", e, chunk_id=chunk_id)
            return None

