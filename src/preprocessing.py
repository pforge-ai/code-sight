# src/preprocessing.py
# -*- coding: utf-8 -*-

"""
(V2 Refactor) 负责对解析后的程序代码节点进行预处理，包括：
1. 定义程序代码分块 (Chunking) 策略 (基于 AST 节点)。
2. 使用配置的 LLM 客户端为每个程序代码块生成语义摘要。
3. 使用配置的 LLM 客户端尝试提取程序代码块之间的角色和关系。
4. 输出结构化的、包含摘要和关系的程序代码块列表，用于后续的 RAG 索引和图构建。
"""

import logging
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# Import configuration, utilities, and LLM factory
from .utils import config_loader as cfg
from .llm_clients.factory import get_client_for_identifier
# Import code_parser only needed for the __main__ test block
# from . import code_parser

logger = logging.getLogger(__name__)

# --- 预处理主函数 ---

def preprocess_project(all_nodes_info: Dict[str, Dict[str, Any]], all_imports_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对整个项目的解析结果进行预处理，提取摘要和关系。
    使用配置的 LLM 客户端和模型标识符执行任务。

    Args:
        all_nodes_info (dict): code_parser 解析出的节点信息。
        all_imports_info (dict): code_parser 解析出的导入信息。

    Returns:
        list[dict]: 包含预处理后信息的块列表。每个字典包含:
                    'id', 'code', 'summary', 'filepath', 'type', 'lineno',
                    'relationships' (list[dict] | None)
    """
    tasks = [] # 收集需要处理的块信息

    logger.info(f"Starting preprocessing for {len(all_nodes_info)} files...")
    logger.info(f"Summarization Model Identifier: {cfg.SUMMARIZATION_MODEL_IDENTIFIER}")
    logger.info(f"Relationship Model Identifier: {cfg.RELATIONSHIP_MODEL_IDENTIFIER}")
    logger.info(f"Max workers: {cfg.PREPROCESSING_MAX_WORKERS}")

    # 1. 收集函数、类别和模块的任务信息
    # Add function and class nodes
    for filepath, nodes_in_file in all_nodes_info.items():
        for node_id, node_data in nodes_in_file.items():
            # Ensure basic required data exists
            if node_data.get('type') in ['function', 'class'] and 'code' in node_data:
                tasks.append({
                    'id': node_id,
                    'type': node_data['type'],
                    'code': node_data.get('code', ''),
                    'filepath': filepath,
                    'lineno': node_data.get('lineno')
                })
            else:
                 logger.warning(f"Skipping node {node_id} due to missing type or code.")

    # Add module nodes (based on imports)
    for filepath, imports in all_imports_info.items():
        module_id = f"{filepath}::module"
        # Create a representative code snippet for the module (imports)
        module_code_for_llm = f"# Imports for module: {filepath}\n"
        import_lines = []
        for alias, info in imports.items():
             if info.get('name'): # from module import name as alias
                  level_dots = '.' * info.get('level', 0)
                  module_part = info.get('module', '')
                  import_lines.append(f"from {level_dots}{module_part} import {info['name']} as {alias}")
             else: # import module as alias
                  import_lines.append(f"import {info.get('module', '')} as {alias}")
        module_code_for_llm += "\n".join(import_lines)

        tasks.append({
            'id': module_id,
            'type': 'module',
            'code': module_code_for_llm, # Use imports as representative code
            'filepath': filepath,
            'lineno': 1 # Module level
        })

    logger.info(f"Created {len(tasks)} preprocessing tasks (functions, classes, modules).")

    # 2. 使用线程池并行处理每个任务 (摘要 + 关系提取)
    preprocessed_chunks: List[Dict[str, Any]] = []
    processed_count = 0
    total_tasks = len(tasks)
    start_process_time = time.time()

    # --- Get client instances outside the loop (if possible/safe) ---
    # Note: If clients have internal state that isn't thread-safe,
    # getting them inside process_single_task might be safer.
    # However, our factory uses caching, so it should return the same instance.
    try:
        summarization_client = get_client_for_identifier(cfg.SUMMARIZATION_MODEL_IDENTIFIER)
        relationship_client = get_client_for_identifier(cfg.RELATIONSHIP_MODEL_IDENTIFIER)
    except ValueError as client_init_error:
         logger.error(f"Failed to initialize required LLM clients: {client_init_error}. Preprocessing aborted.")
         return [] # Abort if essential clients can't be created

    if not summarization_client:
         logger.error(f"Could not get LLM client for summarization ({cfg.SUMMARIZATION_MODEL_IDENTIFIER}). Preprocessing aborted.")
         return []
    if not relationship_client:
         # Relationship extraction might be considered optional, log warning instead of aborting?
         logger.warning(f"Could not get LLM client for relationship extraction ({cfg.RELATIONSHIP_MODEL_IDENTIFIER}). Relationships will not be extracted.")
         # relationship_client will be None, handled in process_single_task


    def process_single_task(task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single code chunk: gets summary and relationships using configured clients."""
        chunk_id = task_data['id']
        chunk_type = task_data['type']
        code_snippet = task_data['code']
        filepath = task_data['filepath']
        summary: Optional[str] = None
        relationships: Optional[List[Dict[str, Any]]] = None

        # Step 1: Get Summary
        if summarization_client:
            try:
                summary = summarization_client.summarize_chunk(
                    chunk_id=chunk_id,
                    chunk_type=chunk_type,
                    code_snippet=code_snippet,
                    filepath=filepath
                )
            except Exception as e:
                logger.error(f"Error during summarization call for {chunk_id}: {e}", exc_info=True)
                summary = "[摘要调用失败]" # Mark as failed
        else:
             # This case should ideally be caught before starting the pool
             logger.error("Summarization client is None, cannot summarize.")
             summary = "[摘要客户端不可用]"


        # Step 2: Get Relationships (only if relationship client is available)
        if relationship_client:
            try:
                # Pass the generated summary (even if failed) as context
                relationships = relationship_client.extract_relationships(
                    chunk_id=chunk_id,
                    chunk_type=chunk_type,
                    code_snippet=code_snippet,
                    filepath=filepath,
                    summary=summary # Pass summary result
                )
            except Exception as e:
                 logger.error(f"Error during relationship extraction call for {chunk_id}: {e}", exc_info=True)
                 relationships = None # Mark as failed / unavailable
        # else: # Relationship client was already logged as unavailable

        # --- Assemble Result ---
        # We include chunks even if summarization failed, marking the summary accordingly.
        # Chunks without successful summaries might be less useful for RAG but could still be indexed.
        if summary is None: # If summarize_chunk returned None (critical failure)
             summary = "[摘要生成失败]"

        return {
            'id': chunk_id,
            'code': code_snippet,
            'summary': summary, # Store summary result (could be failure placeholder)
            'filepath': filepath,
            'type': chunk_type,
            'lineno': task_data.get('lineno'),
            'relationships': relationships # Store relationships result (could be None)
        }


    # --- Execute tasks in parallel ---
    with ThreadPoolExecutor(max_workers=cfg.PREPROCESSING_MAX_WORKERS) as executor:
        future_to_id = {executor.submit(process_single_task, task): task['id'] for task in tasks}

        for future in as_completed(future_to_id):
            chunk_id = future_to_id[future]
            try:
                result_chunk = future.result()
                if result_chunk: # If task returned data (even with failed summary)
                    preprocessed_chunks.append(result_chunk)

                processed_count += 1
                if processed_count % 20 == 0 or processed_count == total_tasks: # Log every 20 tasks
                     elapsed_time = time.time() - start_process_time
                     avg_time = elapsed_time / processed_count if processed_count > 0 else 0
                     logger.info(f"Processed {processed_count}/{total_tasks} tasks... (Avg time per task: {avg_time:.2f}s)")

            except Exception as exc:
                # This catches exceptions raised *within* process_single_task if not handled there,
                # or exceptions during future.result() itself.
                logger.error(f"Task for chunk {chunk_id} generated an exception during processing: {exc}", exc_info=True)

    end_process_time = time.time()
    successful_chunks = len(preprocessed_chunks)
    # Count chunks with valid summaries (not None and not placeholder)
    valid_summary_count = sum(1 for chunk in preprocessed_chunks if chunk.get('summary') and not chunk['summary'].startswith("["))
    logger.info(f"Finished preprocessing. Generated {successful_chunks} chunks out of {total_tasks} tasks.")
    logger.info(f"  Chunks with valid summaries: {valid_summary_count}/{successful_chunks}")
    logger.info(f"Total preprocessing time: {end_process_time - start_process_time:.2f} seconds.")

    return preprocessed_chunks


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set higher level for noisy libraries if needed
    logging.getLogger('httpx').setLevel(logging.WARNING) # Example for Ollama's underlying library if too verbose

    # Import parser locally for test
    from . import code_parser

    test_project_path = "." # Use current directory for test
    logger.info(f"--- Running Preprocessing Test on Project: {os.path.abspath(test_project_path)} ---")

    # 1. Parse the project first (respecting ignore rules defined in config)
    nodes_data, imports_data = code_parser.parse_project(test_project_path)

    if nodes_data or imports_data: # Proceed if parsing found anything
        # 2. Run preprocessing
        preprocessed_data = preprocess_project(nodes_data, imports_data)

        print("\n--- Preprocessing Summary ---")
        print(f"Total chunks generated: {len(preprocessed_data)}")

        # 3. Print example chunks
        print("\nExample Preprocessed Chunks (first 5):")
        for i, chunk in enumerate(preprocessed_data[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"  ID: {chunk.get('id')}")
            print(f"  Type: {chunk.get('type')}")
            print(f"  Filepath: {chunk.get('filepath')}")
            # Truncate long code snippets for display
            code_preview = chunk.get('code', '')
            if len(code_preview) > 150: code_preview = code_preview[:150] + "..."
            print(f"  Code Preview: {code_preview}")
            print(f"  Summary: {chunk.get('summary', 'N/A')}")
            print(f"  Relationships: {chunk.get('relationships', 'N/A')}") # Print extracted relationships
    else:
        print("Parsing failed or found no nodes/imports to preprocess.")

    print("\n--- Preprocessing Test Complete ---")

