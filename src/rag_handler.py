# src/rag_handler.py
# -*- coding: utf-8 -*-

"""
(V2 Refactor) 负责 RAG (Retrieval-Augmented Generation) 的核心逻辑，包括：
- 使用配置的 LLM 客户端进行文本嵌入。
- 使用 FAISS 进行向量索引的构建、保存、加载和搜索 (每个项目独立存储)。
- 管理 FAISS 索引 ID 与程序代码块元数据的映射。
- 使用配置的 LLM 客户端结合检索到的上下文生成回应。
"""

import faiss # pip install faiss-cpu
import numpy as np
import pickle
import logging
import os
import time
from typing import List, Dict, Optional, Any, Tuple

# Import configuration, utilities, and LLM factory
from .utils import config_loader as cfg
from .utils import hashing
from .llm_clients.factory import get_client_for_identifier
from .llm_clients.base import LLMClient # Import base class for type hinting

logger = logging.getLogger(__name__)

# Constants (can be moved to config if needed)
DEFAULT_INDEX_FILENAME = "vector_index.faiss"
DEFAULT_METADATA_FILENAME = "vector_metadata.pkl"

class RAGHandler:
    """
    封装 RAG 流程的类别。
    为每个项目维护独立的索引和元数据。
    """
    def __init__(self, project_path: str):
        """
        初始化 RAGHandler，针对特定的项目路径。

        Args:
            project_path (str): 要处理的项目的根目录路径。

        Raises:
            ValueError: 如果无法初始化嵌入客户端。
        """
        self.project_path = os.path.abspath(project_path)
        self.project_hash = hashing.get_project_hash(self.project_path)
        self.project_data_dir = os.path.join(cfg.DATA_ROOT_DIR, self.project_hash)

        # Ensure project-specific data directory exists
        os.makedirs(self.project_data_dir, exist_ok=True)

        self.index_path = os.path.join(self.project_data_dir, DEFAULT_INDEX_FILENAME)
        self.metadata_path = os.path.join(self.project_data_dir, DEFAULT_METADATA_FILENAME)

        logger.info(f"Initializing RAGHandler for project: {self.project_path}")
        logger.info(f"Project Hash: {self.project_hash}")
        logger.info(f"Data directory: {self.project_data_dir}")

        # --- Get Embedding Client ---
        logger.info(f"Attempting to get embedding client for: {cfg.EMBEDDING_MODEL_IDENTIFIER}")
        try:
            self.embedding_client: Optional[LLMClient] = get_client_for_identifier(cfg.EMBEDDING_MODEL_IDENTIFIER)
        except ValueError as e: # Catch initialization errors (like missing keys)
             logger.error(f"Failed to initialize embedding client: {e}")
             raise # Re-raise critical initialization error

        if not self.embedding_client:
            # This case might happen if the provider is unknown, error logged by factory
            err_msg = f"Could not get embedding client for identifier '{cfg.EMBEDDING_MODEL_IDENTIFIER}'. RAG handler cannot function."
            logger.error(err_msg)
            raise ValueError(err_msg) # Raise error as embedding is essential

        self.embedding_dim = cfg.EMBEDDING_DIM # Get dimension from config

        # FAISS index and metadata maps
        self.index: Optional[faiss.Index] = None
        self.faiss_id_to_metadata: Dict[int, Dict[str, Any]] = {} # Map FAISS index (int) -> chunk data (dict)
        self.chunk_id_to_faiss_id: Dict[str, int] = {} # Map unique chunk ID (str) -> FAISS index (int)

        self._load_index_and_metadata()

    def _get_embedding_vector(self, text: str) -> Optional[np.ndarray]:
        """
        Helper method to get embedding using the configured client.
        Returns a NumPy array suitable for FAISS.
        """
        if not self.embedding_client: # Should not happen due to __init__ check, but belt-and-suspenders
             logger.error("Embedding client is not available.")
             return None

        embedding_list = self.embedding_client.get_embedding(text)
        if embedding_list:
            # Ensure correct dimension again here, although client should also check
            if len(embedding_list) != self.embedding_dim:
                 logger.error(f"Embedding dimension mismatch from client! Expected {self.embedding_dim}, got {len(embedding_list)}. Skipping.")
                 return None
            return np.array(embedding_list, dtype=np.float32).reshape(1, -1) # Reshape for FAISS
        return None


    def build_index(self, preprocessed_chunks: List[Dict[str, Any]], force_rebuild: bool = False):
        """
        根据预处理后的程序代码块构建 FAISS 索引和元数据映射。
        索引和元数据将保存在项目特定的目录中。

        Args:
            preprocessed_chunks (list[dict]): 从 preprocessing.py 获取的块列表。
            force_rebuild (bool): 是否强制重新构建索引，即使文件已存在。
        """
        if not force_rebuild and self.index is not None and self.faiss_id_to_metadata:
            logger.info("Index and metadata already loaded. Skipping build. Use force_rebuild=True to override.")
            return

        if not preprocessed_chunks:
            logger.warning("No preprocessed chunks provided to build index.")
            return

        logger.info(f"Building FAISS index and metadata from {len(preprocessed_chunks)} chunks for project {self.project_hash}...")
        start_time = time.time()

        embeddings_list = []
        temp_metadata = {}
        temp_chunk_id_map = {}
        processed_count = 0

        for i, chunk_data in enumerate(preprocessed_chunks):
            chunk_id = chunk_data.get('id')
            code = chunk_data.get('code', '')
            summary = chunk_data.get('summary', '') # May contain failure placeholder

            if not chunk_id:
                logger.warning(f"Skipping chunk at index {i} due to missing 'id'.")
                continue

            # Use summary for embedding only if it's likely valid
            text_to_embed = f"Code Snippet:\n```python\n{code[:2000]}\n```" # Embed code primarily
            if summary and not summary.startswith("["): # Add summary if it seems valid
                 text_to_embed = f"Summary: {summary}\n\n{text_to_embed}"

            embedding_vector = self._get_embedding_vector(text_to_embed)

            if embedding_vector is not None:
                embeddings_list.append(embedding_vector)
                # FAISS index ID corresponds to the index in the embeddings_list
                faiss_index_id = processed_count
                temp_metadata[faiss_index_id] = chunk_data
                temp_chunk_id_map[chunk_id] = faiss_index_id
                processed_count += 1
            else:
                logger.warning(f"Failed to get embedding for chunk {chunk_id}. Skipping.")

            if (i + 1) % 50 == 0 or (i + 1) == len(preprocessed_chunks):
                logger.info(f"  Processed {processed_count}/{i+1} chunks for embedding...")


        if not embeddings_list:
            logger.error("No embeddings were successfully generated. Index cannot be built.")
            self.index = None
            self.faiss_id_to_metadata = {}
            self.chunk_id_to_faiss_id = {}
            return

        # Convert list of (1, dim) arrays to a single (N, dim) array
        all_embeddings = np.vstack(embeddings_list)

        # Create FAISS index
        logger.info(f"Creating FAISS index (IndexFlatL2) with dimension {self.embedding_dim}...")
        try:
            new_index = faiss.IndexFlatL2(self.embedding_dim)
            new_index.add(all_embeddings) # Add vectors to index
            self.index = new_index
            self.faiss_id_to_metadata = temp_metadata
            self.chunk_id_to_faiss_id = temp_chunk_id_map
        except Exception as faiss_err:
             logger.error(f"Failed to create or add embeddings to FAISS index: {faiss_err}", exc_info=True)
             self.index = None
             self.faiss_id_to_metadata = {}
             self.chunk_id_to_faiss_id = {}
             return


        end_time = time.time()
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")
        logger.info(f"Metadata map created with {len(self.faiss_id_to_metadata)} entries.")
        logger.info(f"Index building took {end_time - start_time:.2f} seconds.")

        # Save the new index and metadata
        self.save_index_and_metadata()

    def _load_index_and_metadata(self):
        """从项目特定的文件加载 FAISS 索引和元数据映射。"""
        index_loaded = False
        metadata_loaded = False

        # Load Index
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading FAISS index from {self.index_path}...")
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors.")
                # Check dimension consistency
                if self.index.d != self.embedding_dim:
                    logger.warning(f"Loaded index dimension ({self.index.d}) does not match configured dimension ({self.embedding_dim}). This may cause issues.")
                    # Option: Update self.embedding_dim based on loaded index? Risky.
                    # Option: Refuse to load? Safer.
                    logger.error("Index dimension mismatch. Please rebuild the index or fix EMBEDDING_DIM config.")
                    self.index = None # Invalidate loaded index
                else:
                     index_loaded = True
            except Exception as e:
                logger.error(f"Error loading FAISS index from {self.index_path}: {e}", exc_info=True)
                self.index = None
        else:
            logger.info(f"FAISS index file not found at {self.index_path}. Index needs to be built.")

        # Load Metadata
        if os.path.exists(self.metadata_path):
            try:
                logger.info(f"Loading metadata from {self.metadata_path}...")
                with open(self.metadata_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    # Check format (expecting tuple: (faiss_id_to_meta, chunk_id_to_faiss))
                    if isinstance(loaded_data, tuple) and len(loaded_data) == 2 and isinstance(loaded_data[0], dict) and isinstance(loaded_data[1], dict):
                        self.faiss_id_to_metadata, self.chunk_id_to_faiss_id = loaded_data
                        logger.info(f"Metadata loaded with {len(self.faiss_id_to_metadata)} entries.")
                        metadata_loaded = True
                    else:
                         logger.error(f"Metadata file {self.metadata_path} has unexpected format. Expected tuple(dict, dict).")
                         self.faiss_id_to_metadata = {}
                         self.chunk_id_to_faiss_id = {}
            except Exception as e:
                logger.error(f"Error loading metadata from {self.metadata_path}: {e}", exc_info=True)
                self.faiss_id_to_metadata = {}
                self.chunk_id_to_faiss_id = {}
        else:
            logger.info(f"Metadata file not found at {self.metadata_path}. Metadata needs to be created during build.")

        if not index_loaded or not metadata_loaded:
            logger.warning("Index or metadata not fully loaded. Run build_index() if needed.")

    def save_index_and_metadata(self):
        """保存 FAISS 索引和元数据映射到项目特定的文件。"""
        if self.index is not None:
            try:
                logger.info(f"Saving FAISS index to {self.index_path}...")
                faiss.write_index(self.index, self.index_path)
                logger.info("FAISS index saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self.index_path}: {e}", exc_info=True)
        else:
            logger.warning("FAISS index is None, cannot save.")

        if self.faiss_id_to_metadata and self.chunk_id_to_faiss_id:
            try:
                logger.info(f"Saving metadata to {self.metadata_path}...")
                with open(self.metadata_path, 'wb') as f:
                    # Save both maps together
                    pickle.dump((self.faiss_id_to_metadata, self.chunk_id_to_faiss_id), f)
                logger.info("Metadata saved successfully.")
            except Exception as e:
                logger.error(f"Error saving metadata to {self.metadata_path}: {e}", exc_info=True)
        else:
            logger.warning("Metadata maps are empty or incomplete, cannot save.")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        根据自然语言查询，从索引中检索 top-k 最相关的程序代码块。

        Args:
            query (str): 自然语言查询。
            k (int): 要检索的块数量。

        Returns:
            list[dict]: 检索到的块信息列表，按相关性排序。每个字典包含块的完整数据。
                       如果索引未准备好或检索失败，返回空列表。
        """
        if self.index is None or not self.faiss_id_to_metadata:
            logger.error("FAISS index or metadata is not available. Cannot retrieve. Please run build_index().")
            return []
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Cannot retrieve.")
            return []

        logger.info(f"Retrieving top {k} chunks for query: '{query[:100]}...'")
        start_time = time.time()

        query_embedding = self._get_embedding_vector(query)

        if query_embedding is None:
            logger.error("Failed to get query embedding. Cannot retrieve.")
            return []

        try:
            # Search the FAISS index
            # D: distances (float array), I: indices (int array)
            distances, indices = self.index.search(query_embedding, k)
            end_time = time.time()
            logger.info(f"FAISS search completed in {end_time - start_time:.4f} seconds.")

            retrieved_results = []
            if len(indices) > 0 and len(distances) > 0:
                for i, faiss_id in enumerate(indices[0]): # indices is often [[id1, id2, ...]]
                    if faiss_id != -1: # FAISS uses -1 for invalid indices
                        if faiss_id in self.faiss_id_to_metadata:
                            # Make a copy to avoid modifying the cached metadata
                            chunk_data = self.faiss_id_to_metadata[faiss_id].copy()
                            # Add retrieval score (lower L2 distance is better)
                            chunk_data['retrieval_score'] = float(distances[0][i])
                            retrieved_results.append(chunk_data)
                            logger.debug(f"  Retrieved chunk ID: {chunk_data.get('id')}, Score: {distances[0][i]:.4f}")
                        else:
                            logger.warning(f"Retrieved FAISS index {faiss_id} not found in metadata map!")
            return retrieved_results

        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        通过块的唯一 ID 直接获取预处理后的块数据。

        Args:
            chunk_id (str): 块的唯一 ID (例如 'filepath::function_name')。

        Returns:
            dict | None: 对应的块数据字典，如果找不到则返回 None。
        """
        if not self.chunk_id_to_faiss_id or not self.faiss_id_to_metadata:
            logger.warning("Metadata maps not loaded or empty, cannot get chunk by ID.")
            return None

        faiss_id = self.chunk_id_to_faiss_id.get(chunk_id)
        if faiss_id is not None:
            # Return a copy to prevent accidental modification of cached data
            return self.faiss_id_to_metadata.get(faiss_id, {}).copy()
        else:
            logger.debug(f"Chunk ID '{chunk_id}' not found in chunk_id_to_faiss_id map.")
            return None


    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        结合检索到的上下文，使用配置的 LLM 客户端生成对查询的回应。

        Args:
            query (str): 原始的用户查询。
            retrieved_chunks (list[dict]): 从 retrieve() 方法获取的块列表。

        Returns:
            str: LLM 生成的回应文本。
        """
        # --- Get Generation Client ---
        logger.debug(f"Attempting to get generation client for: {cfg.GENERATION_MODEL_IDENTIFIER}")
        try:
             generation_client = get_client_for_identifier(cfg.GENERATION_MODEL_IDENTIFIER)
        except ValueError as e:
             logger.error(f"Failed to initialize generation client: {e}")
             return f"[生成回答时出错：无法初始化客户端 {cfg.GENERATION_MODEL_IDENTIFIER}]"

        if not generation_client:
            # Error should have been logged by factory/get_client
            return f"[生成回答时出错：无法获取客户端 {cfg.GENERATION_MODEL_IDENTIFIER}]"

        # --- Prepare Context String ---
        context_str: Optional[str] = None
        if not retrieved_chunks:
            logger.warning("No retrieved context provided for generation. Asking LLM without explicit code context.")
            # context_str remains None, client should handle this
            context_str = "没有找到相关的程序代码上下文。" # Or pass None
        else:
            logger.info(f"Generating response based on {len(retrieved_chunks)} retrieved chunks.")
            context_lines = []
            for i, chunk in enumerate(retrieved_chunks):
                context_lines.append(f"--- 上下文片段 {i+1} (ID: {chunk.get('id', 'N/A')}) ---")
                context_lines.append(f"来源: {chunk.get('filepath', 'N/A')} (类型: {chunk.get('type', 'N/A')})")
                summary = chunk.get('summary')
                if summary and not summary.startswith('['): # Add summary if valid
                    context_lines.append(f"功能摘要: {summary}")
                code_snippet = chunk.get('code', '')
                if code_snippet:
                    context_lines.append("程序代码片段 (部分):")
                    context_lines.append("```python")
                    context_lines.append(f"{code_snippet[:500]}...") # Limit context length per chunk
                    context_lines.append("```")
            context_str = "\n".join(context_lines)

        # --- Call Generation Client ---
        logger.debug(f"Calling generate_response on client {type(generation_client).__name__}...")
        try:
            # Pass query and context separately to the client's method
            # The client implementation (Ollama, DeepSeek) will format the final prompt
            generated_text = generation_client.generate_response(
                prompt=query,
                context=context_str
            )
            return generated_text
        except Exception as e:
            # Catch potential errors during the client call itself
            logger.error(f"Error calling generate_response on client {type(generation_client).__name__}: {e}", exc_info=True)
            return f"[生成回答时客户端调用出错: {e}]"


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running RAG Handler Refactor Test ---")

    # Define project path for testing (e.g., current directory)
    TEST_PROJECT_PATH = "."
    # Path to load preprocessed data (assuming it exists from preprocessing test)
    PREPROCESSED_DATA_PICKLE = "preprocessed_data.pkl" # Adjust if needed

    try:
        # 1. Initialize RAG Handler for the test project
        # This will also test client initialization and data dir creation
        rag_handler_instance = RAGHandler(project_path=TEST_PROJECT_PATH)

        # 2. Load preprocessed data for building the index
        preprocessed_chunks_for_test = []
        if os.path.exists(PREPROCESSED_DATA_PICKLE):
             try:
                 with open(PREPROCESSED_DATA_PICKLE, 'rb') as f:
                     preprocessed_chunks_for_test = pickle.load(f)
                 logger.info(f"Loaded {len(preprocessed_chunks_for_test)} preprocessed chunks from {PREPROCESSED_DATA_PICKLE}")
             except Exception as e:
                  logger.error(f"Error loading preprocessed data from {PREPROCESSED_DATA_PICKLE}: {e}", exc_info=True)
        else:
             logger.warning(f"{PREPROCESSED_DATA_PICKLE} not found. Using minimal dummy data.")
             # Create minimal dummy data if file not found
             preprocessed_chunks_for_test = [
                 {'id': 'utils.py::helper_func', 'code': 'def helper_func():\n  print("Helper")', 'summary': '一个简单的辅助函数', 'filepath': 'utils.py', 'type': 'function', 'lineno': 1, 'relationships': []},
                 {'id': 'main.py::run_app', 'code': 'import utils\ndef run_app():\n  utils.helper_func()', 'summary': '运行应用的主函数，调用辅助函数', 'filepath': 'main.py', 'type': 'function', 'lineno': 2, 'relationships': [{'target': 'utils.py::helper_func', 'type': 'CALLS'}]}
             ]

        if not preprocessed_chunks_for_test:
             logger.error("No preprocessed data available. Cannot proceed with test.")
        else:
            # 3. Build index (force rebuild for test consistency)
            logger.info("\n--- Testing Index Building ---")
            rag_handler_instance.build_index(preprocessed_chunks_for_test, force_rebuild=True)

            if rag_handler_instance.index is None or rag_handler_instance.index.ntotal == 0:
                 logger.error("Index building failed or resulted in an empty index.")
            else:
                 logger.info("Index built successfully.")

                 # 4. Test Retrieval
                 test_query = "哪个函数运行应用？"
                 logger.info(f"\n--- Testing Retrieval for query: '{test_query}' ---")
                 retrieved = rag_handler_instance.retrieve(test_query, k=2)
                 print(f"Retrieved {len(retrieved)} chunks:")
                 if retrieved:
                     for i, chunk in enumerate(retrieved):
                         print(f"  {i+1}. ID: {chunk.get('id')}, Score: {chunk.get('retrieval_score'):.4f}, Summary: {chunk.get('summary')}")
                 else:
                      print("  No chunks retrieved.")


                 # 5. Test Generation
                 logger.info(f"\n--- Testing Generation for query: '{test_query}' ---")
                 # Note: Generation depends on the configured generation client (Ollama or DeepSeek)
                 # Ensure the respective service is running and configured correctly in .env
                 generated_response = rag_handler_instance.generate_response(test_query, retrieved)
                 print("\nGenerated Response:")
                 print(generated_response)


                 # 6. Test get_chunk_by_id
                 test_chunk_id = 'main.py::run_app' # Choose an existing ID
                 logger.info(f"\n--- Testing get_chunk_by_id for ID: '{test_chunk_id}' ---")
                 specific_chunk = rag_handler_instance.get_chunk_by_id(test_chunk_id)
                 if specific_chunk:
                     print(f"Found chunk data for {test_chunk_id}:")
                     print(f"  Summary: {specific_chunk.get('summary')}")
                     print(f"  Relationships: {specific_chunk.get('relationships')}")
                 else:
                     print(f"Chunk data not found for ID: {test_chunk_id}")

    except ValueError as init_err:
         logger.error(f"Failed to initialize RAGHandler: {init_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the RAG handler test: {e}", exc_info=True)

    logger.info("\n--- RAG Handler Refactor Test Complete ---")

