# src/graph_builder.py
# -*- coding: utf-8 -*-

"""
(V2 Refactor) 根据 preprocessing.py 提供的预处理块信息 (包含摘要和关系)
以及 code_parser 提供的原始节点和导入信息，建立项目的依赖关系图。
目标是构建一个更丰富、可能包含层级关系的图。
"""
import networkx as nx
import logging
import os
import pickle # Used only in __main__ for loading test data
from typing import Dict, List, Any, Optional, Tuple

# Import configuration if needed (currently not directly used here)
# from .utils import config_loader as cfg

# Import other necessary refactored modules only if needed (e.g., for type hints or __main__)
# from .preprocessing import preprocess_project # Example if needed
# from .rag_handler import RAGHandler # Example if RAG bootstrapping is re-enabled

logger = logging.getLogger(__name__)

# --- Helper Function (Kept internal to this module for now) ---
def _resolve_relative_import_path(importer_path: str, module_name: Optional[str], level: int) -> Optional[str]:
    """
    Resolves the potential file path for a relative import.

    Args:
        importer_path (str): The relative path of the file performing the import.
        module_name (Optional[str]): The name of the module being imported (e.g., 'utils', 'models.user'). None for 'from . import name'.
        level (int): The relative import level (e.g., 1 for '.', 2 for '..').

    Returns:
        Optional[str]: The potential relative path to the imported .py file (e.g., 'utils.py', 'models/user.py'),
                       or a directory path (e.g., 'models') if importing a package directly.
                       Returns None for absolute imports (level 0) or if resolution fails.
                       Path uses '/' separators.
    """
    if level == 0:
        return None # Absolute import, handled differently

    importer_dir = os.path.dirname(importer_path).replace(os.sep, '/')
    if importer_dir == '.': importer_dir = '' # Handle imports from root

    # Go up 'level' directories
    path_parts = importer_dir.split('/')
    if level > len(path_parts) and importer_dir != '':
         logger.debug(f"Relative import level {level} exceeds directory depth for '{importer_path}'.")
         return None # Cannot go up that many levels

    # Calculate base directory for the import
    if importer_dir == '':
         base_parts = ['..'] * (level -1) # Relative from root
    else:
         base_parts = path_parts[:len(path_parts) - (level - 1)]

    base_dir = "/".join(base_parts) if base_parts else "." # Use '.' if base becomes empty (e.g. from root going up)

    # Construct potential target path
    if module_name:
        module_parts = module_name.split('.')
        target_path = os.path.join(base_dir, *module_parts).replace(os.sep, '/')
        # Could be a .py file or a package directory
        potential_py_file = target_path + ".py"
        potential_package_init = os.path.join(target_path, "__init__.py").replace(os.sep, '/')
        # Simplification: Return the base path, let caller check for .py or /__init__.py later
        # Or, prioritize .py file path for direct linking? Let's return base + .py for now.
        # A more robust solution would check file system or pre-parsed file list.
        logger.debug(f"Resolved relative import '{module_name}' from '{importer_path}' (level {level}) to potential path: {potential_py_file}")
        return potential_py_file # Assume it resolves to a file for linking purposes
    else:
         # Case: from . import name (module_name is None)
         # The target 'name' is resolved directly within the target directory 'base_dir'
         # This function primarily resolves the *module path*, so return base_dir
         logger.debug(f"Resolved relative import 'from .' from '{importer_path}' (level {level}) to directory: {base_dir}")
         return base_dir # Return the directory path


# --- Main Graph Building Function ---
def build_dependency_graph_v2(
    preprocessed_chunks: List[Dict[str, Any]],
    all_nodes_info: Dict[str, Dict[str, Any]],
    all_imports_info: Dict[str, Dict[str, Any]],
    file_tree: List[str],          # 新增参数
    readme_content: Optional[str], # 新增参数
    project_root_path: str        # 新增参数，用于确定根节点
    ) -> nx.DiGraph:
    """
    (V2 Refactor + Filetree) Builds a richer NetworkX DiGraph.

    Args:
        preprocessed_chunks (list[dict]): Output from preprocessing.py.
        all_nodes_info (dict): Original node info from code_parser.
        all_imports_info (dict): Original import info from code_parser.
        file_tree (list[str]): List of relative paths from project root.
        readme_content (Optional[str]): Content of README.md if found.
        project_root_path (str): Absolute path to the project root.

    Returns:
        nx.DiGraph: Dependency graph including file structure.
    """
    G = nx.DiGraph()
    node_ids_in_graph = set()
    project_root_name = os.path.basename(project_root_path) # Or use "." for simplicity

    logger.info("Building dependency graph V2+Filetree...")

    # --- Pass 0: Add File System Nodes ---
    logger.debug("Adding file system nodes...")
    # Add project root node
    G.add_node(project_root_name, label=project_root_name, type='project', title=f"Project Root: {project_root_path}")
    node_ids_in_graph.add(project_root_name)

    # Add nodes for directories and files from the tree
    # Sort paths to potentially process directories before files within them (helps linking)
    for rel_path in sorted(file_tree):
        node_id = rel_path # Use relative path as node ID
        node_label = os.path.basename(rel_path) or rel_path # Use basename or full path if root file
        parent_rel_path = os.path.dirname(rel_path)
        parent_node_id = parent_rel_path if parent_rel_path else project_root_name # Link root files/dirs to project root

        # Determine type and add node
        abs_path = os.path.join(project_root_path, rel_path)
        node_type = 'unknown'
        if os.path.isdir(abs_path):
             node_type = 'directory'
        elif os.path.isfile(abs_path):
             node_type = 'file'
             # Special handling for README
             if rel_path.upper() == 'README.MD':
                  node_type = 'readme' # More specific type for styling

        # Only add node if it doesn't exist (avoid overwriting later code nodes if IDs clash, although using rel_path should be mostly unique)
        if node_id not in node_ids_in_graph:
             G.add_node(node_id, label=node_label, type=node_type, title=f"{node_type.capitalize()}: {rel_path}", filepath=rel_path) # Store relative path here too
             node_ids_in_graph.add(node_id)
             logger.debug(f"Added filesystem node: {node_id} (Type: {node_type})")

             # Add containment edge from parent
             # Ensure parent node exists (it should due to sorted iteration or root node)
             if parent_node_id in G:
                 G.add_edge(parent_node_id, node_id, type='CONTAINS', label='')
                 logger.debug(f"Added containment edge: {parent_node_id} -> {node_id}")
             else:
                  # This might happen if parent was ignored but child wasn't, or root path issues
                  logger.warning(f"Parent node '{parent_node_id}' not found for filesystem node '{node_id}'. Skipping containment edge.")

        # Add README content if this is the README node
        if node_type == 'readme' and readme_content:
             G.nodes[node_id]['content'] = readme_content
             # Optional: Generate and add summary here or in preprocessing
             # summary = summarize_readme(readme_content) # Requires LLM call
             # G.nodes[node_id]['summary'] = summary

    logger.debug("Finished adding file system nodes.")

    # --- Pass 1: Add Code Nodes (Functions, Classes, Modules) from Preprocessed Chunks ---
    logger.debug("Adding code nodes from preprocessed chunks...")
    code_node_parent_map: Dict[str, str] = {} # Map code node ID -> containing file/dir path (rel_path)

    for chunk in preprocessed_chunks:
        node_id = chunk.get('id') # e.g., "src/utils.py::my_func"
        if not node_id: continue

        node_type = chunk.get('type', 'unknown')
        filepath = chunk.get('filepath', 'unknown_file') # Relative path from chunk data
        short_name = node_id.split('::')[-1]
        # ... (rest of the title generation logic as before) ...
        title = f"ID: {node_id}\nType: {node_type}\nFile: {filepath}\n..." # Simplified example

        # Add the code node
        if node_id not in node_ids_in_graph:
            G.add_node(
                node_id,
                label=short_name,
                title=title,
                filepath=filepath, # Keep original filepath info
                type=node_type,
                code=chunk.get('code', ''),
                summary=chunk.get('summary'),
                relationships=chunk.get('relationships'),
                short_name=short_name,
                lineno=chunk.get('lineno'),
                scope=all_nodes_info.get(filepath, {}).get(node_id, {}).get('scope')
            )
            node_ids_in_graph.add(node_id)
            logger.debug(f"Added code node: {node_id} (Type: {node_type})")

            # Find its parent file/module node from Pass 0
            # Map code node ID to its file path for linking
            code_node_parent_map[node_id] = filepath

            # Link Code Node to its File Node
            parent_file_node_id = filepath # Use the relative path as the ID for the file node
            if parent_file_node_id in G and G.nodes[parent_file_node_id]['type'] in ['file', 'readme']: # Ensure parent exists and is a file
                 # Only add containment edge if it's a function or class inside a file
                 if node_type in ['function', 'class']:
                      G.add_edge(parent_file_node_id, node_id, type='DEFINES', label='') # Or 'CONTAINS'
                      logger.debug(f"Added definition edge: {parent_file_node_id} -> {node_id}")
            # Handle module nodes linking (less critical now file structure is explicit)
            elif node_type == 'module':
                 # Module nodes might represent the file itself, link might be redundant
                 # Or they represent imports summary - link to the file?
                 if parent_file_node_id in G:
                      G.add_edge(parent_file_node_id, node_id, type='MODULE_SUMMARY_FOR', label='') # Example new edge type
                      logger.debug(f"Added module summary edge: {parent_file_node_id} -> {node_id}")


        else:
             # Node ID already exists (e.g., a file path conflicted with a code node ID? Unlikely with '::')
             logger.warning(f"Node ID {node_id} already exists in graph. Skipping duplicate add.")


    logger.debug("Finished adding code nodes.")

    # --- Pass 2: Add Edges (Calls, Relationships) ---
    # This part remains largely the same, operating on the code nodes added in Pass 1.
    # Ensure call resolution logic still works with potentially more nodes.
    logger.debug("Adding call and relationship edges...")
    call_edge_count = 0
    rel_edge_count = 0
    # ... (The existing loop iterating through all_nodes_info['calls'] and chunk['relationships']) ...
    # ... (Call resolution logic) ...
    # ... (Relationship resolution logic) ...
    # Important: Make sure the resolved target IDs exist in `node_ids_in_graph` before adding edges.
    # --- (End of Pass 2 code) ---

    logger.info(f"Added {call_edge_count} call edges and {rel_edge_count} relationship edges.")
    logger.info(f"Final graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G