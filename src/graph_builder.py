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
    all_nodes_info: Dict[str, Dict[str, Any]], # Still needed for 'calls'
    all_imports_info: Dict[str, Dict[str, Any]], # Needed for call resolution
    # rag_handler_instance: Optional[RAGHandler] = None # Keep commented out unless RAG bootstrap is used
    ) -> nx.DiGraph:
    """
    (V2 Refactor) Builds a richer NetworkX DiGraph using preprocessed data.

    Args:
        preprocessed_chunks (list[dict]): Output from preprocessing.py.
        all_nodes_info (dict): Original node info from code_parser (contains 'calls').
        all_imports_info (dict): Original import info from code_parser.
        # rag_handler_instance (Optional[RAGHandler]): Optional RAG handler for bootstrapping.

    Returns:
        nx.DiGraph: Dependency graph with nodes representing functions, classes, modules,
                    and edges representing calls, containment, and extracted relationships.
    """
    G = nx.DiGraph()
    node_ids_in_graph = set() # Track nodes added to the graph

    logger.info("Building dependency graph V2 using preprocessed data...")

    # --- Pass 1: Add nodes from preprocessed chunks ---
    module_nodes: Dict[str, str] = {} # Map filepath -> module node ID
    for chunk in preprocessed_chunks:
        node_id = chunk.get('id')
        if not node_id:
             logger.warning(f"Skipping chunk due to missing 'id': {chunk}")
             continue

        node_type = chunk.get('type', 'unknown')
        filepath = chunk.get('filepath', 'unknown_file')
        short_name = node_id.split('::')[-1] # Get the last part as short name

        # Basic title for hover info in visualization tools
        title_parts = [
            f"ID: {node_id}",
            f"Type: {node_type}",
            f"File: {filepath}"
        ]
        if chunk.get('lineno'): title_parts.append(f"Line: {chunk['lineno']}")
        summary = chunk.get('summary')
        if summary and not summary.startswith('['): # Add valid summary preview
            title_parts.append(f"Summary: {summary[:150]}...")
        title = "\n".join(title_parts)

        # Add node to graph
        G.add_node(
            node_id,
            label=short_name, # Short name for display
            title=title,      # Hover text
            filepath=filepath,
            type=node_type,
            code=chunk.get('code', ''),
            summary=summary, # Store full summary (or placeholder)
            relationships=chunk.get('relationships'), # Store extracted relationships
            short_name=short_name,
            lineno=chunk.get('lineno'),
            # Store original scope from parser if available (needed for self.method resolution)
            scope=all_nodes_info.get(filepath, {}).get(node_id, {}).get('scope')
        )
        node_ids_in_graph.add(node_id)
        logger.debug(f"Added node: {node_id} (Type: {node_type})")

        # Handle module nodes for hierarchy / containment edges
        if node_type == 'module':
            module_nodes[filepath] = node_id # Map filepath to this module node ID
        elif node_type in ['function', 'class']:
            # Find or create the parent module node
            module_node_id = module_nodes.get(filepath)
            if not module_node_id:
                 # If module wasn't explicitly preprocessed (e.g., empty __init__.py), create a basic node
                 module_node_id = f"{filepath}::module"
                 if module_node_id not in node_ids_in_graph:
                      G.add_node(module_node_id, label=filepath, title=f"Module: {filepath}", type='module', filepath=filepath)
                      node_ids_in_graph.add(module_node_id)
                      module_nodes[filepath] = module_node_id
                      logger.debug(f"Added implicit module node: {module_node_id}")

            # Add containment edge from module to function/class
            if module_node_id in G: # Check module node exists
                 G.add_edge(module_node_id, node_id, type='CONTAINS', label='') # Label often omitted for clarity
                 logger.debug(f"Added containment edge: {module_node_id} -> {node_id}")
            else:
                 # Should not happen if logic above is correct
                 logger.warning(f"Module node {module_node_id} not found when trying to add containment edge to {node_id}")


    logger.info(f"Added {len(G.nodes)} nodes initially from preprocessed chunks.")

    # --- Pass 2: Add edges based on calls (from parser) and extracted relationships (from LLM) ---
    call_edge_count = 0
    rel_edge_count = 0

    # Iterate through original nodes from parser to get accurate 'calls'
    for filepath, nodes_in_file in all_nodes_info.items():
        file_imports = all_imports_info.get(filepath, {}) # Imports for the current file

        for caller_id, node_data in nodes_in_file.items():
            # Ensure caller node exists in our graph (it should if preprocessing included it)
            if caller_id not in node_ids_in_graph:
                # This might happen if preprocessing skipped a node that parser found
                logger.warning(f"Caller node {caller_id} from parser output not found in graph nodes. Skipping its calls.")
                continue

            # A. Process 'calls' detected by code_parser
            for call_info in node_data.get('calls', []):
                call_name = call_info['name']
                call_base = call_info.get('base') # e.g., 'os.path' or 'self' or '[CallResult]'
                resolved_target_id: Optional[str] = None

                # --- Call Resolution Logic ---
                # This remains heuristic-based and can be complex.
                try:
                    if call_base is None: # Direct call: func() or ClassName()
                        # 1. Check for local definition in the same file
                        potential_target_id = f"{filepath}::{call_name}"
                        if potential_target_id in node_ids_in_graph:
                            resolved_target_id = potential_target_id
                        # 2. Check if it's an imported name
                        elif call_name in file_imports:
                            import_info = file_imports[call_name]
                            imported_module = import_info.get('module')
                            imported_name = import_info.get('name') or call_name # Original name if not alias
                            import_level = import_info.get('level', 0)

                            if import_level > 0: # Relative import
                                target_module_path = _resolve_relative_import_path(filepath, imported_module, import_level)
                                if target_module_path:
                                     # Assume imported name is defined in the target module file
                                     potential_target_id = f"{target_module_path}::{imported_name}"
                                     if potential_target_id in node_ids_in_graph: resolved_target_id = potential_target_id
                                     # TODO: Handle 'from .module import *'? Difficult.
                            # else: Absolute import - resolution is harder without full environment knowledge
                            # Could check if 'imported_module.py::imported_name' exists?

                    elif call_base == 'self': # self.method()
                        # Look for method in the same class scope
                        caller_scope = G.nodes[caller_id].get('scope') # Scope of the calling function/method
                        if caller_scope: # Should be non-empty if it's a method
                             class_scope_id = f"{filepath}::{'::'.join(caller_scope)}" # ID of the containing class
                             potential_target_id = f"{class_scope_id}::{call_name}" # Potential method ID
                             if potential_target_id in node_ids_in_graph: resolved_target_id = potential_target_id
                             # TODO: Add inheritance checks? Very complex via static analysis.

                    elif call_base in file_imports: # module_alias.func() or module_alias.ClassName()
                        import_info = file_imports[call_base]
                        imported_module = import_info.get('module') # Original module name
                        import_level = import_info.get('level', 0)

                        if import_level > 0: # Relative import of a module
                             target_module_path = _resolve_relative_import_path(filepath, imported_module or call_base, import_level)
                             if target_module_path:
                                  # Target could be func/class 'call_name' inside target_module_path
                                  potential_target_id = f"{target_module_path}::{call_name}"
                                  if potential_target_id in node_ids_in_graph: resolved_target_id = potential_target_id
                                  # TODO: Handle Class().method() on imported class?
                        # else: Absolute import 'import os; os.path.join()' - hard to resolve statically

                    # Add more heuristics? e.g., calls on variables assigned from imports?

                except Exception as res_err:
                     logger.warning(f"Error during call resolution for '{call_base or ''}.{call_name}' in {caller_id}: {res_err}", exc_info=True)


                # --- Add Edge if Resolved ---
                if resolved_target_id and resolved_target_id in G:
                    if resolved_target_id != caller_id: # Avoid self-loops from calls for now
                        G.add_edge(caller_id, resolved_target_id, type='CALLS', label=f"calls (L{call_info['lineno']})")
                        call_edge_count += 1
                        logger.debug(f"Added call edge: {caller_id} -> {resolved_target_id}")
                # else: logger.debug(f"Call target '{call_base or ''}.{call_name}' not resolved or not in graph.")


            # B. Process 'relationships' extracted by LLM during preprocessing
            source_chunk_data = G.nodes[caller_id] # Get data stored on the node
            extracted_relationships = source_chunk_data.get('relationships')
            if extracted_relationships:
                for rel in extracted_relationships:
                    target_name = rel.get('target')
                    rel_type = rel.get('type', 'RELATED_TO').upper() # Standardize type
                    rel_desc = rel.get('description', '')

                    if not target_name: continue

                    # --- Attempt to resolve target_name to a node ID ---
                    # This is heuristic and might need refinement.
                    resolved_rel_target_id: Optional[str] = None
                    # 1. Check if target_name is an exact node ID (less likely but possible)
                    if target_name in node_ids_in_graph:
                         resolved_rel_target_id = target_name
                    # 2. Check if target_name matches an imported alias in the current file
                    elif target_name in file_imports:
                         # TODO: Resolve the imported module/name similar to call resolution
                         pass # Add resolution logic here if needed
                    # 3. Check if target_name matches a node defined in the same file
                    else:
                         potential_target_id = f"{filepath}::{target_name}"
                         if potential_target_id in node_ids_in_graph:
                              resolved_rel_target_id = potential_target_id
                         else:
                              # Check if it matches just the short name of any node (more ambiguous)
                              # This could lead to incorrect links if names are not unique.
                              # matching_nodes = [nid for nid, data in G.nodes(data=True) if data.get('short_name') == target_name]
                              # if len(matching_nodes) == 1: resolved_rel_target_id = matching_nodes[0]
                              pass # Avoid ambiguous short name matching for now

                    # 4. Handle conceptual links (future enhancement)
                    # if rel_type == 'RELATES_TO_CONCEPT': ...

                    # --- Add Edge if Resolved ---
                    if resolved_rel_target_id and resolved_rel_target_id in G:
                        if caller_id != resolved_rel_target_id: # Avoid self-loops
                            # Add edge with data from relationship
                            G.add_edge(caller_id, resolved_rel_target_id, type=rel_type, label=rel_type, description=rel_desc)
                            rel_edge_count += 1
                            logger.debug(f"Added relationship edge: {caller_id} -[{rel_type}]-> {resolved_rel_target_id}")
                    # else: logger.debug(f"Could not resolve target '{target_name}' for relationship '{rel_type}' from node {caller_id}.")


    logger.info(f"Added {call_edge_count} edges based on parsed calls.")
    logger.info(f"Added {rel_edge_count} edges based on extracted relationships.")
    logger.info(f"Final graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Import necessary modules for test data generation/loading
    from . import code_parser
    from . import preprocessing # Need this to potentially generate test data

    TEST_PROJECT_PATH = "." # Use current directory for test
    PREPROCESSED_DATA_PICKLE = "preprocessed_data.pkl" # Expected output from preprocessing

    logger.info(f"--- Running Graph Builder V2 Test ---")

    # 1. Get required input data (parsing and preprocessing results)
    # Option A: Run parsing and preprocessing first
    # logger.info("Running parser and preprocessor to generate test data...")
    # nodes_data, imports_data = code_parser.parse_project(TEST_PROJECT_PATH)
    # preprocessed_chunks = preprocessing.preprocess_project(nodes_data, imports_data)
    # # Optionally save preprocessed data
    # try:
    #     with open(PREPROCESSED_DATA_PICKLE, 'wb') as f: pickle.dump(preprocessed_chunks, f)
    # except Exception as e: logger.error(f"Failed to save test preprocessed data: {e}")

    # Option B: Load data from previous runs (if available)
    nodes_data, imports_data = code_parser.parse_project(TEST_PROJECT_PATH) # Need parser data anyway
    preprocessed_chunks = []
    if os.path.exists(PREPROCESSED_DATA_PICKLE):
         logger.info(f"Loading preprocessed data from {PREPROCESSED_DATA_PICKLE}")
         try:
             with open(PREPROCESSED_DATA_PICKLE, 'rb') as f:
                 preprocessed_chunks = pickle.load(f)
         except Exception as e:
              logger.error(f"Failed to load preprocessed data: {e}", exc_info=True)
    else:
         logger.warning(f"Preprocessed data file '{PREPROCESSED_DATA_PICKLE}' not found. Graph might be incomplete.")
         # Optionally run preprocessing here if needed
         # preprocessed_chunks = preprocessing.preprocess_project(nodes_data, imports_data)


    # 2. Build the graph
    if nodes_data and imports_data and preprocessed_chunks:
        dependency_graph = build_dependency_graph_v2(
            preprocessed_chunks,
            nodes_data,
            imports_data,
            # rag_handler_instance=None # Keep RAG bootstrap disabled for now
        )

        # 3. Print graph summary
        print("\n--- Graph Build V2 Summary ---")
        try:
            # nx.info might be deprecated depending on version, use basic len checks
            print(f"Nodes: {len(dependency_graph.nodes)}")
            print(f"Edges: {len(dependency_graph.edges)}")
            # Example: Count edge types
            edge_types = {}
            for u, v, data in dependency_graph.edges(data=True):
                edge_type = data.get('type', 'UNKNOWN')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            print("\nEdge Type Counts:")
            for edge_type, count in sorted(edge_types.items()):
                print(f"  - {edge_type}: {count}")
        except Exception as info_err:
             print(f"Could not print graph info: {info_err}")

        # Optional: Save graph for inspection
        # try:
        #     nx.write_gml(dependency_graph, "test_graph_v2.gml")
        #     logger.info("Saved test graph to test_graph_v2.gml")
        # except Exception as save_err:
        #     logger.error(f"Failed to save graph: {save_err}")

    else:
        print("Parsing or loading preprocessed data failed or yielded no data, cannot build graph.")

    print("\n--- Graph Builder V2 Test Complete ---")
