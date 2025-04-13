# src/code_parser.py
# -*- coding: utf-8 -*-

"""
负责使用 AST (Abstract Syntax Trees) 解析 Python 程序代码文件，
提取函数、类别、导入、调用等结构化信息。
(V2 Refactor: 集成忽略规则)
"""

import ast
import astor # Consider replacing with ast.unparse in Python 3.9+ if possible
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

# Import configuration and utilities
from .utils import config_loader as cfg # Relative import

logger = logging.getLogger(__name__) # Use named logger

class CodeVisitor(ast.NodeVisitor):
    """
    AST 访问器，用于遍历语法树并收集节点信息。
    (基本与原版相同)
    """
    def __init__(self, source_code_lines: List[str], filepath: str):
        self.source_code_lines = source_code_lines
        self.filepath = filepath # Should be relative path
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.current_node_scope: List[str] = []
        self.imports: Dict[str, Dict[str, Any]] = {} # Stores import info for the current file

    def _get_node_code(self, node: ast.AST) -> str:
        """Helper to get the source code of an AST node."""
        try:
            # astor is used for compatibility, ast.unparse is preferred in Python 3.9+
            return astor.to_source(node)
        except Exception as e:
            logger.warning(f"astor failed to get source for node at line {getattr(node, 'lineno', 'N/A')} in {self.filepath}: {e}. Falling back to line numbers.")
            # Fallback using line numbers if available
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and node.end_lineno is not None:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                if 0 <= start_line < end_line <= len(self.source_code_lines):
                    return "\n".join(self.source_code_lines[start_line:end_line])
            return f"[Code Extraction Failed at line {getattr(node, 'lineno', 'N/A')}]"

    def _get_unique_id(self, node_name: str) -> str:
        """Generates a unique ID based on file path and scope."""
        scope_prefix = "::".join(self.current_node_scope)
        if scope_prefix:
            return f"{self.filepath}::{scope_prefix}::{node_name}"
        else:
            return f"{self.filepath}::{node_name}"

    def _get_call_base_name(self, func_node: ast.expr) -> Optional[str]:
        """Helper to get the base object of a method call (e.g., 'os.path')."""
        if isinstance(func_node, ast.Name):
            return None # Direct function call, no base
        elif isinstance(func_node, ast.Attribute):
            base_parts = []
            curr = func_node.value
            while isinstance(curr, ast.Attribute):
                base_parts.insert(0, curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                base_parts.insert(0, curr.id)
            # Handle cases like instance.method() or func().method() less precisely
            elif isinstance(curr, (ast.Call, ast.Subscript, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
                 try:
                     base_parts.insert(0, f"[{type(curr).__name__}]") # Represent result type
                 except Exception:
                      base_parts.insert(0, "[ComplexBase]")
            else:
                 # For other complex types, represent generically
                 base_parts.insert(0, "[ComplexBase]")
            return ".".join(base_parts) if base_parts else None
        return None # Not a simple Name or Attribute call

    def visit_Import(self, node: ast.Import):
        """Handles 'import module' or 'import module as alias'."""
        for alias in node.names:
            module_name = alias.name
            alias_name = alias.asname if alias.asname else module_name
            self.imports[alias_name] = {'module': module_name, 'name': None, 'level': 0}
            logger.debug(f"[{self.filepath}] Found import: {alias_name} = import {module_name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handles 'from module import name' or 'from . import name'."""
        module_name = node.module if node.module else "" # Can be empty for 'from . import ...'
        level = node.level # Relative import level (0 for absolute)
        for alias in node.names:
            original_name = alias.name
            alias_name = alias.asname if alias.asname else original_name
            self.imports[alias_name] = {'module': module_name, 'name': original_name, 'level': level}
            logger.debug(f"[{self.filepath}] Found import: {alias_name} = from {'.' * level}{module_name} import {original_name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visits function definitions."""
        node_name = node.name
        unique_id = self._get_unique_id(node_name)
        logger.debug(f"Visiting Function: {unique_id}")
        self.nodes[unique_id] = {
            'name': node_name, 'type': 'function',
            'code': self._get_node_code(node), 'docstring': ast.get_docstring(node),
            'lineno': node.lineno, 'end_lineno': getattr(node, 'end_lineno', None),
            'calls': [], 'filepath': self.filepath, 'scope': list(self.current_node_scope)
        }
        self.current_node_scope.append(node_name)
        self.generic_visit(node) # Visit children (like calls inside the function)
        self.current_node_scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visits async function definitions (treat similar to FunctionDef)."""
        self.visit_FunctionDef(node) # Reuse FunctionDef logic

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visits class definitions."""
        node_name = node.name
        unique_id = self._get_unique_id(node_name)
        logger.debug(f"Visiting Class: {unique_id}")
        self.nodes[unique_id] = {
            'name': node_name, 'type': 'class',
            'code': self._get_node_code(node), 'docstring': ast.get_docstring(node),
            'lineno': node.lineno, 'end_lineno': getattr(node, 'end_lineno', None),
            'calls': [], 'filepath': self.filepath, 'scope': list(self.current_node_scope) # Store scope *before* entering class
        }
        self.current_node_scope.append(node_name)
        self.generic_visit(node) # Visit children (methods, nested classes)
        self.current_node_scope.pop()

    def visit_Call(self, node: ast.Call):
        """Visits function/method calls."""
        call_name: Optional[str] = None
        call_base: Optional[str] = None

        # Determine the name and base of the call
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
            call_base = self._get_call_base_name(node.func)
        else:
            # Handle other callable types less specifically (e.g., lambda, subscript result)
             try:
                 call_name = f"[{type(node.func).__name__}_Result]"
             except Exception:
                  call_name = "[ComplexCall]"
             logger.debug(f"  Call type: Other ({type(node.func).__name__}) at line {node.lineno}")


        if call_name:
            # Find the ID of the immediate containing function/method/class scope
            current_scope_id = None
            if self.current_node_scope:
                # Reconstruct the ID of the containing node
                containing_scope_name = self.current_node_scope[-1]
                scope_prefix_list = self.current_node_scope[:-1]
                scope_prefix = "::".join(scope_prefix_list)
                if scope_prefix:
                    current_scope_id = f"{self.filepath}::{scope_prefix}::{containing_scope_name}"
                else:
                    current_scope_id = f"{self.filepath}::{containing_scope_name}"

            if current_scope_id and current_scope_id in self.nodes:
                call_info = {
                    'name': call_name,
                    'base': call_base, # Can be None
                    'lineno': node.lineno,
                    'resolved_target': None # To be filled later if possible
                }
                self.nodes[current_scope_id]['calls'].append(call_info)
                logger.debug(f"  Added call '{call_base or ''}.{call_name}' to node '{current_scope_id}'")
            else:
                # Log calls made at the module level or if scope ID wasn't found
                if not self.current_node_scope:
                     logger.debug(f"  Call '{call_base or ''}.{call_name}' at line {node.lineno} occurred at module level.")
                else:
                     logger.debug(f"  Call '{call_base or ''}.{call_name}' at line {node.lineno} occurred but containing scope '{current_scope_id}' not found in nodes dict.")

        # Continue visiting child nodes (arguments of the call, etc.)
        self.generic_visit(node)


def parse_python_file(filepath: str, project_root: str) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Parses a single Python file using AST.

    Args:
        filepath (str): Absolute path to the Python file.
        project_root (str): Absolute path to the project root directory.

    Returns:
        Optional[Tuple[str, Dict, Dict]]: (relative_filepath, nodes_dict, imports_dict)
                                          Returns None if parsing fails.
    """
    try:
        # Calculate relative path for IDs and reporting
        relative_filepath = os.path.relpath(filepath, project_root).replace(os.sep, '/')
        logger.info(f"Parsing file: {relative_filepath}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: # Ignore decoding errors
            source_code = f.read()
        tree = ast.parse(source_code, filename=relative_filepath) # Provide filename for better error messages
        source_code_lines = source_code.splitlines()
        visitor = CodeVisitor(source_code_lines, relative_filepath)
        visitor.visit(tree)
        return relative_filepath, visitor.nodes, visitor.imports
    except SyntaxError as e:
        logger.error(f"SyntaxError parsing {relative_filepath}: {e}")
        return None
    except Exception as e:
        # Log other errors like MemoryError, RecursionError etc.
        logger.error(f"Unexpected error parsing {relative_filepath}: {e}", exc_info=True)
        return None

def parse_project(project_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parses all Python files in a project directory, respecting ignore patterns.

    Args:
        project_dir (str): The root directory of the project to parse.

    Returns:
        Tuple[Dict, Dict]: (all_nodes_info, all_imports_info)
                           Dictionaries keyed by relative file path.
    """
    all_nodes_info: Dict[str, Dict[str, Any]] = {}
    all_imports_info: Dict[str, Dict[str, Any]] = {}
    python_files_to_parse: List[str] = []
    absolute_project_dir = os.path.abspath(project_dir)

    logger.info(f"Starting project parsing for: {absolute_project_dir}")
    logger.info(f"Using ignore patterns: {cfg.IGNORE_PATTERNS}")

    for root, dirs, files in os.walk(absolute_project_dir, topdown=True):
        # --- Directory Ignore Logic ---
        # Modify dirs in-place to prevent walking into ignored directories
        original_dirs = list(dirs) # Copy original list for iteration
        dirs[:] = [] # Clear the list that os.walk will use
        for d in original_dirs:
             dir_abs_path = os.path.join(root, d)
             dir_rel_path = os.path.relpath(dir_abs_path, absolute_project_dir).replace(os.sep, '/')
             # Check both 'dir_name' and 'dir_name/' patterns
             if not cfg.should_ignore(dir_rel_path) and not cfg.should_ignore(dir_rel_path + '/'):
                 dirs.append(d) # Keep directory if not ignored
             else:
                  logger.debug(f"Ignoring directory: {dir_rel_path}")

        # --- File Ignore Logic ---
        for file in files:
            if file.endswith('.py'):
                file_abs_path = os.path.join(root, file)
                file_rel_path = os.path.relpath(file_abs_path, absolute_project_dir).replace(os.sep, '/')

                if not cfg.should_ignore(file_rel_path):
                    python_files_to_parse.append(file_abs_path)
                else:
                     logger.debug(f"Ignoring file: {file_rel_path}")

    logger.info(f"Found {len(python_files_to_parse)} Python files to parse after applying ignore rules.")
    if not python_files_to_parse:
        logger.warning(f"No Python files found or all files were ignored in {absolute_project_dir}")
        return {}, {}

    # --- Parse Files ---
    parsed_count = 0
    for py_file_abs in python_files_to_parse:
        result = parse_python_file(py_file_abs, absolute_project_dir)
        if result:
            relative_filepath, parsed_nodes, file_imports = result
            all_nodes_info[relative_filepath] = parsed_nodes
            all_imports_info[relative_filepath] = file_imports
            parsed_count += 1

    logger.info(f"Finished parsing. Successfully parsed {parsed_count}/{len(python_files_to_parse)} files.")
    return all_nodes_info, all_imports_info

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Create dummy files and ignore patterns for testing ---
    TEST_DIR = "_test_parser_project"
    IGNORE_FILE = os.path.join(TEST_DIR, ".myignore")
    os.makedirs(os.path.join(TEST_DIR, "src"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, "venv"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, "data", "cache"), exist_ok=True)

    with open(os.path.join(TEST_DIR, "src", "main.py"), "w") as f:
        f.write("import helper\n\ndef run():\n  helper.do_work()\n")
    with open(os.path.join(TEST_DIR, "src", "helper.py"), "w") as f:
        f.write("def do_work():\n  print('Working')\n")
    with open(os.path.join(TEST_DIR, "venv", "lib.py"), "w") as f:
        f.write("# Should be ignored\n")
    with open(os.path.join(TEST_DIR, "data", "cache", "temp.py"), "w") as f:
        f.write("# Should be ignored by data/\n")
    with open(os.path.join(TEST_DIR, "config.py"), "w") as f:
        f.write("# Should be ignored by name\n")

    # Create ignore file
    with open(IGNORE_FILE, "w") as f:
        f.write("# Ignore virtual environment\n")
        f.write("venv/\n")
        f.write("\n") # Empty line
        f.write("# Ignore data directory\n")
        f.write("data/\n")
        f.write("# Ignore specific files by name\n")
        f.write("config.py\n")
        f.write("*.log\n") # Wildcard example

    # --- Override config for testing ---
    # Temporarily point config loader to our test ignore file
    original_ignore_file = cfg.IGNORE_PATTERNS_FILE
    original_ignore_patterns = cfg.IGNORE_PATTERNS
    cfg.IGNORE_PATTERNS_FILE = IGNORE_FILE
    cfg.IGNORE_PATTERNS = cfg.load_ignore_patterns(IGNORE_FILE)
    logger.info(f"--- Running Parser Test with Ignore File: {IGNORE_FILE} ---")
    logger.info(f"--- Test Ignore Patterns: {cfg.IGNORE_PATTERNS} ---")

    # --- Run Parsing ---
    nodes_data, imports_data = parse_project(TEST_DIR)

    # --- Print Summary ---
    print("\n--- Parsing Summary (Test) ---")
    print(f"Total files parsed: {len(nodes_data)}")
    for rel_path in nodes_data.keys():
        print(f"  - Parsed: {rel_path}")
    total_nodes = sum(len(nodes) for nodes in nodes_data.values())
    print(f"Total nodes (functions/classes) found: {total_nodes}")
    total_calls = sum(len(n.get('calls', [])) for file_nodes in nodes_data.values() for n in file_nodes.values())
    print(f"Total calls recorded: {total_calls}")

    # Example details
    if "src/main.py" in nodes_data:
        print("\n--- Details for src/main.py ---")
        print("Nodes:", list(nodes_data["src/main.py"].keys()))
        print("Imports:", imports_data.get("src/main.py", {}))
        if "src/main.py::run" in nodes_data["src/main.py"]:
             print("Calls in run():", nodes_data["src/main.py"]["src/main.py::run"].get('calls'))

    # --- Clean up test directory ---
    # Restore original config
    cfg.IGNORE_PATTERNS_FILE = original_ignore_file
    cfg.IGNORE_PATTERNS = original_ignore_patterns
    try:
        import shutil
        shutil.rmtree(TEST_DIR)
        logger.info(f"Cleaned up test directory: {TEST_DIR}")
    except Exception as e:
        logger.error(f"Failed to clean up test directory {TEST_DIR}: {e}")

    print("\n--- Parser Test Complete ---")

