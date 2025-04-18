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


def scan_file_tree(project_dir: str, ignore_patterns: List[str]) -> List[str]:
    """
    扫描项目目录，返回所有文件和目录的相对路径列表（尊重忽略规则）。
    """
    file_tree_paths = []
    absolute_project_dir = os.path.abspath(project_dir)
    logger.info(f"Scanning file tree for: {absolute_project_dir}")

    for root, dirs, files in os.walk(absolute_project_dir, topdown=True):
        # --- 过滤忽略的目录 ---
        original_dirs = list(dirs)
        dirs[:] = [] # 清空 walk 将要访问的子目录列表
        for d in original_dirs:
            dir_abs_path = os.path.join(root, d)
            dir_rel_path = os.path.relpath(dir_abs_path, absolute_project_dir).replace(os.sep, '/')
            # 检查目录名 和 目录名/ 两种模式
            if not cfg.should_ignore(dir_rel_path, ignore_patterns) and \
               not cfg.should_ignore(dir_rel_path + '/', ignore_patterns):
                dirs.append(d) # 保留未被忽略的目录
                # 将未忽略的目录本身添加到列表
                file_tree_paths.append(dir_rel_path)
            else:
                 logger.debug(f"Ignoring directory during scan: {dir_rel_path}")

        # --- 处理文件 ---
        for file in files:
            file_abs_path = os.path.join(root, file)
            file_rel_path = os.path.relpath(file_abs_path, absolute_project_dir).replace(os.sep, '/')

            if not cfg.should_ignore(file_rel_path, ignore_patterns):
                file_tree_paths.append(file_rel_path) # 添加未忽略的文件
            else:
                logger.debug(f"Ignoring file during scan: {file_rel_path}")

    logger.info(f"Found {len(file_tree_paths)} files/directories in the tree.")
    return file_tree_paths

def read_readme(project_dir: str) -> Optional[str]:
    """
    尝试读取项目根目录下的 README.md 文件。
    """
    readme_path = os.path.join(project_dir, "README.md")
    readme_content = None
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                readme_content = f.read()
            logger.info(f"Successfully read README.md from {readme_path}")
        except Exception as e:
            logger.error(f"Error reading README.md from {readme_path}: {e}")
    else:
        logger.info(f"README.md not found at {readme_path}")
    return readme_content


def parse_project(project_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Optional[str]]:
    """
    解析项目，返回 AST 节点、导入信息、文件树结构和 README 内容。

    Returns:
        Tuple[Dict, Dict, List[str], Optional[str]]:
            (all_nodes_info, all_imports_info, file_tree, readme_content)
    """
    all_nodes_info: Dict[str, Dict[str, Any]] = {}
    all_imports_info: Dict[str, Dict[str, Any]] = {}
    absolute_project_dir = os.path.abspath(project_dir)

    logger.info(f"Starting project parsing and scanning for: {absolute_project_dir}")

    # 1. 扫描文件树 (传入当前的忽略模式)
    file_tree = scan_file_tree(project_dir, cfg.IGNORE_PATTERNS)

    # 2. 读取 README
    readme_content = read_readme(project_dir)

    # 3. 查找并解析 Python 文件 (从扫描到的文件树中筛选)
    python_files_to_parse = [
        os.path.join(absolute_project_dir, rel_path)
        for rel_path in file_tree
        if rel_path.endswith('.py') and os.path.isfile(os.path.join(absolute_project_dir, rel_path)) # 确保是文件
    ]

    logger.info(f"Found {len(python_files_to_parse)} Python files to parse based on file tree scan.")
    parsed_count = 0
    for py_file_abs in python_files_to_parse:
        # 确保文件仍然存在且未被忽略 (双重检查，scan_file_tree 应该已经处理)
        file_rel_path_check = os.path.relpath(py_file_abs, absolute_project_dir).replace(os.sep, '/')
        if not cfg.should_ignore(file_rel_path_check, cfg.IGNORE_PATTERNS):
             result = parse_python_file(py_file_abs, absolute_project_dir)
             if result:
                 relative_filepath, parsed_nodes, file_imports = result
                 all_nodes_info[relative_filepath] = parsed_nodes
                 all_imports_info[relative_filepath] = file_imports
                 parsed_count += 1

    logger.info(f"Finished parsing. Successfully parsed {parsed_count}/{len(python_files_to_parse)} Python files.")
    return all_nodes_info, all_imports_info, file_tree, readme_content