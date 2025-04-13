# src/utils/config_loader.py
# -*- coding: utf-8 -*-

import os
import logging
from dotenv import load_dotenv
import fnmatch # For matching ignore patterns
from typing import Tuple, Optional

# Load environment variables from .env file
# It's good practice to load this early
load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL_STR = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_STR, logging.INFO)

# Configure root logger
# Note: Streamlit might override this, but it's good for standalone script runs
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to: {LOG_LEVEL_STR}")

# --- LLM Provider Configuration ---
# Load base URLs and specific model names/IDs for each provider
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
OLLAMA_GENERATION_MODEL = os.getenv('OLLAMA_GENERATION_MODEL', 'qwen2.5-coder:7b')
OLLAMA_SUMMARIZATION_MODEL = os.getenv('OLLAMA_SUMMARIZATION_MODEL', OLLAMA_GENERATION_MODEL) # Default to generation model
OLLAMA_RELATIONSHIP_MODEL = os.getenv('OLLAMA_RELATIONSHIP_MODEL', OLLAMA_GENERATION_MODEL) # Default to generation model

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
# DEEPSEEK_EMBEDDING_MODEL = os.getenv('DEEPSEEK_EMBEDDING_MODEL') # Not currently used
DEEPSEEK_CHAT_MODEL = os.getenv('DEEPSEEK_CHAT_MODEL', 'deepseek-chat')

# --- Task-Specific Model Identifiers ---
# Load the identifier string from .env, providing defaults using Ollama models
EMBEDDING_MODEL_IDENTIFIER = os.getenv('EMBEDDING_MODEL_IDENTIFIER', f'ollama::{OLLAMA_EMBEDDING_MODEL}')
GENERATION_MODEL_IDENTIFIER = os.getenv('GENERATION_MODEL_IDENTIFIER', f'ollama::{OLLAMA_GENERATION_MODEL}')
SUMMARIZATION_MODEL_IDENTIFIER = os.getenv('SUMMARIZATION_MODEL_IDENTIFIER', f'ollama::{OLLAMA_SUMMARIZATION_MODEL}')
RELATIONSHIP_MODEL_IDENTIFIER = os.getenv('RELATIONSHIP_MODEL_IDENTIFIER', f'ollama::{OLLAMA_RELATIONSHIP_MODEL}')

# --- Helper function to parse identifiers ---
def parse_model_identifier(identifier: str, default_provider: str = 'ollama', default_model: str = '') -> Tuple[str, str]:
    """
    Parses a model identifier string "provider::model_name".

    Args:
        identifier (str): The identifier string.
        default_provider (str): Provider to return if parsing fails.
        default_model (str): Model name to return if parsing fails.

    Returns:
        Tuple[str, str]: (provider, model_name)
    """
    if identifier and '::' in identifier:
        parts = identifier.split('::', 1)
        provider = parts[0].lower()
        model_name = parts[1]
        if provider and model_name:
             # Basic validation for known providers
             if provider not in ['ollama', 'deepseek']:
                  logger.warning(f"Unknown provider '{provider}' in identifier '{identifier}'. Falling back to default.")
                  return default_provider, default_model
             return provider, model_name
        else:
             logger.warning(f"Invalid identifier format '{identifier}'. Using default.")
             return default_provider, default_model
    else:
        logger.warning(f"Could not parse identifier '{identifier}'. Using default.")
        return default_provider, default_model

# Example usage (optional, for clarity or direct use elsewhere)
# EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME = parse_model_identifier(EMBEDDING_MODEL_IDENTIFIER, default_model=OLLAMA_EMBEDDING_MODEL)
# GENERATION_PROVIDER, GENERATION_MODEL_NAME = parse_model_identifier(GENERATION_MODEL_IDENTIFIER, default_model=OLLAMA_GENERATION_MODEL)
# etc.

# --- Embedding Configuration ---
try:
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '768'))
except ValueError:
    logger.warning("Invalid EMBEDDING_DIM in .env, using default 768.")
    EMBEDDING_DIM = 768

# --- Preprocessing Configuration ---
try:
    PREPROCESSING_MAX_WORKERS = int(os.getenv('PREPROCESSING_MAX_WORKERS', '4'))
except ValueError:
    logger.warning("Invalid PREPROCESSING_MAX_WORKERS in .env, using default 4.")
    PREPROCESSING_MAX_WORKERS = 4

# --- Data Storage ---
# Use absolute path for data root to avoid relative path issues
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Assumes this file is in src/utils
DATA_ROOT_DIR_RELATIVE = os.getenv('DATA_ROOT_DIR', './data')
DATA_ROOT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, DATA_ROOT_DIR_RELATIVE))
# Ensure data directory exists
os.makedirs(DATA_ROOT_DIR, exist_ok=True)
logger.info(f"Data root directory: {DATA_ROOT_DIR}")


# --- Ignore Patterns ---
IGNORE_PATTERNS_FILE_RELATIVE = os.getenv('IGNORE_PATTERNS_FILE', './config/ignore.patterns')
IGNORE_PATTERNS_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, IGNORE_PATTERNS_FILE_RELATIVE))
IGNORE_PATTERNS = []

def load_ignore_patterns(filepath: str = IGNORE_PATTERNS_FILE) -> list[str]:
    """
    Loads ignore patterns from a file (similar to .gitignore).
    Filters out empty lines and comments (#).

    Args:
        filepath (str): The path to the ignore patterns file.

    Returns:
        list[str]: A list of patterns.
    """
    patterns = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        patterns.append(stripped_line)
            logger.info(f"Loaded {len(patterns)} ignore patterns from {filepath}")
        except Exception as e:
            logger.error(f"Error loading ignore patterns from {filepath}: {e}", exc_info=True)
    else:
        logger.info(f"Ignore patterns file not found at {filepath}. No patterns loaded.")
    return patterns

# Load patterns on module import
IGNORE_PATTERNS = load_ignore_patterns()

def should_ignore(relative_path: str, patterns: list[str] = IGNORE_PATTERNS) -> bool:
    """
    Checks if a given relative path matches any of the ignore patterns.
    Uses fnmatch for Unix shell-style wildcards.

    Args:
        relative_path (str): The relative path to check (e.g., 'src/utils/helper.py', 'data/temp').
        patterns (list[str]): The list of patterns to check against.

    Returns:
        bool: True if the path should be ignored, False otherwise.
    """
    # Normalize path separators for consistent matching
    normalized_path = relative_path.replace(os.sep, '/')
    for pattern in patterns:
        # Check if the path itself matches
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Check if any parent directory matches (for directory patterns like 'node_modules/')
        # Ensure pattern ends with '/' to specifically match directories
        if pattern.endswith('/'):
             # Check if path starts with the directory pattern
             # Need to be careful: 'src/' should match 'src/file.py'
             # Check if the path starts with the pattern OR pattern + '/'
             # Or if a full directory name matches exactly
             if normalized_path.startswith(pattern) or normalized_path == pattern.rstrip('/'):
                  return True
             # Check intermediate directories
             path_parts = normalized_path.split('/')
             for i in range(1, len(path_parts)): # Check directory paths like 'a/', 'a/b/'
                 dir_path = "/".join(path_parts[:i]) + "/"
                 if fnmatch.fnmatch(dir_path, pattern):
                      return True

    return False


# --- Log loaded configuration for verification (Optional, use DEBUG level) ---
logger.debug("--- Configuration Loaded ---")
logger.debug(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
logger.debug(f"OLLAMA_EMBEDDING_MODEL: {OLLAMA_EMBEDDING_MODEL}")
logger.debug(f"OLLAMA_GENERATION_MODEL: {OLLAMA_GENERATION_MODEL}")
logger.debug(f"DEEPSEEK_API_KEY: {'Set' if DEEPSEEK_API_KEY else 'Not Set'}")
logger.debug(f"DEEPSEEK_CHAT_MODEL: {DEEPSEEK_CHAT_MODEL}")
logger.debug(f"EMBEDDING_MODEL_IDENTIFIER: {EMBEDDING_MODEL_IDENTIFIER}")
logger.debug(f"GENERATION_MODEL_IDENTIFIER: {GENERATION_MODEL_IDENTIFIER}")
logger.debug(f"SUMMARIZATION_MODEL_IDENTIFIER: {SUMMARIZATION_MODEL_IDENTIFIER}")
logger.debug(f"RELATIONSHIP_MODEL_IDENTIFIER: {RELATIONSHIP_MODEL_IDENTIFIER}")
logger.debug(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
logger.debug(f"PREPROCESSING_MAX_WORKERS: {PREPROCESSING_MAX_WORKERS}")
logger.debug(f"DATA_ROOT_DIR: {DATA_ROOT_DIR}")
logger.debug(f"IGNORE_PATTERNS_FILE: {IGNORE_PATTERNS_FILE}")
logger.debug(f"IGNORE_PATTERNS: {IGNORE_PATTERNS}")
logger.debug("--------------------------")

# --- Validation (Updated) ---
# Check API key requirements based on selected providers in identifiers
required_keys = set()
all_identifiers = {
    "Embedding": EMBEDDING_MODEL_IDENTIFIER,
    "Generation": GENERATION_MODEL_IDENTIFIER,
    "Summarization": SUMMARIZATION_MODEL_IDENTIFIER,
    "Relationship": RELATIONSHIP_MODEL_IDENTIFIER,
}

for task, identifier in all_identifiers.items():
    provider, model_name = parse_model_identifier(identifier) # Use our parser
    if provider == 'deepseek':
        required_keys.add('DEEPSEEK_API_KEY')
    # Add checks for other providers requiring keys here if needed

if 'DEEPSEEK_API_KEY' in required_keys and not DEEPSEEK_API_KEY:
    logger.error("A DeepSeek model is specified for at least one task, but DEEPSEEK_API_KEY is not set in .env. Operations using DeepSeek will likely fail.")

# Validate embedding provider selection (since DeepSeek doesn't have one)
emb_provider, emb_model = parse_model_identifier(EMBEDDING_MODEL_IDENTIFIER)
if emb_provider == 'deepseek':
     logger.error(f"EMBEDDING_MODEL_IDENTIFIER is set to use DeepSeek ('{EMBEDDING_MODEL_IDENTIFIER}'), but DeepSeek does not provide a public embedding API. Please configure a different provider (e.g., 'ollama::{OLLAMA_EMBEDDING_MODEL}').")


