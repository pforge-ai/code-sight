# src/utils/hashing.py
# -*- coding: utf-8 -*-

import hashlib
import os
import logging

logger = logging.getLogger(__name__)

def get_project_hash(project_path: str, length: int = 12) -> str:
    """
    Generates a unique hash string based on the absolute project path.

    This hash is used to create unique directory names for storing
    project-specific data (like RAG indexes and caches).

    Args:
        project_path (str): The path to the project directory.
        length (int): The desired length of the output hash string.
                      Defaults to 12, which is usually sufficient to avoid
                      collisions for typical numbers of projects.

    Returns:
        str: A hexadecimal hash string of the specified length.
    """
    try:
        # Ensure the path is absolute and normalized for consistency
        absolute_path = os.path.abspath(project_path)
        # Encode the path to bytes for hashing
        path_bytes = absolute_path.encode('utf-8')
        # Create a SHA-256 hash object
        hasher = hashlib.sha256()
        # Update the hash object with the path bytes
        hasher.update(path_bytes)
        # Get the full hexadecimal digest
        full_hash = hasher.hexdigest()
        # Truncate the hash to the desired length
        truncated_hash = full_hash[:length]
        logger.debug(f"Generated hash '{truncated_hash}' for project path '{project_path}' (absolute: '{absolute_path}')")
        return truncated_hash
    except Exception as e:
        logger.error(f"Error generating hash for project path '{project_path}': {e}", exc_info=True)
        # Fallback to a simple replacement if hashing fails, though unlikely
        # This fallback is very basic and might lead to collisions easily.
        fallback_hash = "".join(c if c.isalnum() else '_' for c in os.path.basename(project_path))[:length]
        logger.warning(f"Falling back to basic hash '{fallback_hash}' due to error.")
        return fallback_hash

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    test_paths = [
        ".",
        "/Users/cuiyubao/dev/pforge/orelm",
        "/Users/cuiyubao/dev/pforge/orelm/", # Should produce same hash as above
        "../another_project",
        "C:\\Users\\Test\\Documents\\ProjectA",
    ]

    print("--- Testing get_project_hash ---")
    for path in test_paths:
        project_hash = get_project_hash(path)
        print(f"Path: '{path}' -> Hash: '{project_hash}'")

    # Test length parameter
    print("\n--- Testing different hash lengths ---")
    path_for_length_test = "/path/to/my/code"
    print(f"Path: '{path_for_length_test}'")
    for l in [8, 12, 16, 32]:
        h = get_project_hash(path_for_length_test, length=l)
        print(f"  Length {l}: '{h}'")

