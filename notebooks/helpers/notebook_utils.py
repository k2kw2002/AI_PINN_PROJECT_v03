"""Notebook utility functions.

Common setup, path management, and backend module import helpers.
"""
import sys
from pathlib import Path


def setup_project_root():
    """Add project root to sys.path for backend imports."""
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
