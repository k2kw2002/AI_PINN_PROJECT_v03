"""Notebook utility functions.

Common setup, path management, and backend module import helpers.
"""
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml marker.

    Works regardless of Jupyter CWD setting (workspace root or notebook dir).
    """
    path = Path.cwd().resolve()
    for _ in range(10):  # max 10 levels up
        if (path / "pyproject.toml").exists():
            return path
        if path == path.parent:
            break
        path = path.parent
    raise FileNotFoundError(
        "Could not find project root (pyproject.toml). "
        "Make sure you're running from within the project."
    )


def setup_project_root() -> Path:
    """Find project root and add to sys.path for backend imports."""
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
