"""Frozen-editor proposals with exact protected-pixel preservation."""
from .core import audit_arrays, project_arrays
from .masks import build_masks

__version__ = "0.1.0"
__all__ = ["audit_arrays", "project_arrays", "build_masks"]
