# extraction/general/token/split/__init__.py
"""
split
=====

Does: Expose public token-splitting utilities for glued-token recovery.
Exports: has_token_overlap, fallback_split_on_longest_substring, recursive_token_split
Used by: General tokenization and color/expression extraction pipelines.
"""

from .split_core import (
    has_token_overlap,
    fallback_split_on_longest_substring,
    recursive_token_split,
)

__all__ = [
    "has_token_overlap",
    "fallback_split_on_longest_substring",
    "recursive_token_split",
]
