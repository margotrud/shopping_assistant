# extraction/general/token/__init__.py
"""
token
=====

Does: Provide base token utilities for normalization, base recovery, and splitting.
Exports: normalize_token, singularize, get_tokens_and_counts,
         recover_base, recursive_token_split, fallback_split_on_longest_substring
Used by: General tokenization and color/expression extraction pipelines.
"""

from __future__ import annotations

# Public, légers, stables
from .normalize import (
    normalize_token,
    singularize,
    get_tokens_and_counts,
)
from .base_recovery import (
    recover_base,
)
from .split.split_core import (
    recursive_token_split,
    fallback_split_on_longest_substring,
)

# Optionnel: import interne pour tests/caching, non exporté
try:
    from .base_recovery import _recover_base_cached_with_params as _recover_base_cached_with_params  # noqa: F401
except Exception:  # pragma: no cover
    pass

__all__ = [
    # normalize
    "normalize_token",
    "singularize",
    "get_tokens_and_counts",
    # recovery
    "recover_base",
    # split
    "recursive_token_split",
    "fallback_split_on_longest_substring",
]
