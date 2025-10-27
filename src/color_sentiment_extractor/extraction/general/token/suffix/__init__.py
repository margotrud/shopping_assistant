# extraction/general/token/suffix/__init__.py
"""
suffix.
======

Does: Provide suffix vocabulary builders and recovery functions.
Exports: build_augmented_suffix_vocab, is_suffix_variant, SUFFIX_RECOVERY_FUNCS
Used by: Suffix vocab builders, base-recovery flows, and token extraction pipelines.
"""

from __future__ import annotations

from .recovery import (
    build_augmented_suffix_vocab,
    is_suffix_variant,
)
from .registry import (
    SUFFIX_RECOVERY_FUNCS,
    # recover_with_registry,  # uncomment to expose dispatcher
)

__all__ = [
    "build_augmented_suffix_vocab",
    "is_suffix_variant",
    "SUFFIX_RECOVERY_FUNCS",
    # "recover_with_registry",
]
