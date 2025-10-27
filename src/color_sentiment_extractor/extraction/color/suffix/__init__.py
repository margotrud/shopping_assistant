"""
suffix package.
==============

Public API for suffix handling (generation and recovery).
This module re-exports functions from rules.py so callers can import
from `...suffix` directly without referencing submodules.
"""

from .rules import (
    _apply_reverse_override,
    _collapse_repeated_consonant,
    build_ey_variant,
    build_y_variant,
    is_cvc_ending,
    is_y_suffix_allowed,
)

__all__ = [
    "is_y_suffix_allowed",
    "is_cvc_ending",
    "build_y_variant",
    "build_ey_variant",
    "_apply_reverse_override",
    "_collapse_repeated_consonant",
]
