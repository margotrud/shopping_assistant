"""
classification
Does: Expose public API for building and formatting tone↔modifier mappings.
Exports: build_tone_modifier_mappings, format_tone_modifier_mappings.
"""

from __future__ import annotations

# ── Public API re-exports ─────────────────────────────────────────────────────
from .categorizer import (
    build_tone_modifier_mappings,
    format_tone_modifier_mappings,
)

__all__ = [
    "build_tone_modifier_mappings",
    "format_tone_modifier_mappings",
]
