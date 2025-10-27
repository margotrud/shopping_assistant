# extraction/color/token/__init__.py
"""
token package.
=============

Does: Regroupe les utilitaires de découpe de tokens couleur collés/hyphénés
      avec validation suffix-aware et récupération de base stricte.
Used By: Stratégies compound (adjacent/glued) et pipelines de résolution couleur.
Exports: split_glued_tokens, split_tokens_to_parts.
"""

from __future__ import annotations

# Public API
from .split import (
    split_glued_tokens,
    split_tokens_to_parts,
)

__all__ = (
    "split_glued_tokens",
    "split_tokens_to_parts",
)

# Optionnel : standardiser la doc pour les outils/IDE
__docformat__ = "google"
