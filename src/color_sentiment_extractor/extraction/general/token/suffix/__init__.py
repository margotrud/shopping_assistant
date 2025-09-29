# token/suffix/__init__.py

from __future__ import annotations

# Expose core builders/predicates
from .recovery import (
    build_augmented_suffix_vocab,
    is_suffix_variant,
)

# Expose registry (ordered recovery funcs + dispatcher si tu l’as ajouté)
from .registry import (
    SUFFIX_RECOVERY_FUNCS,
    # recover_with_registry,   # ← dé-commente si tu veux l’exposer aussi
)

__all__ = [
    "build_augmented_suffix_vocab",
    "is_suffix_variant",
    "SUFFIX_RECOVERY_FUNCS",
    # "recover_with_registry",
]
