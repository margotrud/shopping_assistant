# token/__init__.py

from __future__ import annotations

# ✅ Public, légers, stables
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

# (Optionnel) on importe la fonction cachee pour tests internes, mais on ne l’exporte pas
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
    # ❌ on n’exporte PAS la fonction privée de cache
]
