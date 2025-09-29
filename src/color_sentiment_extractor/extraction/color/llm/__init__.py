"""
llm
===

Does: Expose the public LLM interface: client factory, API-key check, and RGB querying.
Returns: Re-exports of stable symbols from `llm_api_client` for external use.
Used by: Descriptive color resolution, RGB pipelines, and higher-level orchestrators.
Example:
    client = get_llm_client(); rgb = query_llm_for_rgb("dusty rose")
"""

from __future__ import annotations

# ── Public API re-exports ─────────────────────────────────────────────────────
from .llm_api_client import (
    OpenRouterClient,
    get_llm_client,
    has_api_key,
    query_llm_for_rgb,
)

__all__ = [
    "OpenRouterClient",
    "get_llm_client",
    "has_api_key",
    "query_llm_for_rgb",
]

# Keep docformat explicit for tooling consistency.
__docformat__ = "google"
