"""
llm
===

Public interface for LLM access used to resolve descriptive color phrases
into simplified tokens or RGB values.

Example:
    client = get_llm_client()
    rgb = query_llm_for_rgb(client, "dusty rose")
"""

from __future__ import annotations

# Public API re-exports
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

# Optional: explicit docformat for tooling consistency.
__docformat__ = "google"
