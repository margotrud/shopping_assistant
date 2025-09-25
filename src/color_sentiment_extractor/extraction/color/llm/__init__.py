"""
llm
===

Client interface and helpers for querying LLMs to resolve descriptive
color phrases into simplified tokens or RGB values.
"""

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
