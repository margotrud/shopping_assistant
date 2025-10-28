"""
types.py.

Does: Define the single canonical LLM client protocol for the color pipeline.
Used by: phrase_pipeline, rgb_pipeline, llm_recovery.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    Canonical structural contract for any LLM client in the color pipeline.

    The client must implement:
    - simplify(phrase): Normalize a noisy/descriptive color phrase into a short,
      canonical form (≤2 words) like "dusty rose" or "warm beige".
      Returns "" if undecided.
    - query(text): Handle a free-form prompt (used for RGB extraction logic).
      Returns a response string, or None if no answer.
    """

    def simplify(self, phrase: str) -> str: ...
    def query(self, text: str) -> str | None: ...
