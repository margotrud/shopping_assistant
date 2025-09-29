# color_sentiment_extractor/extraction/general/types.py
from __future__ import annotations

from typing import Protocol

class TokenLike(Protocol):
    """Does: Minimal protocol for spaCy-like tokens used in recovery modules."""
    text: str
    pos_: str

__all__ = ["TokenLike"]
