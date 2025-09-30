# color_sentiment_extractor/extraction/general/types.py
from __future__ import annotations

"""
types.py
========

Does: Define shared typing protocols for NLP tokens to avoid direct dependency on spaCy internals.
Returns: TokenLike → minimal interface with `.text` and `.pos_` attributes.
Used by: Recovery modules, suffix handling, and fuzzy matching utilities.
"""

from typing import Protocol

class TokenLike(Protocol):
    text: str
    pos_: str

__all__ = ["TokenLike"]

__docformat__ = "google"
