# color_sentiment_extractor/extraction/general/types.py
from __future__ import annotations

from typing import Protocol

"""
types.py.

Does: Define lightweight structural Protocols used for type hints
across extractors and pipelines.
"""


class TokenLike(Protocol):
    text: str
    pos_: str


__all__ = ["TokenLike"]

__docformat__ = "google"
