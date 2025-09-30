"""
sentiment
=========

Package for sentiment analysis and color preference routing.

Submodules:
- core   : Clause splitting, hybrid sentiment detection, negation handling.
- router : Aggregation of color phrases & RGBs per sentiment.

Exports:
- analyze_sentence_sentiment
- classify_segments_by_sentiment_no_neutral
- build_color_sentiment_summary
"""

from .core import (
    analyze_sentence_sentiment,
    classify_segments_by_sentiment_no_neutral,
)
from .router import build_color_sentiment_summary

__all__ = [
    "analyze_sentence_sentiment",
    "classify_segments_by_sentiment_no_neutral",
    "build_color_sentiment_summary",
]

__docformat__ = "google"
