"""
sentiment
=========

Package for sentiment analysis and color preference routing.

Exports:
- analyze_sentence_sentiment: Split sentences into clauses and detect sentiment.
- classify_segments_by_sentiment_no_neutral: Classify segments strictly as pos/neg.
- build_color_sentiment_summary: Aggregate color phrases and RGBs per sentiment.
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
