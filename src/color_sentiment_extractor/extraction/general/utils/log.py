"""
log.py.

Does: Lightweight debug logger controlled by CHATBOT_DEBUG_TOPICS (comma-sep or 'all').
Returns: Prints timestamped lines with topic + level. Used across extraction/utils/tests.
"""

import os
import sys
from datetime import datetime
from typing import TextIO

__all__ = ["debug", "reload_topics"]


def _load_topics() -> set[str]:
    raw = os.getenv("CHATBOT_DEBUG_TOPICS", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


_DEBUG_TOPICS = _load_topics()


def reload_topics() -> None:
    """Does: Reload topics from environment variable CHATBOT_DEBUG_TOPICS."""
    global _DEBUG_TOPICS
    _DEBUG_TOPICS = _load_topics()


def debug(
    msg: str,
    topic: str = "extraction",
    *,
    level: str = "DEBUG",
    stream: TextIO | None = None,
) -> None:
    """Does: Print a timestamped debug line with topic and level
    if enabled via CHATBOT_DEBUG_TOPICS.
    """
    if stream is None:
        stream = sys.stderr
    topic_key = topic.lower().strip()
    if not _DEBUG_TOPICS or "all" in _DEBUG_TOPICS or topic_key in _DEBUG_TOPICS:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{topic_key}][{level.upper()}] {msg}", file=stream)
