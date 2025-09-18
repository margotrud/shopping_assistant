# Chatbot/extraction/general/utils/log.py
import os, sys
from datetime import datetime
from typing import Set, TextIO, Optional

def _load_topics() -> Set[str]:
    raw = os.getenv("CHATBOT_DEBUG_TOPICS", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}

_DEBUG_TOPICS = _load_topics()

def reload_topics() -> None:
    global _DEBUG_TOPICS
    _DEBUG_TOPICS = _load_topics()

def debug(
    msg: str,
    topic: str = "extraction",
    *,
    level: str = "DEBUG",
    stream: Optional[TextIO] = None,   # ⬅️ change ici
) -> None:
    """
    Does: Print a timestamped debug line with topic and level if enabled via CHATBOT_DEBUG_TOPICS ("topic1,topic2" or "all").
    """
    if stream is None:                  # ⬅️ resolve au moment de l'appel (compatible pytest)
        stream = sys.stderr
    if not _DEBUG_TOPICS or "all" in _DEBUG_TOPICS or topic.lower() in _DEBUG_TOPICS:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{topic}][{level}] {msg}", file=stream)
