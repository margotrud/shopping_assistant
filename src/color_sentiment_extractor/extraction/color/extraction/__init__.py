# Chatbot/1.2/extraction/color/extraction/__init__.py

from .compound import (
    extract_compound_phrases,
    extract_from_adjacent,
    extract_from_glued,
    extract_from_split,
    attempt_mod_tone_pair,
)

from .standalone import (
    extract_lone_tones,
    extract_standalone_phrases,  # attention à l’orthographe
)
