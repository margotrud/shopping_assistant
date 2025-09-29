"""
strategies
==========

Color extraction strategies:
- compound  : resolve modifier + tone phrases
- standalone: extract lone tones or standalone phrases
"""

from .compound import (
    extract_compound_phrases,
    extract_from_adjacent,
    extract_from_glued,
    extract_from_split,
    attempt_mod_tone_pair,
)
from .standalone import (
    extract_lone_tones,
    extract_standalone_phrases,
)

__all__ = [
    # compound
    "extract_compound_phrases",
    "extract_from_adjacent",
    "extract_from_glued",
    "extract_from_split",
    "attempt_mod_tone_pair",
    # standalone
    "extract_lone_tones",
    "extract_standalone_phrases",
]

__docformat__ = "google"
