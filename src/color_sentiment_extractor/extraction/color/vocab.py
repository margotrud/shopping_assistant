# Chatbot/extraction/color/vocab.py

"""
vocab.py
========

Defines color-specific vocabularies used in color extraction and normalization.

Includes:
- CSS3 and CSS2.1 color names
- XKCD color names
- Cosmetic fallback tones not covered in web standards
"""

import webcolors
from matplotlib.colors import XKCD_COLORS

# CSS/XKCD raw sources
css3 = set(webcolors.CSS3_NAMES_TO_HEX.keys())
css21 = set(webcolors.CSS21_NAMES_TO_HEX.keys())
xkcd = set(name.replace("xkcd:", "") for name in XKCD_COLORS.keys())

# Manually added tones missing from standards
cosmetic_fallbacks = {"nude", "ash", "ink", "almond", "champagne"}

# Full tone set used for color tone matching
known_tones = set(name.lower() for name in css3 | css21 | xkcd | cosmetic_fallbacks)

# CSS-only names (no XKCD or fallback) for strict matching
all_webcolor_names = set(name.lower() for name in css3 | css21)

