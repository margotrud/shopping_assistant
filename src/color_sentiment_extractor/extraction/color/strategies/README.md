# Strategies (color)

**Does**: Provide modular strategies for extracting descriptive color phrases  
from raw text, either as **compound pairs** (modifier + tone) or as **standalone terms**.

---

## Submodules

### `compound.py`
- Resolves `(modifier, tone)` pairs from adjacent, glued, or split tokens.
- Uses suffix/base recovery, semantic conflict checks, and optional LLM fallback.
- Ensures valid compounds like `"dusty rose"` or `"soft lilac"` are extracted.

### `standalone.py`
- Extracts lone tones and modifiers directly from tokens.
- Applies rule-based filtering, cosmetic-noun blocking, and expression-based injection.
- Combines strict matches with controlled LLM-assisted fallback.

---

## Public API

Exposed via [`__init__.py`](./__init__.py):

```python
from color_sentiment_extractor.extraction.color.strategies import (
    extract_compound_phrases,
    extract_from_adjacent,
    extract_from_glued,
    extract_from_split,
    attempt_mod_tone_pair,
    extract_lone_tones,
    extract_standalone_phrases,
)
