# Pipelines

This folder implements high-level orchestration logic for **color phrase extraction** and **RGB resolution**.  
It bridges low-level token utilities, suffix/base recovery, and LLM-based simplification into coherent pipelines.

---

## Files

### `phrase_pipeline.py`
**Does**:  
- Extracts descriptive color phrases from raw text (compound, standalone, lone tones).  
- Validates phrases via base recovery, fuzzy thresholds, and blocked/filtered tokens.  
- Aggregates simplified tone names and resolves aligned RGB values.  

**Returns**:  
- Deduplicated phrase lists (lowercased, stable order).  
- Tone sets and phrase→RGB dictionaries for downstream sentiment/aggregation.  

**Used By**:  
- `rgb_pipeline.py`  
- Higher-level sentiment analysis flows.  

---

### `rgb_pipeline.py`
**Does**:  
- Resolves RGB from descriptive color phrases using multi-stage strategy:  
  1. Rule-based simplification  
  2. Pre-normalization (modifier–tone alignment)  
  3. LLM fallback (optional)  
  4. DB/fuzzy fallbacks  
- Applies semantic conflict resolution to ensure consistency.  

**Returns**:  
- `(simplified_phrase, RGB or None)` for each input phrase.  
- Public helpers for DB-first and LLM-first resolution strategies.  

**Used By**:  
- Modifier–tone mapping  
- Color resolution pipelines  
- User input parsing and grounding  

---

## Design Notes
- **Deterministic**: Outputs are consistently lowercased, sorted, and deduplicated.  
- **Guardrails**: Cosmetic noun filters, semantic conflict resolution, and blocked token rules prevent false positives.  
- **Performance**: spaCy is lazy-loaded and cached; fuzzy matching prefers `rapidfuzz` with fallback to `fuzzywuzzy`.  
- **Extensibility**: LLM client is optional, injected via `Protocol`, enabling easy mocking or replacement.  

---

## Example Usage

```python
from color_sentiment_extractor.extraction.color.logic.pipelines import (
    extract_phrases_from_segment,
    process_color_phrase,
)

phrases = extract_phrases_from_segment("dusty rose tint", known_modifiers, known_tones, all_webcolor_names, expression_map)
print(phrases)  # {"dusty rose"}

simplified, rgb = process_color_phrase("dusty rose", known_modifiers, known_tones, llm_client)
print(simplified, rgb)  # "dusty rose", (197, 162, 159)
