# recovery

Recovery strategies for noisy or ambiguous tokens in color phrases.

## Does
Provides multiple resolution strategies to normalize user input:
- Suffix/root equivalence checks
- Modifier token resolution (direct, suffix, compound, fuzzy)
- LLM-aided simplification with safety filters

## Submodules
- **fuzzy_recovery**  
  Detects suffix/root variants (`rosy` ↔ `rose`, `beigey` ↔ `beige`)  
  while blocking semantic and rhyming conflicts.

- **modifier_resolution**  
  Resolves descriptive modifiers via direct match, suffix fallback,  
  singularization, compound parsing, and conflict filters.

- **llm_recovery**  
  Uses an LLM to simplify tokens when rules are insufficient.  
  Preserves valid surface forms (e.g. `dusty rose`) and bans unsafe standalone tones.

## Public API
From `color.recovery` you can import:
```python
from color_sentiment_extractor.extraction.color import recovery

recovery.is_suffix_root_match(...)
recovery.resolve_modifier_token(...)
recovery.match_direct_modifier(...)
recovery.match_suffix_fallback(...)
recovery.recover_y_with_fallback(...)
recovery.simplify_phrase_if_needed(...)
recovery.simplify_color_description_with_llm(...)
recovery.is_blocked_modifier_tone_pair(...)
recovery.is_known_tone(...)
