# fuzzy (general)

**Does**: Facade + utilities for robust fuzzy matching across the project: core token matching, scoring, conflicts, alias validation, and expression matching.  
**Returns**: Stable public API to score, compare, and resolve aliases/expressions with context-aware guards.  
**Used by**: Color/expression extraction pipelines, token recovery flows, and higher-level matching logic.

---

## Module map

- **`fuzzy_core.py`** – Core engine (exact/strong match, safe best-match vs vocab).
- **`scoring.py`** – Hybrid scorer (ratio/partial + prefix bonus − rhyme/length penalties) and token-list overlap.
- **`conflict_rules.py`** – Negation detection (`no/without/sans/pas de …`) + single-token morphological embedding.
- **`alias_validation.py`** – Single/multi-word alias validation with suffix/root equivalence and conflict guards.
- **`expression_match.py`** – Expression alias + modifiers matching (longest-wins, de-dup, embedded-conflict cleanup).
- **`__init__.py`** – Thin facade re-exporting the stable API.

---

## Public API (import via `general.fuzzy`)

### Core
- `is_exact_match(a, b)`  
  **Does**: Normalize + strict compare with high-ratio fallback. **Returns**: `bool`.
- `is_strong_fuzzy_match(a, b, threshold=82, *, conflict_groups=None, negation_check=True)`  
  **Does**: Fuzzy match with conflict/negation guards. **Returns**: `bool`.
- `fuzzy_match_token_safe(raw_token, known_tokens, threshold=82, debug=False)`  
  **Does**: Safe best-match (exact/edit/normalized/base → fuzzy). **Returns**: `str|None`.
- `fuzzy_token_match(a, b)`  
  **Does**: 100 for exact/derivational/edit-like else project score. **Returns**: `float`.
- `collapse_duplicates(s)` / `is_single_transposition(a, b)` / `is_single_substitution(a, b)`  
  **Does**: Micro-edit detectors. **Returns**: `str|bool`.

### Scoring
- `fuzzy_token_score(a, b)`  
  **Does**: Hybrid ratio/partial + prefix bonus − rhyme/length penalties. **Returns**: `float`.
- `rhyming_conflict(a, b)`  
  **Does**: Detect rhyme-only similarity for short tokens. **Returns**: `bool`.
- `fuzzy_token_overlap_count(a_tokens, b_tokens, threshold=85)`  
  **Does**: Count unique overlaps across lists (exact/fuzzy). **Returns**: `int`.

### Conflicts
- `is_negation_conflict(a, b)`  
  **Does**: Detect “no/without/sans/pas de …” vs base token. **Returns**: `bool`.

### Aliases & Expressions
- `_handle_multiword_alias(alias, input_text, debug=False)` *(internal helper)*  
  **Does**: Delegate multiword acceptance (partial/reorder/glue). **Returns**: `bool`.
- `is_valid_singleword_alias(alias, input_text, tokens, matched_aliases, debug=False)`  
  **Does**: Validate single token against context with guards. **Returns**: `bool`.
- `cached_match_expression_aliases(input_text)`  
  **Does**: Cached expression matching from config. **Returns**: `set[str]`.
- `match_expression_aliases(input_text, expression_map, debug=False)`  
  **Does**: Aliases (longest-first) → modifiers (scored) → conflict cleanup. **Returns**: `set[str]`.

---

## Quick usage

```python
from color_sentiment_extractor.extraction.general.fuzzy import (
    is_exact_match, is_strong_fuzzy_match, fuzzy_match_token_safe,
    fuzzy_token_score, rhyming_conflict, fuzzy_token_overlap_count,
    cached_match_expression_aliases,
)

# Token scoring & checks
assert is_exact_match("Rose-Gold", "rosegold")
score = fuzzy_token_score("dusty", "dusky")  # e.g. 86
is_ok = is_strong_fuzzy_match("bluu", "blue", threshold=82)

# Safe best-match against a vocabulary
best = fuzzy_match_token_safe("rosegoold", {"rose gold","gold","rose"}, threshold=82)

# Expression aliases (from config)
hits = cached_match_expression_aliases("soft rose gold glow")
