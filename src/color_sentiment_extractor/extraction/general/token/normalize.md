Token Normalizer
================

Shared normalizer for the extraction pipeline.  
Cleans tokens/phrases, handles hyphens/underscores consistently, and applies domain-aware singularization (cosmetic nouns only).  
Designed to be deterministic and fast; no I/O, no side effects.

Features
--------

* ✨ **Deterministic normalization**: lowercase, trim, collapse spaces; `_` → space; configurable hyphen handling.
* 🎯 **Domain-aware singulars**: only the **last** word is singularized and only if it’s a cosmetic noun; protects invariants (`series`, `species`).
* 🔗 **Integration**: shared by color/brand extractors; lives under `extraction/general/token/`.
* 📊 **Counting helper**: regex tokenization + normalization to produce `{token: count}`.

Functions
---------

* `singularize(text: str) -> str`  
  Back-compat API: singularizes a single word, or only the **last** word in a phrase, using safe rules (`ies→y`, `…es` after `sh/ch/x/z/ss`, else `…s→…`).

* `normalize_token(token: str, keep_hyphens: bool = False) -> str`  
  Normalizes (`_`→space; hyphens tightened or spaced) then singularizes the last cosmetic noun if present; returns a clean token/phrase.

* `get_tokens_and_counts(text: str, keep_hyphens: bool = False) -> dict[str, int]`  
  Regex-tokenizes text, normalizes each token (optionally preserving hyphens), and aggregates frequencies into a dictionary.

* `_singularize_word(w: str) -> str`  
  Internal helper implementing the safe singular rules; used by both `singularize` and `normalize_token`.

* `_singularize_phrase_if_cosmetic_last(text: str) -> str`  
  Internal: singularizes the last token **only if** it’s a cosmetic noun from project vocab.

Example Usage
-------------

```python
from extraction.general.token.normalize import singularize, normalize_token, get_tokens_and_counts

# Singularization (safe + back-compat)
print(singularize("blushes"))          # "blush"
print(singularize("soft pinks"))       # "soft pink"
print(singularize("limited series"))   # "limited series"

# Normalization
print(normalize_token("soft-pinks"))                       # "soft pink"
print(normalize_token("rose - gold", keep_hyphens=True))   # "rose-gold"
print(normalize_token("rose_gold"))                        # "rose gold"

# Counting
text = "Soft-pinks, nudes! NUDES."
print(get_tokens_and_counts(text))     # {"soft": 1, "pink": 1, "nude": 2}
