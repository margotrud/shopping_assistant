# NLP Utils

Minimal natural language helpers for the extraction pipeline.  
Provides antonym detection via WordNet and token lemmatization via spaCy, with caching for performance.  
All tokens are normalized through the project-wide `normalize_token` before analysis to ensure consistency.

## Features
- 🔍 **Antonym checking**: fast, cached lookups with WordNet, normalized to project standards.
- ⚡ **Caching**: `@lru_cache` on both antonym and lemmatization helpers (10k / 20k entries).
- 🛡️ **Robustness**: safe fallbacks when WordNet data or spaCy models are unavailable.
- 🔗 **Integration**: uses the shared token normalizer from `general/token/normalize.py`.

## Functions
- `_normalized_antonyms(word: str) -> set[str]`  
  Internal helper that collects and normalizes antonyms of a word from WordNet; returns empty set if unavailable.
- `are_antonyms(word1: str, word2: str) -> bool`  
  Public API: checks if two words are antonyms by comparing against each other’s normalized antonym set; returns False safely if inputs are empty or WordNet resources are missing.
- `_get_spacy() -> spaCy.Language | None`  
  Loads and caches spaCy `en_core_web_sm` with unnecessary components disabled; returns None if spaCy or the model is unavailable.
- `lemmatize_token(token: str) -> str`  
  Lemmatizes a token via spaCy if available, otherwise returns the normalized token unchanged; cached for 20,000 entries.

## Example Usage

```python
from Chatbot.extraction.general.utils import nlp_utils

# Antonym checking
print(nlp_utils.are_antonyms("hot", "cold"))   # True (if WordNet data is available)
print(nlp_utils.are_antonyms("big", "small"))  # True
print(nlp_utils.are_antonyms("blue", "green")) # False

# Lemmatization
print(nlp_utils.lemmatize_token("running"))    # "run" (if spaCy en_core_web_sm is installed)
print(nlp_utils.lemmatize_token("dogs"))       # "dog"
print(nlp_utils.lemmatize_token("happy"))      # "happy" (unchanged if spaCy not available)
