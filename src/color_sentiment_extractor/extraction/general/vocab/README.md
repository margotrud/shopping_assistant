# Vocab (general)

## Does
Defines domain-specific vocabularies used across the extraction pipeline.  
These vocabularies provide curated token sets for filtering, blocking, and semantic enrichment.  

## Submodules
- **cosmetic_nouns.py** : Minimal set of cosmetic product nouns (e.g., *lipstick, blush, mascara*)  
  Used to block cosmetic terms from being falsely recognized as valid tones.

## Returns
Exports simple, typed constants (`set[str]`, `list[str]`, etc.) ready for direct lookup or membership checks.  
They are imported by higher-level extraction strategies and recovery logic.
