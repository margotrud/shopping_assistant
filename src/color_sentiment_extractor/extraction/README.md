# Extraction

## Does
Central package for all extraction logic.  
Defines pipelines, strategies, vocabularies, and general utilities for parsing and resolving descriptive color phrases.

## Submodules
- **color/** : Color-specific modules (suffix rules, vocabularies, recovery, strategies, logic pipelines).  
- **general/** : Shared helpers (token utilities, fuzzy matching, vocab, sentiment, config loaders, etc.).  
- **__init__.py** : Compatibility shim for legacy imports (`from extraction.x import y`).  
- **orchestrator.py** : High-level orchestration layer combining sentiment analysis, color pipelines, and RGB resolution.

## Returns
Provides stable APIs for higher-level consumers (UI, product search, preference modeling).  
Ensures backward compatibility via the `extraction` namespace.
