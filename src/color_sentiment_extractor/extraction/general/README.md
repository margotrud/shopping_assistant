# General

## Does
Provides general-purpose modules and shared resources used across the extraction pipeline.  
Includes utilities, vocabularies, typing protocols, and sentiment/fuzzy helpers.

## Submodules
- **utils/** : Config loading, logging, and lightweight NLP helpers.  
- **vocab/** : Curated domain vocabularies (e.g., cosmetic nouns) for filtering and enrichment.  
- **fuzzy/** : Fuzzy matching logic, scoring heuristics, and alias validation.  
- **sentiment/** : Sentence sentiment classification and color–sentiment routing.  
- **types.py** : Shared typing protocol (`TokenLike`) for spaCy-like tokens.  

## Returns
Stable, lightweight APIs imported by higher-level extraction pipelines (color logic, recovery, strategies).  
Ensures clean dependencies and consistent access to shared functionality.
