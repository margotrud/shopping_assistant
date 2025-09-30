# Color Sentiment Extractor

## Does
Analyzes descriptive color phrases, recovers modifier + tone pairs, and resolves them into standardized RGB values.  
Supports sentiment-aware extraction (positive/negative preferences) and robust handling of noisy tokens.

## Structure
- **data/** : Static resources (known modifiers, tones, mappings, config files).  
- **extraction/** : Core extraction logic.  
  - **color/** : Color-specific modules (LLM integration, pipelines, strategies, recovery, suffix rules, vocab, utils).  
  - **general/** : Shared helpers (token utils, fuzzy matching, vocab, sentiment, config loaders).  
  - **orchestrator.py** : High-level orchestration combining sentiment, pipelines, and RGB resolution.  
- **demo.py** : Example entry point for running color sentiment analysis interactively.  
- **__init__.py** : Package initializer.

## Returns
Provides stable APIs to:
- Parse phrases into modifier + tone combinations.  
- Normalize and recover noisy tokens.  
- Resolve phrases into RGB values (CSS/XKCD palette + LLM fallback).  
- Aggregate results by sentiment (positive vs. negative preferences).  

Used by higher-level applications in product search, preference modeling, and UI color grounding.
