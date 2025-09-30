# Color

## Does
Core package for color phrase extraction, normalization, and RGB resolution.  
Implements strategies, recovery flows, suffix rules, and utilities that connect descriptive tokens to standardized color names and values.

## Submodules
- **llm/** : Query language models to resolve descriptive phrases into RGB values.  
- **logic/** : Orchestration pipelines for RGB resolution and tone–modifier mappings.  
- **recovery/** : Recovery strategies for noisy or ambiguous tokens (suffix, fuzzy, LLM).  
- **strategies/** : Extraction strategies (compound phrases, standalone tones).  
- **suffix/** : Rules and helpers for suffix generation and base recovery.  
- **token/** : Token utilities (normalization, splitting, base recovery).  
- **utils/** : Color-specific helpers (RGB distance, representative choice, etc.).  
- **constants.py** : Domain constants used across color extraction.  
- **vocab.py** : Curated vocabularies of tones, modifiers, and webcolor names.  
- **__init__.py** : Public API surface for color extraction components.

## Returns
Provides stable building blocks for higher-level flows:  
- Phrase → (modifier, tone) parsing  
- Token → base form recovery  
- Phrase → RGB resolution (CSS/XKCD + LLM fallback)  

These modules are imported by orchestrators and sentiment pipelines to support end-to-end color sentiment extraction.
