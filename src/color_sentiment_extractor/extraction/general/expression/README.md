# Expression Module

## Overview
The `expression` package provides utilities to handle **expression-driven
color extraction**.  
It supports:
- Building trigger vocabularies from expression definitions (aliases + modifiers)  
- Matching expressions with exact and fuzzy rules  
- Mapping matched expressions to valid tones  
- Injecting modifiers from raw tokens with base recovery and conflict resolution  
- Applying context and suppression rules  
- Supplying a glued-token vocabulary (tones + modifiers + webcolors) for token splitting  

## Structure
- **`__init__.py`**  
  Curates the public API: exposes expression matching, tone mapping, trigger vocab helpers,
  context/suppression rules, and glued-token vocabulary.  
  Keeps internals hidden, except `inject_expression_modifiers` which is optionally re-exported.

- **`expression_helpers.py`**  
  Core implementation:  
  - Cached config loaders (`_get_known_modifiers`, `_get_known_tones`, …)  
  - Trigger token builders (`get_all_trigger_tokens`, `get_all_alias_tokens`)  
  - Expression matching (`extract_exact_alias_tokens`, `get_matching_expression_tags_cached`)  
  - Tone mapping (`map_expressions_to_tones`)  
  - Modifiers injection (`_inject_expression_modifiers`) with suffix & semantic conflict handling  
  - Contextual rules (`apply_expression_context_rules`, `apply_expression_suppression_rules`)  
  - Glued token vocabulary (`get_glued_token_vocabulary`)  

## Public API
The package exports the following helpers:

```python
from color_sentiment_extractor.extraction.general.expression import (
    # Matching & mapping
    map_expressions_to_tones,
    get_matching_expression_tags_cached,

    # Trigger vocab
    get_all_trigger_tokens,
    get_all_alias_tokens,
    extract_exact_alias_tokens,

    # Rules
    apply_expression_context_rules,
    apply_expression_suppression_rules,

    # Glued-token vocab
    get_glued_token_vocabulary,

    # Optional (advanced usage)
    inject_expression_modifiers,
)
