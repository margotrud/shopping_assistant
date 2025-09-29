# token

**Does**: Provide token-splitting utilities for color phrase extraction.  
Includes suffix-aware validation, strict base recovery, multi-hyphen handling, time-budgeted recursion, and a guarded longest-substring fallback.

**Submodules**:
- `split.py` : Functions to split glued or hyphenated tokens into valid parts (suffix-aware vocab, cached base recovery, blocked-pair checks, multi-hyphen support, time-budget, fallback).
- `__init__.py` : Public API re-exports (`split_glued_tokens`, `split_tokens_to_parts`) and package doc.

**Used By**:
- Compound extraction strategies (adjacent/glued) to resolve `(modifier, tone)` pairs  
- Color resolution pipelines needing reliable token decomposition before phrase/RGB mapping
