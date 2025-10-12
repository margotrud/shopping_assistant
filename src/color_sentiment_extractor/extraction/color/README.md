# 🎨 Color Extraction Module

**Does:** Extract, normalize, and resolve color phrases (tones + modifiers) from text,  
mapping them to RGB values through deterministic rules and LLM-based fallbacks.  
**Used by:** `orchestrator.py` and sentiment routing pipelines.

---

## 🧩 Submodules Overview

| Submodule | Purpose |
|------------|----------|
| `llm/` | Queries an LLM to normalize and resolve descriptive color phrases into RGB tuples. |
| `logic/classification/` | Builds strict tone↔modifier mappings; filters cosmetic nouns and invalid pairs. |
| `logic/pipelines/` | Orchestrates phrase extraction and RGB resolution (rules → LLM → DB/fuzzy). |
| `recovery/` | Recovers canonical color tokens via suffix, fuzzy, or LLM-assisted methods. |
| `strategies/` | Extracts compound (`dusty rose`) and standalone (`rose`) color phrases. |
| `suffix/` | Handles morphological variants like `dusty`, `pinkish`, `bronzey`, etc. |
| `token/` | Splits glued tokens, normalizes forms, applies suffix-aware recovery. |
| `utils/` | Shared helpers: RGB math, distance metrics, fuzzy name matching. |
| `constants.py` | Domain constants (blocked tokens, tone/modifier lists). |
| `vocab.py` | Loads and validates curated color vocabularies. |

---

## ⚙️ Processing Flow

1. **Token normalization** – lowercase, strip, recover base forms.  
2. **Suffix & fuzzy recovery** – detect variants like `rosy → rose`.  
3. **Compound extraction** – combine modifiers and tones.  
4. **Phrase validation** – filter cosmetic nouns, blocked pairs, conflicts.  
5. **RGB resolution** – lookup CSS/XKCD or query the LLM fallback.  
6. **Aggregation** – merge results into consistent JSON-ready outputs.

---

## 🧠 Example

```python
from color_sentiment_extractor.extraction.color.logic.pipelines import rgb_pipeline

rgb = rgb_pipeline.resolve("dusty rose")
print(rgb)  # → (231, 180, 188)
