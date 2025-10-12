# 🧰 General Extraction Module

**Does:** Provide shared building blocks for the extraction pipeline:  
token utilities (normalize/recover/split), fuzzy matching, expression mapping,  
sentiment analysis, vocabularies, and lightweight NLP/config helpers.  
**Used by:** Modules in `extraction/color/` and the global orchestrator.

---

## 🧩 Submodules Overview

| Submodule | Purpose |
|---|---|
| `expression/` | Builds and matches expression triggers (aliases + modifiers), maps expressions → tones, applies context/suppression rules. |
| `fuzzy/` | Robust fuzzy layer: matching, custom scoring, conflict and negation guards, cached alias matching. |
| `sentiment/` | Sentence-level sentiment detection (VADER + optional MNLI fallback), clause routing, and preference grouping. |
| `token/` | Core token utilities: normalization, suffix recovery registry, and glued-token splitting. |
| `vocab/` | Curated domain vocabularies (e.g., cosmetic nouns) for filtering and enrichment. |
| `utils/` | Config loader (cached), debug logger, minimal NLP helpers. |
| `types.py` | Shared typing protocols (e.g., `TokenLike`) used across modules. |

---

## ⚙️ Processing Flow (high level)

1. **Normalize tokens** → lowercase, strip, standardize form.  
2. **Suffix/base recovery** → detect and restore variants (`-y`, `-ish`, etc.).  
3. **Fuzzy & expression mapping** → match aliases, inject modifiers, apply conflict guards.  
4. **Split glued tokens** → recursive and budgeted strategies for ambiguous tokens.  
5. **Sentiment classification** → analyze polarity and group segments for color extraction.

---

## 🧠 Example

```python
from color_sentiment_extractor.extraction.general.sentiment.router import build_color_sentiment_summary

text = "I love dusty rose but I hate dark purple."
summary = build_color_sentiment_summary([text])
print(summary)
# → {'positif': ['dusty rose'], 'negatif': ['dark purple']}
