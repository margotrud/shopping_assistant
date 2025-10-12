# 🎨 Color Sentiment Extractor

**Does:** Analyze descriptive color phrases, recover `(modifier, tone)` pairs, and resolve them into standardized RGB values.  
Supports **sentiment-aware extraction** (positive vs. negative) and **robust recovery** for noisy, glued, or suffixed tokens.

---

## 🧩 Structure Overview

| Path | Description |
|------|--------------|
| `data/` | Static resources: known modifiers, tones, mappings, and configuration files. |
| `extraction/` | Core logic and pipelines. |
| ├── `color/` | Color-specific modules: LLM integration, pipelines, recovery, suffix rules, vocab, and utils. |
| ├── `general/` | Shared helpers: token utilities, fuzzy matching, vocab, sentiment, and config loaders. |
| └── `orchestrator.py` | High-level controller combining sentiment, color pipelines, and RGB resolution. |
| `demo.py` | Example entry point for interactive color sentiment analysis. |
| `__init__.py` | Package initializer. |

---

## ⚙️ Processing Flow

1. **Tokenization & normalization** – clean and standardize tokens.  
2. **Modifier + tone pairing** – extract valid descriptive color phrases.  
3. **Recovery & resolution** – map noisy tokens to canonical forms and RGBs (CSS/XKCD + LLM fallback).  
4. **Sentiment routing** – classify color mentions by sentiment polarity.  
5. **Aggregation** – return structured color sentiment summaries.

---

## 🧠 Example

```python
from color_sentiment_extractor.extraction.orchestrator import analyze_text

result = analyze_text("I love dusty rose but I hate dark purple.")
print(result)
# → {
#   "positif": [{"name": "dusty rose", "rgb": [231, 180, 188]}],
#   "negatif": [{"name": "dark purple", "rgb": [128, 0, 128]}]
# }
