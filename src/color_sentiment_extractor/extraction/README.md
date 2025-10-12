# 🧩 Extraction Package

**Does:** Central hub for all extraction logic — defines pipelines, strategies, vocabularies, and shared utilities  
used to parse, normalize, and resolve descriptive color phrases from text.  
**Used by:** `color_sentiment_extractor.demo` and `color_sentiment_extractor.orchestrator`.

---

## 📂 Submodules Overview

| Submodule | Purpose |
|------------|----------|
| `color/` | Color-specific components: suffix rules, vocabularies, recovery, extraction strategies, logic pipelines, and LLM integration. |
| `general/` | Shared modules: token normalization, fuzzy matching, sentiment routing, vocabularies, and config/NLP utilities. |
| `__init__.py` | Compatibility shim for legacy imports (`from extraction.x import y`). |
| `orchestrator.py` | High-level orchestrator combining sentiment analysis, color extraction, and RGB resolution into a unified workflow. |

---

## ⚙️ Processing Flow (overview)

1. **General preprocessing** – normalize text, recover token bases, handle suffixes.  
2. **Expression & fuzzy mapping** – identify descriptive modifiers and tones.  
3. **Color extraction** – parse compound and standalone color phrases.  
4. **RGB resolution** – match CSS/XKCD colors or query LLM fallback.  
5. **Sentiment routing** – group positive vs. negative color preferences.  

---

## 🧠 Example

```python
from color_sentiment_extractor.extraction.orchestrator import analyze_text

result = analyze_text("I love bright red but hate dark purple.")
print(result)
# → {'positif': [{'name': 'bright red', 'rgb': [255, 0, 13]}],
#    'negatif': [{'name': 'dark purple', 'rgb': [128, 0, 128]}]}
