# Color Sentiment Extractor 🎨🧠

<p align="left">
  <!-- CI badge -->
  <a href="https://github.com/margotrud/shopping_assistant/actions/workflows/ci.yml">
    <img src="https://github.com/margotrud/shopping_assistant/actions/workflows/ci.yml/badge.svg" alt="CI status">
  </a>
  <!-- Coverage (Codecov) -->
  <a href="https://codecov.io/gh/margotrud/shopping_assistant">
    <img src="https://img.shields.io/codecov/c/github/margotrud/shopping_assistant" alt="Codecov">
  </a>
  <!-- Python versions -->
  <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue" alt="Python versions">
  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

_Imagine a shopping assistant that understands how you feel about colors._  
_You tell it: “I love red lipstick, but not too shiny or too purple — I prefer deeper, muted shades like raspberry or bordeaux.”_  
_The system then interprets your preferences and suggests matching products available in stock._  

💄 This project builds the **foundation for that vision**:  
it extracts **color names, modifiers, and sentiments** from text,  
and maps them to **RGB tones** to understand user aesthetics at scale.

> 🧩 **Note**  
> This repository is the **first functional module** of a larger *Shopping Assistant* NLP project.  
> It focuses on **color sentiment extraction**, a core capability that will later integrate into  
> product description analysis, style classification, and emotional search pipelines.

Given text like _“I love bright red but I hate purple”_,  
the extractor identifies positive and negative color tones and resolves them to RGB.

---

## 🚀 Quick Start

```bash
# 1) Install in editable mode
pip install -e .

# 2) Get spaCy model
python -m spacy download en_core_web_sm

# 3) Run demo
cse-demo "I love bright red but I hate purple"
```

---

## 📊 Demo Output

Input:
```text
"I love bright red but I hate purple"
```

Output:
```json
{
  "positive": [
    {"name": "bright red", "rgb": [255, 0, 13]},
    {"name": "red", "rgb": [255, 0, 0]},
    {"name": "crimson", "rgb": [220, 20, 60]},
    {"name": "orangered", "rgb": [255, 69, 0]},
    {"name": "firebrick", "rgb": [178, 34, 34]},
    {"name": "brown", "rgb": [165, 42, 42]},
    {"name": "tomato", "rgb": [255, 99, 71]},
    {"name": "chocolate", "rgb": [210, 105, 30]},
    {"name": "darkred", "rgb": [139, 0, 0]},
    {"name": "maroon", "rgb": [128, 0, 0]},
    {"name": "sienna", "rgb": [160, 82, 45]}
  ],
  "negative": [
    {"name": "purple", "rgb": [128, 0, 128]},
    {"name": "darkmagenta", "rgb": [139, 0, 139]},
    {"name": "indigo", "rgb": [75, 0, 130]},
    {"name": "mediumvioletred", "rgb": [199, 21, 133]},
    {"name": "darkslateblue", "rgb": [72, 61, 139]},
    {"name": "darkviolet", "rgb": [148, 0, 211]},
    {"name": "darkorchid", "rgb": [153, 50, 204]},
    {"name": "midnightblue", "rgb": [25, 25, 112]},
    {"name": "blueviolet", "rgb": [138, 43, 226]},
    {"name": "dimgray", "rgb": [105, 105, 105]}
  ]
}
```
### 🖼️ Visual Demo

![Color Sentiment Demo](docs/demo_palette.png)

---

## ✨ Highlights

- **End-to-end NLP pipeline** combining rule-based parsing, fuzzy logic, and suffix recovery.  
- **Data-driven color reasoning**: maps language to RGB through hybrid lookup (CSS/XKCD + LLM).  
- **Typed, tested, and CI-validated**: full pytest suite, mypy typing, and GitHub Actions integration.  
- **Optimized architecture** with modular extractors, caching (LRU), and time-budget safeguards.  
- **Config-driven vocabulary system** for tones/modifiers with realistic linguistic coverage.  
- **Clean packaging** (`src/` layout, editable install, coverage + badges, MIT License).  

---

## 🗂️ Project Structure

```
color-sentiment-extractor/
├── src/
│   └── color_sentiment_extractor/    # Core extraction logic (color + general)
├── tests/                            # Pytest suite
├── README.md                         # Documentation
├── LICENSE                           # MIT License
├── pyproject.toml                    # Build system + metadata
├── .pre-commit-config.yaml           # Linting & formatting hooks
├── .coveragerc                       # Coverage configuration
└── pytest.ini                        # Pytest configuration
```


---
## 🧠 Tech Stack

- **Language:** Python 3.10–3.12  
- **Core NLP:** spaCy, NLTK  
- **Fuzzy Matching:** fuzzywuzzy / rapidfuzz  
- **LLM Fallback:** OpenRouter / transformers  
- **Testing & Quality:** pytest, mypy, Ruff, GitHub Actions, Codecov

## ⚙️ How It Works

1. **Tokenization (spaCy)** – Sentences are parsed into tokens with POS tagging.  
2. **Negation Detection** – Splits colors into positive vs. negative context.  
3. **Color Extraction** – Identifies modifiers (“bright”, “dark”) + tones (“red”, “purple”).  
4. **Normalization & Recovery** – Handles suffixes, fuzzy matches, and known synonyms.  
5. **RGB Resolution**  
   - Primary: matches against CSS/XKCD color databases.  
   - Fallback: calls an LLM to approximate an RGB triplet.  
6. **Output** – JSON with `positive` and `negative` color lists.  

---

## ✅ Tests

```bash
pytest -q
```
---

## 🔮 Roadmap

This color sentiment extractor is only the **first step** of a broader NLP pipeline.  
Next planned modules will include:
- **Product description parsing** (extracting attributes like fabric, shape, and fit),
- **Style and tone classification** (e.g., “minimalist”, “romantic”, “sporty”),
- **User sentiment aggregation** for brand perception and trend analysis.

Each stage builds on the same design principles — **typed, tested, modular, and interpretable**.

---

## 📜 License

MIT License – free to use, modify, and share.  
© 2025 Margot Rudnianski


---

## 💬 Author

**Margot Rudnianski**  
📫 [LinkedIn](https://www.linkedin.com/in/margotrudnianski) · [GitHub](https://github.com/margotrud)
