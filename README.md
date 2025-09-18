# Color Sentiment Extractor 🎨🧠

<p align="left">
  <!-- CI badge -->
  <a href="https://github.com/margotrud/shopping_assistant/actions">
    <img src="https://github.com/margotrud/shopping_assistant/actions/workflows/tests.yml/badge.svg" alt="CI status">
  </a>
  <!-- Coverage (Codecov) -->
  <a href="https://codecov.io/gh/margotrud/shopping_assistant">
    <img src="https://codecov.io/gh/margotrud/shopping_assistant/branch/main/graph/badge.svg" alt="Codecov">
  </a>
  <!-- Python versions -->
  <img src="https://img.shields.io/badge/Python-3.10%20|%203.11-blue" alt="Python versions">
  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>


Given text like _“I love bright red but I hate purple”_, this project:
- extracts **color mentions** (single tones and compound phrases),
- splits them by **positive vs negative** sentiment,
- resolves each color name to **RGB** (CSS/XKCD first, LLM fallback).

<p>
  <a href="https://github.com/margotrud/shopping_assistant/actions">
    <img src="https://github.com/margotrud/shopping_assistant/actions/workflows/tests.yml/badge.svg" alt="CI status">
  </a>
</p>

---

## Quick Start

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Get spaCy model
python -m spacy download en_core_web_sm

# 3) Run demo
python demo.py "I love bright red but I hate purple"



## 📊 Demo Output (example)

Input:
```text
"I love bright red but I hate purple"

Output:
{
  "positif": [
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
  "negatif": [
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


🗂️ Project Structure
shopping_assistant_V6/
├── Chatbot/                   # Core extraction logic
│   ├── extraction/            # NLP + color pipelines
│   │   ├── color/             # Color-specific logic
│   │   └── general/           # General token/recovery utils
│   ├── __init__.py
│   └── orchestrator.py        # Main orchestrator
├── tests/                     # Pytest-based test suite
│   └── test_smoke.py
├── demo.py                    # Quick demo script
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Ignore cache, venv, etc.
├── .coveragerc                # Test coverage configuration
└── pytest.ini                 # Pytest configuration


## ⚙️ How it Works (High Level)

1. **Tokenization (SpaCy)** – Sentences are parsed into tokens with POS tagging.
2. **Negation Detection** – Finds positive vs. negative context.
3. **Color Extraction** – Identifies modifiers ("bright", "dark") + tones ("red", "purple").
4. **Normalization & Recovery** – Handles suffixes, fuzzy matches, and known synonyms.
5. **RGB Resolution**
   - First: matches against CSS/XKCD color databases.
   - Fallback: calls LLM to approximate an RGB triplet.
6. **Output** – JSON with `positif` and `negatif` colors.


✅ Tests
Run all tests with:
pytest -q

📜 License
MIT License – free to use, modify, and share.

