# Color Sentiment Extractor 🎨🧠

Given text like _"I love bright red but I hate purple"_, this extracts color mentions,
splits **positive vs negative**, and maps them to **RGB** (CSS/XKCD first, LLM fallback).

## Quick Start
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python demo.py "I love bright red but I hate purple"
