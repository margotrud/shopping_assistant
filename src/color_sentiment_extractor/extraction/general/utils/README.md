# Utils Package

This package provides general-purpose utilities that support the extraction pipeline.  
It is designed for reusability, clarity, and robustness, with each submodule focusing on a single responsibility.

## Modules

### [`load_config.py`](load_config.py)
- **Purpose**: Load JSON configuration files from the project’s `data/` directory.  
- **Features**:  
  - Locate nearest `data/` folder (or override with `DATA_DIR` env).  
  - Load JSON in different modes:  
    - `raw` → return parsed JSON as-is.  
    - `set` → expects list, returns `frozenset[str]`.  
    - `validated_dict` → expects dict, optional validator function.  
  - Safe caching with `clear_config_cache()`.  
  - Dedicated error classes (`ConfigFileNotFound`, `ConfigParseError`, etc.).

### [`log.py`](log.py)
- **Purpose**: Minimal debug logger controlled by environment variables.  
- **Features**:  
  - Enable topics via `CHATBOT_DEBUG_TOPICS` (`all`, or comma-separated topics).  
  - Timestamped lines with topic and log level.  
  - Reload topics dynamically with `reload_topics()`.  
  - Output to `stderr` by default, or custom stream.

### [`nlp_utils.py`](nlp_utils.py)
- **Purpose**: Lightweight NLP helpers (token-level).  
- **Features**:  
  - `are_antonyms()` → checks if two words are antonyms using WordNet (with normalization and caching).  
  - `lemmatize_token()` → lemmatizes a token with spaCy (`en_core_web_sm` if available), otherwise falls back safely.  
  - Internal helpers cached for performance; safe fallbacks if WordNet or spaCy are unavailable.

## Public API

The `__init__.py` re-exports the main entry points for convenience:  

```python
from Chatbot.extraction.general.utils import (
    load_config, clear_config_cache,
    debug, reload_topics,
    DataDirNotFound, ConfigFileNotFound, ConfigParseError, ConfigTypeError,
)


## Example: Using all utils together

```python
from Chatbot.extraction.general.utils import (
    load_config, clear_config_cache,
    debug, reload_topics,
)
from Chatbot.extraction.general.utils import nlp_utils

# 1. Load a configuration file
try:
    colors = load_config("known_colors", mode="set")
    debug(f"Loaded {len(colors)} colors from config", topic="config")
except Exception as e:
    debug(f"Failed to load config: {e}", topic="config", level="ERROR")

# 2. Use NLP helpers
word1, word2 = "hot", "cold"
if nlp_utils.are_antonyms(word1, word2):
    debug(f"{word1} and {word2} are antonyms", topic="nlp")

lemma = nlp_utils.lemmatize_token("running")
debug(f"Lemmatized 'running' → {lemma}", topic="nlp")

# 3. Dynamically change logging topics
import os
os.environ["CHATBOT_DEBUG_TOPICS"] = "all"
reload_topics()
debug("Now logging everything", topic="anywhere")
