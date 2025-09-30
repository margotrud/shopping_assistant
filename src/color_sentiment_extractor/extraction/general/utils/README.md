# Utils (general)

**Does**  
Provide lightweight utility functions for configuration loading, debug logging, and minimal NLP helpers.

**Modules**  
- `load_config.py` – Locate and load JSON configs with caching, type coercions (`raw` | `set` | `validated_dict`), env overrides, and optional validation.  
- `log.py` – Minimal debug logger controlled by `CHATBOT_DEBUG_TOPICS` (comma-separated or `"all"`). Prints timestamped messages with topic and level.  
- `nlp_utils.py` – Minimal NLP helpers: antonym detection via WordNet and token lemmatization via spaCy, with caching and fallbacks.  

**Used by**  
- Core extraction pipelines  
- Vocabulary loaders  
- Sentiment routing & tests needing hot reload or negation handling  

**Public API** (import via package)  
```python
from Chatbot.extraction.general.utils import (
    load_config, clear_config_cache,
    DataDirNotFound, ConfigFileNotFound, ConfigParseError, ConfigTypeError,
    debug, reload_topics,
    are_antonyms, lemmatize_token,
)
