# LLM (color)

**Does**: Normalize descriptive color phrases and query an LLM for strict RGB tuples.  
**Returns**: `(R, G, B)` or `None` on failure.  
**Used by**: Descriptive color resolution + RGB extraction/caching pipelines.

## Quick start

```python
from color_sentiment_extractor.extraction.color.llm import (
    get_llm_client,
    has_api_key,
    query_llm_for_rgb,
)

# Check environment
assert has_api_key(), "Set OPENROUTER_API_KEY in your environment."

# 1) RGB from phrase (no client needed)
rgb = query_llm_for_rgb("dusty rose", debug=True)
print(rgb)  # e.g. (231, 180, 188)

# 2) Minimal client for phrase simplification (optional feature)
from color_sentiment_extractor.extraction.color.llm import OpenRouterClient
client = get_llm_client()
if client:
    simplified = client.simplify("very dusty pinkish rose")
    print(simplified)  # e.g. "dusty rose"
