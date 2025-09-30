# sentiment

**Does**: Provides sentiment analysis utilities and color preference routing.

## Submodules
- **core.py**  
  Hybrid sentiment detection combining:
  - VADER lexical scoring (fast, interpretable)
  - BART MNLI fallback (robust zero-shot classification)  
  Also includes clause splitting, negation handling (negspaCy + soft-negation), and high-level APIs:
  - `analyze_sentence_sentiment`  
  - `classify_segments_by_sentiment_no_neutral`

- **router.py**  
  Aggregates extracted color phrases and RGB mappings per sentiment.  
  Provides:
  - `build_color_sentiment_summary` → returns representative phrases, base RGB, and adaptive thresholds.

## Public API
- `analyze_sentence_sentiment(sentence: str) -> list[ClauseResult]`  
- `classify_segments_by_sentiment_no_neutral(segments: list[str]) -> dict[str, list[str]]`  
- `build_color_sentiment_summary(...) -> ColorSentimentSummary`

## Notes
- Uses lazy-loading for heavy components (spaCy, VADER, BART).  
- Thresholds and feature switches are configurable via environment variables.  
- Designed for testability (fake pipeline, dependency injection).  
