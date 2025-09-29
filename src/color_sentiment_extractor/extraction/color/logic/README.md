
---

## Modules

### `classification/`
**Does**: Derives strict **modifier↔tone** mappings from descriptive phrases (spaces, hyphens, glued). Enforces base recovery and filters cosmetic nouns.  
**Returns**: Deterministic, JSON-ready mappings and formatted dicts for API use.  
**Exports**:
- `build_tone_modifier_mappings(phrases, known_tones, known_modifiers)` → `(tones, modifiers, mod_to_tone, tone_to_mod)`
- `format_tone_modifier_mappings(phrases, known_tones, known_modifiers)` → `{"modifiers": {...}, "tones": {...}}`

### `pipelines/`
**Does**: High-level orchestration for **phrase extraction** and **RGB resolution** (rules → pre-normalization → optional LLM → DB/fuzzy fallbacks).  
**Returns**: Validated phrases, simplified tones, and `phrase → RGB` dictionaries.  
**Exports**:
- `extract_all_descriptive_color_phrases(text, ...)` → `list[str]`
- `extract_phrases_from_segment(segment, ...)` → `set[str]`
- `process_segment_colors(color_phrases, ...)` → `(list[str], list[RGB|None])`
- `aggregate_color_phrase_results(segments, ...)` → `(set[str], list[str], dict[str, RGB])`
- `resolve_rgb_with_llm(phrase, ...)` / `get_rgb_from_descriptive_color_llm_first(...)`
- `process_color_phrase(phrase, ...)` → `(simplified_phrase, RGB|None)`

---

## Data Flow (overview)

```mermaid
flowchart LR
  A[Raw text] --> B[phrase_pipeline: extract/validate]
  B --> C[process_segment_colors: simplify + RGB attempt]
  C --> D[aggregate_color_phrase_results: tone set + map]
  B --> E[classification: build mappings]
  E --> F[format mappings]
  C --> G[rgb_pipeline: resolve_rgb_with_llm (rules→LLM→DB/fuzzy)]
