# classification

**Does**: Provide utilities to derive and format strict modifier↔tone mappings  
from descriptive color phrases (spaces, hyphens, glued forms).  
Filters cosmetic nouns, enforces base recovery for modifiers, and  
returns deterministic, JSON-ready mappings.

**Exports**:  
- `build_tone_modifier_mappings` – construct bidirectional mappings (mod→tone, tone→mod)  
- `format_tone_modifier_mappings` – format mappings into sorted dicts for API use
