# Token Base Recovery

**Single entrypoint for recovering canonical base tokens across the project.**  
This module normalizes noisy or suffixed tokens into known **modifiers** or **tones**, using a multi-stage pipeline with strong safeguards.

## Pipeline
1. **Overrides** – Apply manual mappings from `RECOVER_BASE_OVERRIDES`.  
2. **Suffix recovery** – Try each function in `SUFFIX_RECOVERY_FUNCS` (e.g. `dusty → dust`).  
3. **Direct hits** – Check against known modifiers/tones vocabularies.  
4. **Fuzzy match** – Safe fuzzy matching with conflict & block guards.  
5. **Abbreviation fallback** – Consonant-skeleton recovery for short tokens.

## Features
- Deterministic, cached, and depth-limited resolution  
- Configurable vocabularies (`known_modifiers`, `known_tones`)  
- Guards against semantic conflicts and blocked pairs  
- Debug logging with `logging.getLogger(__name__)`  
- Public helpers: `is_known_modifier()`, `is_known_tone()`

## Public API
```python
recover_base(token: str, *, allow_fuzzy: bool = True, debug: bool = False, **kwargs) -> Optional[str]
is_known_modifier(token: str) -> bool
is_known_tone(token: str) -> bool

>>> recover_base("dusty")
'dust'

>>> recover_base("bluu", allow_fuzzy=True)
'blue'
