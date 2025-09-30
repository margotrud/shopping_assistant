# suffix (general/token)

**Does**: Provide suffix-based vocabulary builders, recovery functions, and a registry  
for dispatching the correct recovery strategy based on token suffix.

**Exports**:  
- `build_augmented_suffix_vocab` – build a suffix-augmented vocabulary with -y/-ey/-ed variants  
- `is_suffix_variant` – predicate to check if a token is a suffix variant of a known base  
- `recover_*` – helpers for recovering bases from specific suffix forms (-y, -ed, -ing, -ier, etc.)  
- `SUFFIX_RECOVERY_FUNCS` – ordered tuple of all recovery functions  
- `recover_with_registry` – (optional) suffix-aware dispatcher using `_SUFFIX_ORDER`

**Used by**:  
- Base-recovery flows  
- Compound and standalone extractors  
- Tokenization and suffix vocab pipelines
