# suffix (color)

**Does**: Provide utilities for suffix handling in descriptive color tokens.  
Includes heuristics and overrides for generating or recovering `-y` and `-ey` variants,  
CVC doubling detection, reverse overrides, and consonant-collapsing.

**Exports**:
- `is_y_suffix_allowed` – check if a base can accept `-y`
- `is_cvc_ending` – detect consonant–vowel–consonant endings
- `build_y_variant` – generate a valid `-y` form
- `build_ey_variant` – generate a valid `-ey` form
- `_apply_reverse_override` – map suffixed tokens back to override bases
- `_collapse_repeated_consonant` – normalize doubled consonants if valid

**Used by**:
- Suffix vocabulary builders
- Base recovery flows
- Compound & standalone extractors
