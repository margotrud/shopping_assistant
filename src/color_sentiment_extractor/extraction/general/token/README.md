
# token (general)

**Does**: Provide core token utilities for normalization, base recovery, suffix handling, and splitting.  
This package unifies low-level text processing for noisy, glued, or suffixed tokens.  

**Submodules**:  
- `normalize` – safe singularization, cosmetic-noun handling, and normalization rules  
- `base_recovery` – map noisy/suffixed tokens to canonical bases with fuzzy/abbr fallbacks  
- `split` – deglue glued tokens using greedy, backtracking, and budgeted strategies  
- `suffix` – suffix vocab builder, recover_* functions, and suffix-aware dispatcher  

**Used by**: General tokenization and color/expression extraction pipelines.
