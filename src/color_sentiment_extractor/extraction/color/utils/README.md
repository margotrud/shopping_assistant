# utils

**Does**: Provide general-purpose utilities shared across the extraction pipeline.  
Includes normalization helpers, RGB distance calculations, named color lookups, and parsing utilities.

**Submodules**:
- `rgb_distance.py` : Functions for computing color distances (sRGB/Lab), clustering representative RGBs, fuzzy/named color matching, and parsing RGB values.

**Used By**:
- Color similarity and clustering pipelines  
- Recovery and matching logic (nearest/fuzzy name resolution)  
- Sentiment and classification layers needing representative RGBs
