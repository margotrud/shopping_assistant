# split (general/token)

**Does**: Provide utilities to split glued or ambiguous tokens into valid sub-parts.  
**Exports**:  
- `has_token_overlap`: quick overlap check between normalized phrases  
- `fallback_split_on_longest_substring`: budgeted splitter with greedy, backtracking, and substring fallback  
- `recursive_token_split`: recursive binary/prefix/suffix splitter using a validator  

**Used by**: Tokenization and color/expression extraction pipelines that need to recover valid token boundaries.
