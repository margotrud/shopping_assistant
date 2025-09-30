# Data

## Does
Contains static JSON resources that drive the extraction pipeline.  
These files define known vocabularies, expression rules, and product metadata used across color and sentiment analysis.

## Files
- **expression_context_rules.json** : Contextual promotion and suppression rules for expressions.  
- **expression_definition.json** : Canonical expression definitions (aliases, triggers, mapped tones).  
- **known_modifiers.json** : Curated list of descriptive color modifiers (e.g., *dusty, shiny*).  
- **product_metadata.json** : Metadata for product attributes linked to color descriptors.  
- **products.json** : Example product dataset with descriptive color phrases.

## Returns
Provides stable, versioned configuration inputs for pipelines:  
- Expression matching and alias validation  
- Modifier + tone recovery  
- Product-level color grounding
