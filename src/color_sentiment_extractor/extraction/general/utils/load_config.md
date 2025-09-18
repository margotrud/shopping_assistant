# Config Loader Utility

This module provides a robust way to locate and load JSON configuration files for the project.
It is designed for reliability, testability, and CI/CD friendliness, with clear exceptions and caching.

## Features
- 🔍 **Automatic data directory discovery**: walks upward to find the nearest `data/` or `Data/` folder, or uses the `DATA_DIR` environment variable / `base_dir` override.
- 📦 **Three load modes**:
  - **raw** → returns parsed JSON as-is.
  - **set** → expects a list, returns a `frozenset[str]`.
  - **validated_dict** → expects a dict, optionally passed through a user validator function.
- 🛡️ **Dedicated error types**:
  - `DataDirNotFound`, `ConfigFileNotFound`, `ConfigParseError`, `ConfigTypeError`.
- ⚡ **In-memory caching**: avoids re-reading files, with `clear_config_cache()` to bust cache in tests or hot reload.
- 🧪 **Test/CI ready**: easy override of config path via `DATA_DIR` environment variable.

## Functions
- `_default_data_dir()` → Locate nearest `data/` folder, prefer lowercase, return absolute `Path` or raise `DataDirNotFound`.
- `load_config()` → Load `<data>/<file>.json` in the requested mode (`raw`/`set`/`validated_dict`), support env/base_dir overrides, validator, caching, and raise dedicated errors.
- `clear_config_cache()` → Empty the internal cache (useful for tests and reloads).

## Example Usage

```python
from Chatbot.extraction.general.utils.load_config import load_config, clear_config_cache

# Load raw JSON (any structure)
cfg = load_config("settings", mode="raw")

# Load a list of strings as frozenset
tags = load_config("known_tags", mode="set")

# Load and validate a dict
def validate_schema(d: dict) -> dict:
    if "version" not in d:
        raise ValueError("Missing version field")
    return d

config = load_config("pipeline", mode="validated_dict", validator=validate_schema)

# Clear cache if configs change on disk
clear_config_cache()
