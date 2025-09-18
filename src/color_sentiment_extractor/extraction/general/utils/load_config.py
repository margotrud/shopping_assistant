# Chatbot/extraction/general/utils/load_config.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Union, Literal, Optional, Any, Callable, FrozenSet

# --- Exceptions dédiées ---
class DataDirNotFound(FileNotFoundError): ...
class ConfigFileNotFound(FileNotFoundError): ...
class ConfigParseError(ValueError): ...
class ConfigTypeError(TypeError): ...

_CONFIG_CACHE: dict[tuple[Path, str, str, bool], object] = {}
Mode = Literal["raw", "set", "validated_dict"]

def clear_config_cache() -> None:
    """Does: Empties the in-memory config cache (useful for essaitests2/hot-reload). Returns: None."""
    _CONFIG_CACHE.clear()

def _default_data_dir(start: Optional[Path] = None) -> Path:
    """
    Does: Walk upward from start (or this file) to find 'data' folder, preferring 'data' over 'Data' at the same level; returns absolute path or raises DataDirNotFound.
    Params: start optional starting path.
    Returns: Absolute Path to the data directory.
    Raises: DataDirNotFound.
    """
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        for name in ("data", "Data"):
            cand = (p / name).resolve()
            if cand.is_dir():
                return cand
    raise DataDirNotFound(f"No 'data' directory found from {start}")

def load_config(
    file: str,
    mode: Mode = "raw",
    *,
    base_dir: Optional[Path] = None,
    encoding: str = "utf-8",
    validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Union[FrozenSet[str], Dict[str, Any], list, object]:
    """
    Does: Load <data>/<file>.json, parse JSON, coerce by mode ('raw' passthrough; 'set' expects list[str]→frozenset[str]; 'validated_dict' expects dict and optional validator), and cache by (path, mode).
    Args: file (with or without .json), mode ∈ {'raw','set','validated_dict'}, optional base_dir/encoding/validator.
    Returns: Parsed JSON, frozenset[str], or dict[str,Any].
    Raises: DataDirNotFound, ConfigFileNotFound, ConfigParseError, ConfigTypeError, ValueError.
    """
    # ENV override (utile en CI): si DATA_DIR est défini et pas d’override explicite
    if base_dir is None:
        env_dir = os.environ.get("DATA_DIR")
        if env_dir:
            base_dir = Path(env_dir)

    data_dir = (base_dir or _default_data_dir()).resolve()

    file_name = file if file.endswith(".json") else f"{file}.json"
    path = (data_dir / file_name).resolve()
    if not path.is_file():
        raise ConfigFileNotFound(f"Config file not found: {path}")

    # Le cache tient compte de l'encodage et de la présence d'un validator (bool)
    cache_key = (path, mode, encoding, validator is not None)
    if cache_key in _CONFIG_CACHE and validator is None:
        # On ne réutilise PAS un résultat validé car le validator peut changer la sortie.
        return _CONFIG_CACHE[cache_key]

    try:
        with path.open("r", encoding=encoding, newline="") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigParseError(f"Invalid JSON in {path}: {e}") from e
    except OSError as e:
        raise ConfigFileNotFound(f"Cannot read {path}: {e}") from e

    if mode == "raw":
        result: object = data
    elif mode == "set":
        if not isinstance(data, list):
            raise ConfigTypeError(f"{path.name}: expected list for mode 'set', got {type(data).__name__}")
        bad = [x for x in data if not isinstance(x, (str, int, float, bool)) and x is not None]
        if bad:
            raise ConfigTypeError(f"{path.name}: list must contain scalar values only for 'set'; found non-scalars.")
        result = frozenset(map(str, data))
    elif mode == "validated_dict":
        if not isinstance(data, dict):
            raise ConfigTypeError(f"{path.name}: expected dict for mode 'validated_dict', got {type(data).__name__}")
        if validator is not None:
            try:
                data = validator(data)  # may raise
            except Exception as e:
                raise ConfigParseError(f"{path.name}: validator failed: {e}") from e
        result = data
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # On ne met pas en cache si un validator est fourni (sa logique peut varier)
    if validator is None:
        _CONFIG_CACHE[cache_key] = result
    return result
