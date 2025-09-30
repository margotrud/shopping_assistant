# Chatbot/extraction/general/utils/load_config.py
from __future__ import annotations

"""
load_config.py

Does: Locate <data/> and load JSON configs with caching and typed coercions ('raw'|'set'|'validated_dict').
Returns: Raw JSON, frozenset[str], or validated dict depending on mode.
Used by: Vocab loaders, extraction pipelines, tests needing hot-reload of config.
"""

from typing import (
    Any,
    Dict,
    FrozenSet,
    Literal,
    Optional,
    Callable,
    overload,
)
import json
import os
import logging
import threading
from pathlib import Path
from types import TracebackType
from typing import Type as _ExcType  # for exception annotations (internal)

# ── Public surface ────────────────────────────────────────────────────────────
Mode = Literal["raw", "set", "validated_dict"]
__all__ = [
    "Mode",
    "load_config",
    "clear_config_cache",
    "DataDirNotFound",
    "ConfigFileNotFound",
    "ConfigParseError",
    "ConfigTypeError",
]

# ── Exceptions dédiées ───────────────────────────────────────────────────────
class DataDirNotFound(FileNotFoundError):
    """Does: Raised when no 'data' directory is found while walking upwards. Returns: None."""

class ConfigFileNotFound(FileNotFoundError):
    """Does: Raised when the requested config file cannot be read or resolved. Returns: None."""

class ConfigParseError(ValueError):
    """Does: Raised when JSON parsing/validation fails for a config file. Returns: None."""

class ConfigTypeError(TypeError):
    """Does: Raised when the parsed JSON doesn't match the expected structure. Returns: None."""


# ── Logging & cache ──────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
_CACHE_LOCK = threading.RLock()
_CONFIG_CACHE: dict[tuple[Path, str, str, bool], Any] = {}


def clear_config_cache() -> None:
    """Does: Empties the in-memory config cache (useful for pytest/hot-reload). Returns: None."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        log.debug("Config cache cleared.")


def _candidate_data_dirs(start: Optional[Path] = None) -> list[Path]:
    """Does: Compute candidate 'data'/'Data' directories walking up from start. Returns: Ordered list of Paths."""
    start = (start or Path(__file__)).resolve()
    cands: list[Path] = []
    for p in [start, *start.parents]:
        for name in ("data", "Data"):
            cands.append((p / name).resolve())
    return cands


def _default_data_dir(start: Optional[Path] = None) -> Path:
    """Does: Return the first existing candidate directory among _candidate_data_dirs. Returns: Path or raises."""
    for cand in _candidate_data_dirs(start):
        if cand.is_dir():
            return cand
    raise DataDirNotFound(
        "No 'data' directory found.\n"
        f"Tried:\n  " + "\n  ".join(str(p) for p in _candidate_data_dirs(start))
    )


@overload
def load_config(
    file: str | os.PathLike[str],
    mode: Literal["raw"] = "raw",
    *,
    base_dir: Optional[Path] = None,
    encoding: str = "utf-8",
    validator: None = ...,
    allow_comments: bool = False,
) -> Any: ...
@overload
def load_config(
    file: str | os.PathLike[str],
    mode: Literal["set"] = "set",
    *,
    base_dir: Optional[Path] = None,
    encoding: str = "utf-8",
    validator: None = ...,
    allow_comments: bool = False,
) -> FrozenSet[str]: ...
@overload
def load_config(
    file: str | os.PathLike[str],
    mode: Literal["validated_dict"] = "validated_dict",
    *,
    base_dir: Optional[Path] = None,
    encoding: str = "utf-8",
    validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = ...,
    allow_comments: bool = False,
) -> Dict[str, Any]: ...


def load_config(
    file: str | os.PathLike[str],
    mode: Mode = "raw",
    *,
    base_dir: Optional[Path] = None,
    encoding: str = "utf-8",
    validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    allow_comments: bool = False,
) -> Any:
    """
    Does: Load <data>/<file>.json, parse JSON, coerce by mode, cache results (skips cache if validator provided).
    Args: file (w/ or w/o .json), mode ∈ {'raw','set','validated_dict'}, optional base_dir/encoding/validator/allow_comments.
    Returns: Raw JSON (raw), frozenset[str] (set), or dict[str,Any] (validated_dict).
    Raises: DataDirNotFound, ConfigFileNotFound, ConfigParseError, ConfigTypeError, ValueError.
    """
    # Resolve base directory: env override > explicit > discovery
    if base_dir is None:
        env_dir = os.environ.get("DATA_DIR")
        if env_dir:
            base_dir = Path(os.path.expanduser(env_dir))

    data_dir = (base_dir or _default_data_dir()).resolve()

    # Normalize file path and enforce staying under data_dir
    file_str = os.fspath(file)
    file_name = file_str if file_str.endswith(".json") else f"{file_str}.json"
    path = (data_dir / file_name).resolve()
    try:
        path.relative_to(data_dir)
    except ValueError as e:
        raise ConfigFileNotFound(
            f"Refusing to access file outside data dir: {path} (base={data_dir})"
        ) from e

    if not path.is_file():
        raise ConfigFileNotFound(f"Config file not found: {path}")

    cache_key = (path, mode, encoding, validator is not None)

    # Cache hit (only when no validator is used, because validator may change output)
    with _CACHE_LOCK:
        if cache_key in _CONFIG_CACHE and validator is None:
            log.debug("Config cache HIT: %s (mode=%s)", path.name, mode)
            return _CONFIG_CACHE[cache_key]

    # Read & parse
    try:
        with path.open("r", encoding=encoding, errors="strict", newline="", buffering=1) as f:
            if allow_comments:
                # Optional json5 support without hard dependency
                try:
                    import json5  # type: ignore
                    data = json5.load(f)  # allows comments/trailing commas
                except Exception:
                    f.seek(0)
                    data = json.load(f)  # fallback to std json
            else:
                data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigParseError(f"Invalid JSON in {path}: {e}") from e
    except OSError as e:
        raise ConfigFileNotFound(f"Cannot read {path}: {e}") from e

    # Coerce by mode
    if mode == "raw":
        result: Any = data

    elif mode == "set":
        if not isinstance(data, list):
            raise ConfigTypeError(
                f"{path.name}: expected list for mode 'set', got {type(data).__name__}"
            )
        non_scalars = [x for x in data if not isinstance(x, (str, int, float, bool)) and x is not None]
        if non_scalars:
            preview = ", ".join(f"{type(x).__name__}" for x in non_scalars[:3])
            raise ConfigTypeError(
                f"{path.name}: list must contain only scalars for 'set' "
                f"(first bad types: {preview})"
            )
        result = frozenset(map(str, data))

    elif mode == "validated_dict":
        if not isinstance(data, dict):
            raise ConfigTypeError(
                f"{path.name}: expected dict for mode 'validated_dict', got {type(data).__name__}"
            )
        if validator is not None:
            try:
                data = validator(data)
            except Exception as e:
                raise ConfigParseError(f"{path.name}: validator failed: {e}") from e
        result = data

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Store in cache (skip if validator provided)
    if validator is None:
        with _CACHE_LOCK:
            _CONFIG_CACHE[cache_key] = result
            log.debug("Config cache MISS → STORED: %s (mode=%s)", path.name, mode)
    else:
        log.debug("Config loaded (validator present, not cached): %s (mode=%s)", path.name, mode)

    return result
