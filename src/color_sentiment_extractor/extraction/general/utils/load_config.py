# src/color_sentiment_extractor/extraction/general/utils/load_config.py

"""Load JSON configs from a <data/> directory with caching and typed coercions.

Modes:
- "raw"             -> return parsed JSON as-is
- "set"             -> return frozenset[str] (coerce scalars to str)
- "validated_dict"  -> return dict[str, Any] after an optional validator

Used by vocab loaders, extraction pipelines, and tests needing hot reload.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, overload

# --- optional json5 support (no hard dependency) -----------------------------
try:  # mypy: json5 may be missing in most envs
    import json5 as _json5
except Exception:  # pragma: no cover - only hit when json5 missing
    _json5 = None  # type: ignore[assignment]

# ── Public surface ────────────────────────────────────────────────────────────
Mode = Literal["raw", "set", "validated_dict"]
__all__ = [
    "Mode",
    "load_config",
    "clear_config_cache",
    "temp_data_dir",
    "DataDirNotFound",
    "ConfigFileNotFound",
    "ConfigParseError",
    "ConfigTypeError",
]


# ── Exceptions ───────────────────────────────────────────────────────────────
class DataDirNotFound(FileNotFoundError):
    """Raise when no 'data' directory is found while walking upwards."""


class ConfigFileNotFound(FileNotFoundError):
    """Raise when the requested config file cannot be read or resolved."""


class ConfigParseError(ValueError):
    """Raise when JSON parsing/validation fails for a config file."""


class ConfigTypeError(TypeError):
    """Raise when the parsed JSON doesn't match the expected structure."""


# ── Logging & cache ──────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
_CACHE_LOCK = threading.RLock()
# cache key includes: path, mtime, mode, encoding, allow_comments, validator_present
_CONFIG_CACHE: dict[tuple[Path, float, str, str, bool, bool], Any] = {}


def clear_config_cache() -> None:
    """Empty the in-memory config cache (useful for pytest/hot-reload)."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        log.debug("Config cache cleared.")


def _candidate_data_dirs(start: Path | None = None) -> list[Path]:
    """Compute candidate 'data'/'Data' directories walking up from start."""
    start = (start or Path(__file__)).resolve()
    cands: list[Path] = []
    for p in [start, *start.parents]:
        for name in ("data", "Data"):
            cands.append((p / name).resolve())
    # also consider repo-style src-root /data when running from site-packages layout
    return cands


def _default_data_dir(start: Path | None = None) -> Path:
    """Return the first existing candidate directory or raise."""
    for cand in _candidate_data_dirs(start):
        if cand.is_dir():
            return cand
    raise DataDirNotFound(
        "No 'data' directory found.\n"
        "Tried:\n  " + "\n  ".join(str(p) for p in _candidate_data_dirs(start))
    )


def _env_data_dir() -> Path | None:
    """Resolve data dir from env if set."""
    for var in ("DATA_DIR", "COLOR_SENTIMENT_DATA_DIR", "COLOR_DATA_DIR"):
        v = os.environ.get(var)
        if v:
            return Path(os.path.expanduser(v)).resolve()
    return None


@overload
def load_config(
    file: str | os.PathLike[str],
    mode: Literal["raw"] = "raw",
    *,
    base_dir: Path | None = None,
    encoding: str = "utf-8",
    validator: None = ...,
    allow_comments: bool = False,
) -> Any: ...
@overload
def load_config(
    file: str | os.PathLike[str],
    mode: Literal["set"] = "set",
    *,
    base_dir: Path | None = None,
    encoding: str = "utf-8",
    validator: None = ...,
    allow_comments: bool = False,
) -> frozenset[str]: ...
@overload
def load_config(
    file: str | os.PathLike[str],
    mode: Literal["validated_dict"] = "validated_dict",
    *,
    base_dir: Path | None = None,
    encoding: str = "utf-8",
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = ...,
    allow_comments: bool = False,
) -> dict[str, Any]: ...


def load_config(
    file: str | os.PathLike[str],
    mode: Mode = "raw",
    *,
    base_dir: Path | None = None,
    encoding: str = "utf-8",
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    allow_comments: bool = False,
) -> Any:
    """Load <data>/<file>.json, parse, coerce by mode, and cache results."""
    # Resolve base directory: env override > explicit > discovery
    if base_dir is None:
        env_dir = _env_data_dir()
        base_dir = env_dir or _default_data_dir()

    data_dir = base_dir.resolve()

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

    # mtime-based cache key for auto-invalidation when file changes
    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        raise ConfigFileNotFound(f"Cannot stat {path}: {e}") from e

    cache_key = (path, mtime, mode, encoding, allow_comments, validator is not None)

    # Cache hit (only when no validator is used, because validator may change output)
    with _CACHE_LOCK:
        if cache_key in _CONFIG_CACHE and validator is None:
            log.debug("Config cache HIT: %s (mode=%s)", path.name, mode)
            return _CONFIG_CACHE[cache_key]

    # Read & parse
    try:
        with path.open("r", encoding=encoding, errors="strict", newline="") as f:
            if allow_comments:
                if _json5 is None:
                    raise ConfigParseError(
                        "json5 requested (allow_comments=True) but not installed"
                    )
                data = _json5.load(f)  # allows comments/trailing commas
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
        non_scalars = [
            x for x in data if not isinstance(x, (str, int, float, bool)) and x is not None
        ]
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


# ── Context manager to temporarily override the data directory ───────────────
class temp_data_dir:
    """Temporarily set the data directory via env for the block."""

    def __init__(self, path: os.PathLike[str] | str):
        self._new = str(path)
        self._old: str | None = None

    def __enter__(self) -> temp_data_dir:
        self._old = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self._new
        clear_config_cache()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._old is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._old
        clear_config_cache()
