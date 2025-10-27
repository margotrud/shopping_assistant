# color_sentiment_extractor/extraction/__init__.py

"""
extraction (compat shim).
========================

Does: Provide a compatibility alias so legacy imports like `from extraction.x import y`
      keep working after the package rename to `color_sentiment_extractor.extraction`.
Returns: Registers the current module under the top-level name "extraction" if absent.
Used by: Older scripts, notebooks, and external integrations.
"""
from __future__ import annotations

import sys as _sys

# Only register the legacy alias if we're not already imported as "extraction".
if __name__ != "extraction" and "extraction" not in _sys.modules:
    _sys.modules["extraction"] = _sys.modules[__name__]

__all__: list[str] = []
__docformat__ = "google"
