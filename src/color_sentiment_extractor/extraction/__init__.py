# Compatibility shim: allow legacy imports like "from extraction.x import y"
import sys as _sys
_sys.modules.setdefault("extraction", _sys.modules[__name__])
