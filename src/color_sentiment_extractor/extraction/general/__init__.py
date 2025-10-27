"""
general.
=======

Shared general-purpose modules used across the extraction pipeline.

Exports:
- TokenLike: Protocol for spaCy-like tokens.
"""

from .types import TokenLike

__all__ = ["TokenLike"]
