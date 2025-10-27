# extraction/general/token/split/split_core.py

"""
split_core.py.

Does: Fast, budgeted splitting of glued tokens via 2-way greedy,
      greedy-prefix, and memoized backtracking, with trigram
      early-filter and cached vocab prep.
Returns: Lists of normalized parts (str) or []/None when no valid split.
Used by: Tokenization/extraction stages that must deglue candidate tokens.
"""
from __future__ import annotations

import logging
import time as _time
from collections.abc import Callable
from functools import lru_cache

from color_sentiment_extractor.extraction.general.token import normalize_token  # ← move up here

__all__ = [
    "has_token_overlap",
    "fallback_split_on_longest_substring",
    "recursive_token_split",
]

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────


def has_token_overlap(a: str, b: str) -> bool:
    """
    Does: Check if two phrases share ≥1 normalized token.
    Returns: True if intersection of token sets is non-empty.
    Used by: Quick guards to avoid redundant splitting work.
    """
    a_tokens = set(normalize_token(a).split())
    b_tokens = set(normalize_token(b).split())
    return bool(a_tokens & b_tokens)


def _strip_fancy_hyphens_apostrophes(s: str) -> str:
    # Normalize common Unicode punctuation variants
    hyphens = "\u2010\u2011\u2012\u2013\u2014\u2212"  # ‐ - ‒ – — −
    quotes = "\u2018\u2019\u201b\u2032\u02bc"  # ‘ ’ ‛ ′ ʼ
    for ch in hyphens + quotes:
        s = s.replace(ch, "-")
    return s


def _norm(s: str) -> str:
    # Compact normalization for glued-token logic
    s = _strip_fancy_hyphens_apostrophes(s or "")
    return normalize_token(s, keep_hyphens=False).replace(" ", "")


def _build_norm_vocab(vocab: set[str]) -> set[str]:
    return {_norm(v) for v in (vocab or set()) if v}


def _vocab_trigram_index(vset: set[str]) -> set[str]:
    """
    Does: Build set of all trigrams present in normalized vocab.
    Returns: Set[str] of trigrams for early filtering.
    Used by: Early-reject of hopeless split candidates.
    """
    idx: set[str] = set()
    for e in vset:
        n = len(e)
        if n >= 3:
            idx.update(e[i : i + 3] for i in range(n - 2))
    return idx


def _has_vocab_trigram(raw: str, vocab_tris: set[str]) -> bool:
    """
    Does: Test if any trigram of `raw` appears in vocab trigrams.
    Returns: True if at least one trigram matches.
    Used by: Cheap early filter before heavier search.
    """
    n = len(raw)
    if n < 3:
        return False
    for i in range(n - 2):
        if raw[i : i + 3] in vocab_tris:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Cached vocab preparation
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def _prep_vocab_cached(fv: frozenset[str]) -> tuple[set[str], tuple[str, ...], set[str]]:
    """
    Does: Prepare normalized vocab, sorted view, and trigram index.
    Returns: (norm_vocab, sorted_vocab_len_desc, vocab_trigrams).
    Used by: All splitting strategies sharing the same vocab.
    """
    norm_vocab = set(fv)
    # Deterministic: length desc, then lexicographic asc
    sorted_vocab = tuple(sorted(norm_vocab, key=lambda w: (-len(w), w)))
    vocab_tris = _vocab_trigram_index(norm_vocab)
    return norm_vocab, sorted_vocab, vocab_tris


# ─────────────────────────────────────────────────────────────────────────────
# Main splitter (budgeted with safeguards)
# ─────────────────────────────────────────────────────────────────────────────


def fallback_split_on_longest_substring(
    token: str,
    vocab: set[str],
    *,
    debug: bool = False,
    prefer_split_over_glued: bool = True,
    time_budget_sec: float = 0.040,
    min_part_len: int = 2,
    max_parts: int = 6,
) -> list[str]:
    """
    Does: Deterministically segment `token` with vocab under a time
          budget, preferring real splits over glued; falls back to
          exact/substring.
    Returns: Parts covering token (≥2) or exact/substring fallback;
             else [].
    Used by: Degluing before higher-level color/expression extraction.
    """
    t0 = _time.monotonic()

    def over_budget() -> bool:
        return (_time.monotonic() - t0) > time_budget_sec

    def d(*args):
        if debug:
            log.debug(" ".join(str(a) for a in args))

    raw = _norm(token)
    d(f"RAW={raw!r}")
    if not raw:
        return []

    norm_vocab_input = _build_norm_vocab(vocab)
    if not norm_vocab_input:
        return []
    fv = frozenset(norm_vocab_input)
    norm_vocab, sorted_vocab, vocab_tris = _prep_vocab_cached(fv)

    if not _has_vocab_trigram(raw, vocab_tris):
        return []

    # A) Two-way split pass (deterministic tie-breaks)
    if prefer_split_over_glued and not over_budget():
        best_two: tuple[str, str] | None = None
        for i in range(min_part_len, len(raw) - min_part_len + 1):
            if over_budget():
                break
            left, right = raw[:i], raw[i:]
            if left in norm_vocab and right in norm_vocab:
                cand = (left, right)
                if best_two is None or (len(left), len(right), cand) > (
                    len(best_two[0]),
                    len(best_two[1]),
                    best_two,
                ):
                    best_two = cand
        if best_two and 2 <= max_parts:
            return [best_two[0], best_two[1]]

    # B) Greedy longest-prefix segmentation
    def greedy_segment(s: str) -> list[str]:
        parts: list[str] = []
        rest = s
        while rest:
            if over_budget():
                return []
            chosen = next(
                (w for w in sorted_vocab if len(w) >= min_part_len and rest.startswith(w)),
                None,
            )
            if not chosen:
                return []
            parts.append(chosen)
            if len(parts) > max_parts:
                return []
            rest = rest[len(chosen) :]
        return parts

    if not over_budget():
        parts = greedy_segment(raw)
        if parts and "".join(parts) == raw and len(parts) >= 2:
            return parts

    # C) Backtracking (memoized), avoid swallowing whole at top level
    @lru_cache(maxsize=4096)
    def backtrack(s: str, allow_whole: bool) -> tuple[str, ...] | None:
        if over_budget():
            return None
        if not s:
            return tuple()
        for w in sorted_vocab:
            if over_budget():
                return None
            if len(w) < min_part_len:
                continue
            if len(w) == len(s) and not allow_whole:
                continue
            if s.startswith(w):
                rest = backtrack(s[len(w) :], True)
                if rest is not None:
                    parts = (w,) + rest
                    if len(parts) <= max_parts:
                        return parts
        if allow_whole and s in norm_vocab:
            return (s,)
        return None

    if not over_budget():
        bt = backtrack(raw, False)
        if bt and "".join(bt) == raw and 2 <= len(bt) <= max_parts:
            return list(bt)

    # D) Exact match (no split found)
    if not over_budget() and raw in norm_vocab:
        return [raw]

    # E) Single longest-substring split (sides may be out-of-vocab)
    if not over_budget():
        for sub in sorted_vocab:
            if over_budget():
                break
            if len(sub) < min_part_len:
                continue
            idx = raw.find(sub)
            if idx != -1 and sub != raw:
                out: list[str] = []
                if idx > 0:
                    out.append(raw[:idx])
                out.append(sub)
                end = idx + len(sub)
                if end < len(raw):
                    out.append(raw[end:])
                if len(out) <= max_parts:
                    return out

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Recursive splitter (binary + prefix/suffix fallbacks)
# ─────────────────────────────────────────────────────────────────────────────


def recursive_token_split(token: str, is_valid: Callable[[str], bool]) -> list[str] | None:
    """
    Does: Recursively decompose a glued token using `is_valid` parts
          and binary splits with prefix/suffix fallbacks.
    Returns: List of parts or None when no valid decomposition exists.
    Used by: Generic recovery when only a validator is available
             (no vocab).
    """
    raw = _norm(token)
    if not raw:
        return None

    @lru_cache(maxsize=4096)
    def _split(t: str) -> tuple[str, ...] | None:
        if not t:
            return None
        if is_valid(t):
            return (t,)

        n = len(t)
        # Binary splits between positions 3..len-3
        for i in range(3, max(3, n - 2)):
            left, right = t[:i], t[i:]
            if not left or not right:
                continue
            l_parts = _split(left)
            if l_parts:
                r_parts = _split(right)
                if r_parts:
                    return l_parts + r_parts

        # Valid prefix (longest-first)
        for i in range(n - 1, 2, -1):
            prefix, rest = t[:i], t[i:]
            if is_valid(prefix):
                r_parts = _split(rest)
                if r_parts:
                    return (prefix,) + r_parts

        # Valid suffix (shortest-first)
        for i in range(3, n):
            rest, suffix = t[:i], t[i:]
            if is_valid(suffix):
                r_parts = _split(rest)
                if r_parts:
                    return r_parts + (suffix,)

        return None

    parts = _split(raw)
    return list(parts) if parts else None
