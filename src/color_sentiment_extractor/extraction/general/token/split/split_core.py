# extraction/general/token/split/split_core.py
"""
Generic token splitting utilities (no color-specific deps).

Provides:
- recursive_token_split(token, is_valid): recursive + fallback splitter
- fallback_split_on_longest_substring(token, vocab): longest-substring heuristic
- has_token_overlap(a, b): quick overlap check between phrases

Notes:
- This version adds a time budget to the fallback splitter to prevent stalls
  on non-splittable common words (e.g. 'love', 'hate').
- It also adds an early trigram check to cheaply reject hopeless tokens.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Set, Tuple
from functools import lru_cache

from extraction.general.token.normalize import normalize_token


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def has_token_overlap(a: str, b: str) -> bool:
    """
    Return True if `a` and `b` share at least one token after normalization.
    """
    a_tokens = set(normalize_token(a).split())
    b_tokens = set(normalize_token(b).split())
    return bool(a_tokens & b_tokens)


def _norm(s: str) -> str:
    return normalize_token(s, keep_hyphens=False).replace(" ", "")


def _build_norm_vocab(vocab: Set[str]) -> Set[str]:
    return {
        _norm(v)
        for v in (vocab or set())
        if v
    }


def _has_vocab_trigram(raw: str, norm_vocab: Set[str]) -> bool:
    """
    Cheap early filter: if no trigram of `raw` occurs in any vocab entry,
    splitting attempts are extremely unlikely to help. Avoids heavy work.
    """
    if len(raw) < 3:
        return False
    tris = {raw[i:i+3] for i in range(len(raw) - 2)}
    # Stop on first hit
    for entry in norm_vocab:
        e = entry  # already normalized
        for tri in tris:
            if tri in e:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main splitter (fallback) with budget & safeguards
# ─────────────────────────────────────────────────────────────────────────────

def fallback_split_on_longest_substring(
    token: str,
    vocab: Set[str],
    *,
    debug: bool = False,
    prefer_split_over_glued: bool = True,
    time_budget_sec: float = 0.040,     # ~40ms hard cap
    min_part_len: int = 2,              # 2 for generic (color layer applies stricter guards)
) -> List[str]:
    """
    Prefer true segmentation (≥2 parts fully covering the token) over returning the glued word,
    even if the glued word is also in vocab. Keeps exact/substring fallbacks if no split exists.

    Added:
      - time_budget_sec: stop searching when budget is exceeded (caps worst-cases)
      - early trigram filter to quickly reject non-candidate tokens
    """

    # local time import (avoid name collision with datetime.time)
    import time as _time
    t0 = _time.time()

    def d(*args):
        if debug:
            print("[splitDBG]", *args)

    raw = _norm(token)
    d(f"RAW={raw!r}")
    if not raw:
        d("Empty after normalize → []")
        return []

    # Normalize vocab once
    norm_vocab = _build_norm_vocab(vocab)
    d(f"VOCAB_SIZE={len(norm_vocab)}")
    if not norm_vocab:
        d("Empty vocab → []")
        return []

    # Early reject: no trigram overlap with vocab → bail early
    if not _has_vocab_trigram(raw, norm_vocab):
        d("Early reject: no trigram overlap with vocab")
        return []

    # Quick bailout helper
    def over_budget() -> bool:
        return (_time.time() - t0) > time_budget_sec

    # ---------- A) QUICK 2-WAY SPLIT PASS (prefer split over glued) ----------
    # Try to split into exactly two vocab pieces (left, right). This is fast and
    # gives the expected result for cases like 'lightmauve' → ['light','mauve'].
    if prefer_split_over_glued and not over_budget():
        best_two: Optional[List[str]] = None
        # i bounds: ensure both sides have reasonable length
        for i in range(min_part_len, len(raw) - min_part_len + 1):
            if over_budget():
                d("Budget hit during 2-way pass")
                break
            left, right = raw[:i], raw[i:]
            if left in norm_vocab and right in norm_vocab:
                # Prefer the longest left piece first (greedy two-way)
                if best_two is None or len(left) > len(best_two[0]):
                    best_two = [left, right]
                    d(f"2-WAY: candidate split at {i}: {best_two}")
        if best_two:
            d(f"2-WAY: ACCEPT {best_two}")
            return best_two

    # ---------- B) GREEDY LONGEST-PREFIX SEGMENTATION ----------
    # (guarded by budget)
    def greedy_segment(s: str) -> List[str]:
        parts: List[str] = []
        rest = s
        sorted_vocab = sorted(norm_vocab, key=len, reverse=True)
        while rest:
            if over_budget():
                return []
            chosen: Optional[str] = None
            for w in sorted_vocab:
                if w and rest.startswith(w) and len(w) >= min_part_len:
                    chosen = w
                    d(f"GREEDY: rest={rest!r} picked={w!r}")
                    break
            if not chosen:
                d(f"GREEDY: dead-end at rest={rest!r}")
                return []
            parts.append(chosen)
            rest = rest[len(chosen):]
        d(f"GREEDY RESULT parts={parts}")
        return parts

    if not over_budget():
        parts = greedy_segment(raw)
        if parts and "".join(parts) == raw and len(parts) >= 2:
            d("ACCEPT GREEDY SPLIT")
            return parts

    # ---------- C) BACKTRACKING (prefer split over glued at depth 0) ----------
    # At depth 0, we skip a whole-word match to allow shorter prefixes to be explored.
    sorted_vocab = tuple(sorted(norm_vocab, key=len, reverse=True))

    @lru_cache(None)
    def backtrack(s: str, depth: int) -> Optional[List[str]]:
        if over_budget():
            return None
        indent = "  " * depth
        d(f"{indent}BT: enter s={s!r}, depth={depth}")
        if not s:
            d(f"{indent}BT: success → []")
            return []

        # First pass: try prefixes strictly shorter than s (avoid swallowing whole word)
        for w in sorted_vocab:
            if over_budget():
                return None
            if not w or len(w) < min_part_len:
                continue
            if len(w) == len(s) and depth == 0:
                # Skip whole-word at the top level to favor a true split
                continue
            if s.startswith(w):
                d(f"{indent}BT: try prefix={w!r} on s={s!r}")
                tail = s[len(w):]
                rest = backtrack(tail, depth + 1)
                if rest is not None:
                    out = [w] + rest
                    d(f"{indent}BT: success via {w!r} → {out}")
                    return out

        # If nothing worked and we are not at top-level (or policy allows), try whole-word
        for w in sorted_vocab:
            if over_budget():
                return None
            if not w:
                continue
            if s.startswith(w) and len(w) == len(s):
                d(f"{indent}BT: accept whole-word={w!r}")
                return [w]

        d(f"{indent}BT: dead-end for s={s!r}")
        return None

    if not over_budget():
        bt = backtrack(raw, 0)
        if bt and "".join(bt) == raw and len(bt) >= 2:
            d("ACCEPT BACKTRACK SPLIT")
            return bt

    # ---------- D) Exact match (only if no split found) ----------
    if not over_budget() and raw in norm_vocab:
        d("EXACT MATCH → [raw]")
        return [raw]

    # ---------- E) Single longest-substring split ----------
    if not over_budget():
        for sub in sorted(norm_vocab, key=len, reverse=True):
            if over_budget():
                break
            if not sub or len(sub) < min_part_len:
                continue
            idx = raw.find(sub)
            if idx != -1 and sub != raw:
                out: List[str] = []
                if idx > 0:
                    out.append(raw[:idx])
                out.append(sub)
                end = idx + len(sub)
                if end < len(raw):
                    out.append(raw[end:])
                d(f"SUBSTRING SPLIT on {sub!r} at {idx} → {out}")
                return out

    d("NO SPLIT FOUND → [] (or budget hit)")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Recursive splitter (kept as-is, optional improvement: min_part_len)
# ─────────────────────────────────────────────────────────────────────────────

def recursive_token_split(token: str, is_valid: Callable[[str], bool]) -> Optional[List[str]]:
    """
    Recursively split a (possibly glued) token into valid parts using `is_valid`.
    Strategy:
      1) Accept the token as-is if valid.
      2) Try recursive binary splits between positions 3..len-3.
      3) Fallback: scan for a valid prefix (longest-first) or valid suffix (shortest-first).
    Returns list of parts or None if no decomposition works.
    """

    raw = _norm(token)
    if not raw:
        return None

    @lru_cache(maxsize=4096)
    def _split(t: str) -> Optional[List[str]]:
        if not t:
            return None
        if is_valid(t):
            return [t]

        n = len(t)
        # Try binary splits
        for i in range(3, max(3, n - 2)):
            left, right = t[:i], t[i:]
            if not left or not right:
                continue
            l_parts = _split(left)
            if l_parts:
                r_parts = _split(right)
                if r_parts:
                    return l_parts + r_parts

        # Fallback: valid prefix (longest-first)
        for i in range(n - 1, 2, -1):
            prefix, rest = t[:i], t[i:]
            if is_valid(prefix):
                r_parts = _split(rest)
                if r_parts:
                    return [prefix] + r_parts

        # Fallback: valid suffix (shortest-first)
        for i in range(3, n):
            rest, suffix = t[:i], t[i:]
            if is_valid(suffix):
                r_parts = _split(rest)
                if r_parts:
                    return r_parts + [suffix]

        return None

    return _split(raw)
