# extraction/color/token/split.py

from __future__ import annotations

import logging
import re
import time as _time  # alias pour éviter le shadowing
from functools import lru_cache
from typing import FrozenSet, List, Optional, Set

# ⬇️ constants via color package
from color_sentiment_extractor.extraction.color import BLOCKED_TOKENS

from color_sentiment_extractor.extraction.general.token.base_recovery import (
    _recover_base_cached_with_params,
    recover_base,
)
from color_sentiment_extractor.extraction.general.token.normalize import normalize_token
from color_sentiment_extractor.extraction.general.token.split.split_core import (
    fallback_split_on_longest_substring,
)
from color_sentiment_extractor.extraction.general.token.suffix.recovery import (
    build_augmented_suffix_vocab,
    is_suffix_variant,
)
from color_sentiment_extractor.extraction.general.utils.load_config import load_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_known_modifiers() -> FrozenSet[str]:
    """Load known modifiers from config once (cached)."""
    return frozenset(load_config("known_modifiers", mode="set"))


def split_glued_tokens(
    token: str,
    known_tokens: Set[str],
    known_modifiers: Optional[Set[str]] = None,
    debug: bool = False,
    vocab: Optional[Set[str]] = None,
    *,
    min_part_len: int = 3,
    max_first_cut: int = 10,
    time_budget_sec: float = 0.050,  # ~50ms par token
) -> List[str]:
    """
    Recursively split a glued token into known parts (suffix-aware) with time budget and fallback.
    Returns: list of valid parts or [] if none.
    """
    t0 = _time.time()
    if known_modifiers is None:
        known_modifiers = _get_known_modifiers()

    if not token or not isinstance(token, str):
        return []

    if len(token) <= 2:
        if debug:
            logger.debug("[split] skip short token: %r", token)
        return []

    tok_norm = normalize_token(token, keep_hyphens=True)
    if not tok_norm or len(tok_norm) <= 2:
        if debug:
            logger.debug("[split] skip after normalize: %r -> %r", token, tok_norm)
        return []

    token = tok_norm

    # Vocab étendu (tones + modifiers + variantes suffix)
    if vocab is None:
        vocab = build_augmented_suffix_vocab(known_tokens, known_modifiers)

    @lru_cache(maxsize=2048)
    def is_valid_cached(t: str) -> bool:
        return (
            t in vocab
            or is_suffix_variant(
                t,
                frozenset(known_modifiers),
                frozenset(known_tokens),
                debug=False,
                allow_fuzzy=False,
            )
        )

    def _cut_range(s: str) -> range:
        # borne haute sécurisée (au moins min_part_len+1 pour éviter range vide)
        upper = min(len(s), max_first_cut)
        upper = max(upper, min_part_len + 1)
        return range(min_part_len, upper)

    @lru_cache(maxsize=2048)
    def recursive_split_cached(tok: str) -> Optional[List[str]]:
        # respect du budget temps
        if (_time.time() - t0) > time_budget_sec:
            return None
        # token entier valide ?
        if is_valid_cached(tok):
            return [tok]

        # raccourci: 2 morceaux valides d'un coup
        for i in _cut_range(tok):
            left, right = tok[:i], tok[i:]
            if not left.isalpha() or len(left) < min_part_len:
                continue
            if is_valid_cached(left) and is_valid_cached(right):
                return [left, right]

        # recherche récursive: choisir la meilleure décomposition
        best_split: Optional[List[str]] = None
        for i in _cut_range(tok):
            left, right = tok[:i], tok[i:]
            if not left.isalpha() or len(left) < min_part_len:
                continue
            if is_valid_cached(left):
                right_parts = recursive_split_cached(right)
                if right_parts:
                    candidate = [left] + right_parts
                    if not best_split or len(candidate) > len(best_split):
                        best_split = candidate
        return best_split

    parts = recursive_split_cached(token)
    if parts:
        if debug:
            dt = _time.time() - t0
            logger.debug("[split] recursive OK %s in %.3fs for %r", parts, dt, token)
        return parts

    # Fallback: longest substring match
    result = fallback_split_on_longest_substring(token, vocab, debug=False) or []
    # On valide seulement si au moins un morceau existe réellement dans les vocabs de base
    if any(t in known_modifiers or t in known_tokens for t in result):
        if debug:
            dt = _time.time() - t0
            logger.debug("[split] fallback OK %s in %.3fs for %r", result, dt, token)
        return result

    if debug:
        dt = _time.time() - t0
        why = "timeout" if (dt > time_budget_sec) else "no-match"
        logger.debug("[split] FAIL (%s) in %.3fs for %r", why, dt, token)
    return []


def split_tokens_to_parts(
    token: str,
    known_tokens: Set[str],
    known_modifiers: Optional[Set[str]] = None,
    debug: bool = False,
    *,
    min_part_len: int = 3,
    time_budget_sec: float = 0.050,
) -> Optional[List[str]]:
    """
    Try a 2-part split (modifier + tone) via scoring + base recovery (no fuzzy) under time budget.
    Returns: [left, right] if found, else None.
    """
    t0 = _time.time()
    if known_modifiers is None:
        known_modifiers = _get_known_modifiers()

    if not token or len(token) <= 2:
        if debug:
            logger.debug("[split2] skip short token: %r", token)
        return None

    token = normalize_token(token, keep_hyphens=True)
    if not token or len(token) <= 2:
        if debug:
            logger.debug("[split2] skip after normalize: %r", token)
        return None

    # évite de retravailler un token déjà connu
    if token in known_tokens:
        if debug:
            logger.debug("[split2] token already known: %r", token)
        return None

    best_split: Optional[List[str]] = None
    best_score = -1

    # Cas simple: 'xxx-yyy' (on accepte mod ou tone des deux côtés)
    if "-" in token:
        left, right = token.split("-", 1)
        l_ok = left in known_tokens or left in known_modifiers
        r_ok = right in known_tokens or right in known_modifiers
        if l_ok and r_ok:
            if debug:
                logger.debug("[HYPHEN SPLIT MATCH] %r + %r", left, right)
            return [left, right]

    if debug:
        logger.debug("[SPLIT2 START] Input: %r", token)

    for i in range(min_part_len, max(min_part_len + 1, len(token)) - 1):
        if (_time.time() - t0) > time_budget_sec:
            if debug:
                logger.debug("[split2] timeout budget reached")
            break

        left, right = token[:i], token[i:]
        if debug:
            logger.debug("[TRY SPLIT2] %r + %r", left, right)

        if len(left) < min_part_len or len(right) < min_part_len:
            if debug:
                logger.debug("[split2] one side too short → skip")
            continue

        score = 0
        left_final: Optional[str] = None
        right_final: Optional[str] = None

        # ── LEFT ──
        if left in known_tokens or left in known_modifiers:
            left_final = left
            score += 3
            if debug:
                logger.debug("[LEFT KNOWN] %r", left)
        else:
            left_rec = _recover_base_cached_with_params(
                raw=left,
                allow_fuzzy=False,
                fuzzy_threshold=88,
                km=frozenset(known_modifiers),
                kt=frozenset(known_tokens),
            )
            # nettoyage chiffres
            if not left_rec and any(ch.isdigit() for ch in left):
                cleaned = re.sub(r"\d+", "", left)
                if cleaned and cleaned != left:
                    left_rec = _recover_base_cached_with_params(
                        raw=cleaned,
                        allow_fuzzy=False,
                        fuzzy_threshold=88,
                        km=frozenset(known_modifiers),
                        kt=frozenset(known_tokens),
                    )
                    if debug:
                        logger.debug("[CLEANED LEFT] %r → %r → %r", left, cleaned, left_rec)

            # garde qualité sur très courts segments
            if left_rec and len(left) < 4 and left_rec != left and left_rec not in known_tokens:
                if debug:
                    logger.debug("[REJECT LEFT WEAK] %r → %r", left, left_rec)
                left_rec = None

            if left_rec:
                left_final = left if left in known_tokens else left_rec
                score += 3 if left_final == left else 1
                if debug:
                    logger.debug("[LEFT RECOVERED] %r → %r", left, left_final)

        # ── RIGHT ──
        if right in known_tokens or right in known_modifiers:
            right_final = right
            score += 3
            if debug:
                logger.debug("[RIGHT KNOWN] %r", right)
        else:
            right_rec = recover_base(
                right,
                known_modifiers=known_modifiers,
                known_tones=known_tokens,
                debug=False,
                fuzzy_fallback=False,  # strict
                fuzzy_threshold=88,
            )
            if not right_rec and any(ch.isdigit() for ch in right):
                cleaned = re.sub(r"\d+", "", right)
                if cleaned and cleaned != right:
                    right_rec = recover_base(
                        cleaned,
                        known_modifiers=known_modifiers,
                        known_tones=known_tokens,
                        debug=False,
                        fuzzy_fallback=False,
                        fuzzy_threshold=88,
                    )
                    if debug:
                        logger.debug("[CLEANED RIGHT] %r → %r → %r", right, cleaned, right_rec)

            # garde qualité
            if right_rec and len(right) < 4 and right_rec != right:
                if debug:
                    logger.debug("[REJECT RIGHT WEAK] %r → %r", right, right_rec)
                right_rec = None

            if right_rec:
                right_final = right if right in known_tokens else right_rec
                score += 3 if right_final == right else 1
                if debug:
                    logger.debug("[RIGHT RECOVERED] %r → %r", right, right_final)

        # Paires bloquées
        lchk = left_final.strip().lower() if left_final else None
        rchk = right_final.strip().lower() if right_final else None
        if lchk and rchk and (lchk, rchk) in BLOCKED_TOKENS:
            if debug:
                logger.debug("[BLOCKED PAIR] (%s, %s) → skip", lchk, rchk)
            continue

        candidate_parts = [x for x in (left_final, right_final) if x]
        if candidate_parts and (
            score > best_score
            or (score == best_score and best_split and len(candidate_parts) > len(best_split))
        ):
            best_split = candidate_parts
            best_score = score
            if debug:
                logger.debug("[BEST SO FAR] score=%d → %s", score, best_split)

    if debug:
        dt = _time.time() - t0
        if best_split:
            logger.debug("[FINAL SPLIT2] %s (score=%d) in %.3fs", best_split, best_score, dt)
        else:
            logger.debug("[NO VALID SPLIT2] in %.3fs", dt)

    return best_split
