# extraction/color/token/split.py
"""
token.split
===========

Does: Scinde des tokens collés/hyphénés en parties valides (suffix-aware), via
      récupération de base stricte et budget-temps dur.
Used By: Stratégies compound (adjacent/glued) et pipelines (décomposition mod+tone).
Returns: split_glued_tokens() → List[str]; split_tokens_to_parts() → Optional[List[str]].
"""

from __future__ import annotations

# ── Imports ──────────────────────────────────────────────────────────────
import logging
import re
import time as _time
from functools import lru_cache
from typing import FrozenSet, Iterable, List, Optional, Set

# Imports internes projet
from color_sentiment_extractor.extraction.color import BLOCKED_TOKENS
from color_sentiment_extractor.extraction.general.token import (
    _recover_base_cached_with_params,
    recover_base,
    normalize_token,
)
from color_sentiment_extractor.extraction.general.token.split import (
    fallback_split_on_longest_substring,
)
from color_sentiment_extractor.extraction.general.token.suffix import (
    build_augmented_suffix_vocab,
    is_suffix_variant,
)
from color_sentiment_extractor.extraction.general.utils import load_config

# ── Logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes locales ───────────────────────────────────────────────────
MIN_PART_LEN_DEFAULT = 3
TIME_BUDGET_DEFAULT = 0.050  # ~50ms par token
MAX_FIRST_CUT_DEFAULT = 10


# ── Accès config en cache ────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_known_modifiers() -> FrozenSet[str]:
    """Does: Charge les known_modifiers (config) une seule fois (cache). Returns: frozenset."""
    # load_config(mode="set") renvoie déjà un frozenset[str] dans notre codebase
    return load_config("known_modifiers", mode="set")


# ── helpers internes typing-friendly ─────────────────────────────────────
def _ensure_modifiers(
    known_modifiers: FrozenSet[str] | None,
) -> FrozenSet[str]:
    """Retourne toujours un FrozenSet[str] non vide (éventuellement vide mais jamais None)."""
    return known_modifiers if known_modifiers is not None else _get_known_modifiers()


def _ensure_vocab(
    known_tokens: Set[str],
    km: FrozenSet[str],
    vocab: Optional[Set[str]],
) -> Set[str]:
    """Construit/retourne le vocab suffix-aware attendu par le splitter."""
    if vocab is not None:
        return vocab
    # build_augmented_suffix_vocab() attend des sets mutables -> on cast via set()
    return build_augmented_suffix_vocab(known_tokens, set(km))


def _any_in(haystack: Iterable[str], km: FrozenSet[str], kt: Set[str]) -> bool:
    """mypy-safe: vérifie si un morceau appartient aux sets connus."""
    for t in haystack:
        if t in km or t in kt:
            return True
    return False


# ── Splitter récursif (glued + suffix-aware) ─────────────────────────────
def split_glued_tokens(
    token: str,
    known_tokens: Set[str],
    known_modifiers: FrozenSet[str] | None = None,
    debug: bool = False,
    vocab: Optional[Set[str]] = None,
    *,
    min_part_len: int = MIN_PART_LEN_DEFAULT,
    max_first_cut: int = MAX_FIRST_CUT_DEFAULT,
    time_budget_sec: float = TIME_BUDGET_DEFAULT,
) -> List[str]:
    """
    Does: Décompose récursivement un token collé en sous-parties valides, via vocab suffix-aware et budget-temps.
    Returns: Liste de morceaux valides (≥1) ou [] si aucun découpage sûr.
    """
    t0 = _time.time()

    km: FrozenSet[str] = _ensure_modifiers(known_modifiers)
    kt: Set[str] = known_tokens

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

    # Vocab étendu (tones + modifiers + variantes suffixées)
    vocab_local: Set[str] = _ensure_vocab(kt, km, vocab)

    @lru_cache(maxsize=2048)
    def is_valid_cached(t: str) -> bool:
        t_norm = normalize_token(t, keep_hyphens=False)
        return (
            t_norm in vocab_local
            or is_suffix_variant(
                t_norm,
                km,
                frozenset(kt),
                debug=False,
                allow_fuzzy=False,
            )
        )

    def _cut_range(s: str) -> range:
        # borne haute sécurisée (≥ min_part_len+1 pour éviter range vide)
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

        # recherche récursive: meilleure décomposition (maximise le nb de morceaux)
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

    # Fallback: longest-substring
    result = fallback_split_on_longest_substring(token, vocab_local, debug=False) or []

    # Valide seulement si au moins un morceau ∈ vocabs de base
    if _any_in(result, km, kt):
        if debug:
            dt = _time.time() - t0
            logger.debug("[split] fallback OK %s in %.3fs for %r", result, dt, token)
        return result

    if debug:
        dt = _time.time() - t0
        why = "timeout" if (dt > time_budget_sec) else "no-match"
        logger.debug("[split] FAIL (%s) in %.3fs for %r", why, dt, token)
    return []


# ── Splitter 2-parties (modifier + tone) strict ──────────────────────────
def split_tokens_to_parts(
    token: str,
    known_tokens: Set[str],
    known_modifiers: FrozenSet[str] | None = None,
    debug: bool = False,
    *,
    min_part_len: int = MIN_PART_LEN_DEFAULT,
    time_budget_sec: float = TIME_BUDGET_DEFAULT,
) -> Optional[List[str]]:
    """
    Does: Tente un split 2-parties (modifier, tone) avec récupération de base stricte (pas de fuzzy) et budget-temps.
    Returns: [left, right] si confiant, sinon None.
    """
    t0 = _time.time()

    km: FrozenSet[str] = _ensure_modifiers(known_modifiers)
    kt: Set[str] = known_tokens

    if not token or len(token) <= 2:
        if debug:
            logger.debug("[split2] skip short token: %r", token)
        return None

    token = normalize_token(token, keep_hyphens=True)
    if not token or len(token) <= 2:
        if debug:
            logger.debug("[split2] skip after normalize: %r", token)
        return None

    # évite de retravailler un token déjà connu comme tone
    if token in kt:
        if debug:
            logger.debug("[split2] token already known: %r", token)
        return None

    best_split: Optional[List[str]] = None
    best_score = -1

    # Hyphens multiples: essaie toutes les coupes possibles
    if "-" in token:
        hyph_parts = token.split("-")
        if len(hyph_parts) >= 2:
            for k in range(1, len(hyph_parts)):
                left = "-".join(hyph_parts[:k])
                right = "-".join(hyph_parts[k:])
                l_ok = (left in kt) or (left in km)
                r_ok = (right in kt) or (right in km)
                if l_ok and r_ok:
                    if debug:
                        logger.debug("[HYPHEN SPLIT MATCH] %r + %r", left, right)
                    return [left, right]

    if debug:
        logger.debug("[SPLIT2 START] Input: %r", token)

    # Garantit deux côtés ≥ min_part_len
    for i in range(min_part_len, len(token) - min_part_len + 1):
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
        if (left in kt) or (left in km):
            left_final = left
            score += 3
            if debug:
                logger.debug("[LEFT KNOWN] %r", left)
        else:
            left_rec = _recover_base_cached_with_params(
                raw=left,
                allow_fuzzy=False,
                fuzzy_threshold=88,
                km=km,
                kt=frozenset(kt),
            )
            # nettoyage chiffres puis retry strict
            if not left_rec and any(ch.isdigit() for ch in left):
                cleaned = re.sub(r"\d+", "", left)
                if cleaned and cleaned != left:
                    left_rec = _recover_base_cached_with_params(
                        raw=cleaned,
                        allow_fuzzy=False,
                        fuzzy_threshold=88,
                        km=km,
                        kt=frozenset(kt),
                    )
                    if debug:
                        logger.debug("[CLEANED LEFT] %r → %r → %r", left, cleaned, left_rec)

            # garde qualité sur segments très courts
            if left_rec and len(left) < 4 and left_rec != left and left_rec not in kt:
                if debug:
                    logger.debug("[REJECT LEFT WEAK] %r → %r", left, left_rec)
                left_rec = None

            if left_rec:
                left_final = left if left in kt else left_rec
                score += 3 if left_final == left else 1
                if debug:
                    logger.debug("[LEFT RECOVERED] %r → %r", left, left_final)

        # ── RIGHT ──
        if (right in kt) or (right in km):
            right_final = right
            score += 3
            if debug:
                logger.debug("[RIGHT KNOWN] %r", right)
        else:
            right_rec = recover_base(
                right,
                known_modifiers=km,
                known_tones=kt,
                debug=False,
                fuzzy_fallback=False,  # strict
                fuzzy_threshold=88,
            )
            if not right_rec and any(ch.isdigit() for ch in right):
                cleaned = re.sub(r"\d+", "", right)
                if cleaned and cleaned != right:
                    right_rec = recover_base(
                        cleaned,
                        known_modifiers=km,
                        known_tones=kt,
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
                right_final = right if right in kt else right_rec
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


# ── Entry Point (optionnel) ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Demo split module: %s", __file__)
    # Mini démo (remplacez par vos sets réels)
    tones = {"rose", "beige", "blue", "navy"}
    mods = frozenset({"dusty", "soft", "deep"})
    print("glued:", split_glued_tokens("dustyrose", tones, mods, debug=True))
    print("2-part:", split_tokens_to_parts("dustyrose", tones, mods, debug=True))
