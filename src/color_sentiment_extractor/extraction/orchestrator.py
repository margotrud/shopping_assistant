# orchestrator.py
from __future__ import annotations

"""
orchestrator.py
===============

Does: High-level orchestration to analyze color phrases (with/without sentiment),
      deduplicate results, and build a nearest-color palette from known web colors.
Returns:
  - analyze_colors_with_sentiment(text, top_k) -> {"positif": [...], "negatif": [...]}
  - analyze_colors(text, top_k) -> {"primary": {...}, "alternatives": [...], "tones": set, "phrases": list, "rgb_map": dict}
Used by: Product/search flows, color preference UIs, and debugging tools.
"""

import logging
import os
import sys
from functools import lru_cache

from color_sentiment_extractor.extraction.color.llm import get_llm_client
from color_sentiment_extractor.extraction.color.logic import (
    aggregate_color_phrase_results,
)
from color_sentiment_extractor.extraction.color.logic.pipelines.rgb_pipeline import (
    resolve_rgb_with_llm,
)
from color_sentiment_extractor.extraction.color.utils.rgb_distance import (
    _try_simplified_match,
    rgb_distance,
)
from color_sentiment_extractor.extraction.color.vocab import (
    all_webcolor_names,
    get_known_tones,
)
from color_sentiment_extractor.extraction.general.sentiment import (
    analyze_sentence_sentiment,
)
from color_sentiment_extractor.extraction.general.token import recover_base
from color_sentiment_extractor.extraction.general.utils.load_config import load_config

logger = logging.getLogger(__name__)

# ── Config snapshots ──────────────────────────────────────────────────────────
known_modifiers: set[str] = load_config("known_modifiers", mode="set")
known_tones: set[str] = get_known_tones()
# validated_dict: Dict[str, Dict[str, List[str]]] (aliases/modifiers/etc.)
expression_map: dict[str, dict[str, list[str]]] = load_config(
    "expression_definition", mode="validated_dict"
)

# Public toggle (env)
USE_LLM = os.getenv("CSE_USE_LLM", "1") == "1"

__all__ = [
    "analyze_colors_with_sentiment",
    "analyze_colors",
]


# =============================================================================
# Optional guard: safer split_glued_tokens (opt-in)
# =============================================================================
def _enable_glued_guard_if_requested() -> None:
    """Activates a guard on split_glued_tokens to short-circuit ultra-short tokens.
    No-ops safely if the module/symbol is missing.
    Triggered only when CSE_ENABLE_GLUED_GUARD=1.
    """
    if os.getenv("CSE_ENABLE_GLUED_GUARD") != "1":
        return
    try:
        import importlib

        m = importlib.import_module(
            "color_sentiment_extractor.extraction.general.token.split.split_core"
        )
    except Exception:
        logger.warning("[patch] split_glued_tokens module NOT found")
        return

    if not hasattr(m, "split_glued_tokens"):
        logger.warning("[patch] split_glued_tokens symbol NOT found in module")
        return

    _orig = m.split_glued_tokens  # type: ignore[attr-defined]

    def _patched(token: str, *a, **kw):
        if not token or len(token) <= 2:
            return []
        return _orig(token, *a, **kw)

    m.split_glued_tokens = _patched  # type: ignore[attr-defined]
    logger.info("[patch] split_glued_tokens guard ACTIVE in %s", m.__name__)


# =============================================================================
# Helpers
# =============================================================================


def _coerce_rgb(rgb: object) -> tuple[int, int, int] | None:
    """Coerces diverse RGB representations to a native (r, g, b) tuple."""
    if rgb is None:
        return None
    if isinstance(rgb, tuple) and len(rgb) == 3:
        try:
            return int(rgb[0]), int(rgb[1]), int(rgb[2])
        except Exception:
            return None
    r = getattr(rgb, "red", None)
    g = getattr(rgb, "green", None)
    b = getattr(rgb, "blue", None)
    if None not in (r, g, b):
        try:
            return int(r), int(g), int(b)
        except Exception:
            return None
    try:
        return int(rgb[0]), int(rgb[1]), int(rgb[2])  # type: ignore[index]
    except Exception:
        return None


def _map_name_rgb_list(
    items: list[dict[str, object]],
) -> list[dict[str, tuple[int, int, int]]]:
    """Apply _coerce_rgb on a list of {'name':..., 'rgb':...} dicts."""
    out: list[dict[str, tuple[int, int, int]]] = []
    for d in items:
        if not isinstance(d, dict):
            continue
        name = d.get("name")
        rgb = d.get("rgb")
        if name is None or rgb is None:
            continue
        coerced = _coerce_rgb(rgb)
        if coerced is None:
            continue
        out.append({"name": str(name), "rgb": coerced})
    return out


def _norm_name(name: str) -> str:
    """Normalize CSS/XKCD names for strict de-duplication:
    - trim
    - lowercase
    - remove spaces, hyphens and underscores
    """
    s = str(name).strip().lower()
    return s.replace(" ", "").replace("-", "").replace("_", "")


def _unique_by_name(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Stable-deduplicate by normalized 'name' while preserving order."""
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        raw = (it.get("name") if isinstance(it, dict) else None) or ""
        key = _norm_name(str(raw))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _drop_items_with_none_rgb(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove entries with missing or malformed RGB."""
    out: list[dict[str, object]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rgb = it.get("rgb")
        if (
            isinstance(rgb, tuple)
            and len(rgb) == 3
            and all(isinstance(c, int) for c in rgb)
        ):
            out.append(it)
    return out


def _dedupe_cross_lists(
    positives: list[dict[str, object]],
    negatives: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove from negatives any color already present in positives (by normalized name)."""
    pos_keys = {
        _norm_name(p.get("name", ""))
        for p in positives
        if isinstance(p, dict) and "name" in p
    }
    return [
        n
        for n in negatives
        if (isinstance(n, dict) and _norm_name(n.get("name", "")) not in pos_keys)
    ]


def _dedup_singles_against_compounds(
    singles: set[str],
    compounds: set[str],
    known_mods: set[str],
    known_tns: set[str],
) -> set[str]:
    """Drop single tokens covered by at least one compound (surface + recover_base)."""
    covered: set[str] = set()
    for comp in compounds:
        for w in comp.split():
            covered.add(w)
            base = recover_base(
                w,
                known_modifiers=known_mods,
                known_tones=known_tns,
                fuzzy_fallback=False,
            )
            if base:
                covered.add(base)
    return {s for s in singles if s not in covered}


def _build_known_palette(names: set[str]) -> dict[str, tuple[int, int, int]]:
    """Build {name -> (r,g,b)} from CSS/XKCD names."""
    palette: dict[str, tuple[int, int, int]] = {}
    for nm in names:
        rgb = _try_simplified_match(nm, debug=False)
        if rgb:
            tpl = _coerce_rgb(rgb)
            if tpl:
                palette[nm] = tpl
    return palette


@lru_cache(maxsize=1)
def _get_known_palette_cached() -> dict[str, tuple[int, int, int]]:
    return _build_known_palette(set(all_webcolor_names))


def _top_k_nearest(
    target_rgb: tuple[int, int, int],
    palette: dict[str, tuple[int, int, int]],
    exclude: set[str],
    k: int,
) -> list[dict[str, tuple[int, int, int]]]:
    """Return top-k nearest colors (name + rgb) without distances."""
    scored: list[tuple[float, str, tuple[int, int, int]]] = []
    exclude_l = {e.strip().lower() for e in exclude}
    for name, rgb in palette.items():
        if str(name).strip().lower() in exclude_l:
            continue
        d = rgb_distance(target_rgb, rgb)
        scored.append((d, name, rgb))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [{"name": name, "rgb": rgb} for _, name, rgb in scored[:k]]


def _pick_primary(tones: set[str], phrases: set[str]) -> str | None:
    """Pick a primary phrase: prefer compounds, otherwise first sorted."""
    if tones:
        comps = [t for t in tones if " " in t]
        return sorted(comps, key=len, reverse=True)[0] if comps else sorted(tones)[0]
    if phrases:
        comps = [p for p in phrases if " " in p]
        return sorted(comps, key=len, reverse=True)[0] if comps else sorted(phrases)[0]
    return None


def _resolve_rgb_for_phrase(
    phrase: str | None,
    llm_client,
    debug: bool = False,
) -> tuple[int, int, int] | None:
    """Resolve RGB robustly: CSS/XKCD first, then LLM fallback if enabled."""
    if not phrase:
        return None
    rgb = _try_simplified_match(phrase, debug=debug)
    if rgb:
        return _coerce_rgb(rgb)
    if USE_LLM:
        rgb = resolve_rgb_with_llm(
            phrase,
            llm_client=llm_client,
            cache=None,
            debug=debug,
            prefer_db_first=True,
        )
        return _coerce_rgb(rgb)
    return None


def _extract_group_colors(
    group_segments: list[str],
    llm_client,
    debug: bool = False,
) -> tuple[set[str], set[str], dict[str, tuple[int, int, int]]]:
    """Run the color pipeline on a list of segments, apply de-duplication,
    and retain an rgb_map aligned with the retained phrases.
    """
    if not group_segments:
        return set(), set(), {}

    tones, phrases, rgb_map = aggregate_color_phrase_results(
        segments=group_segments,
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        expression_map=expression_map,
        all_webcolor_names=all_webcolor_names,
        llm_client=llm_client,
        cache=None,
        debug=debug,
    )

    def _apply_dedup(items: set[str]) -> set[str]:
        singles = {x for x in items if " " not in x}
        comps = {x for x in items if " " in x}
        singles = _dedup_singles_against_compounds(
            singles, comps, known_modifiers, known_tones
        )
        return singles | comps

    tones = _apply_dedup(set(tones))
    phrases = _apply_dedup(set(phrases))
    rgb_map = {k: _coerce_rgb(v) for k, v in rgb_map.items() if k in phrases}
    return tones, phrases, rgb_map


def _bundle_palette(
    primary_phrase: str | None,
    primary_rgb: tuple[int, int, int] | None,
    phrases_set: set[str],
    palette: dict[str, tuple[int, int, int]],
    top_k: int,
) -> list[dict[str, tuple[int, int, int]]]:
    """Build [primary(name+rgb), alt1(name+rgb), ...] without distances."""
    if not primary_phrase or not primary_rgb:
        return []
    exclude = set(phrases_set) | {primary_phrase}
    alts = _top_k_nearest(primary_rgb, palette, exclude=exclude, k=top_k)
    return [{"name": primary_phrase, "rgb": primary_rgb}] + alts


# =============================================================================
# Public APIs
# =============================================================================


def analyze_colors_with_sentiment(
    text: str, *, top_k: int = 12, debug: bool = False
) -> dict[str, list[dict[str, tuple[int, int, int]]]]:
    """Preference mode:
    Returns:
      {
        "positif": [{"name": <primary_like>, "rgb": (...) }, ...],
        "negatif": [{"name": <primary_dislike>, "rgb": (...) }, ...]
      }
    """
    clauses = analyze_sentence_sentiment(text)
    pos_clauses = [c["clause"] for c in clauses if c.get("polarity") == "positive"]
    neg_clauses = [c["clause"] for c in clauses if c.get("polarity") == "negative"]

    llm_client = get_llm_client(debug=debug)
    pos_tones, pos_phrases, _ = _extract_group_colors(pos_clauses, llm_client, debug)
    neg_tones, neg_phrases, _ = _extract_group_colors(neg_clauses, llm_client, debug)

    pos_primary = _pick_primary(pos_tones, pos_phrases)
    neg_primary = _pick_primary(neg_tones, neg_phrases)

    pos_rgb = _resolve_rgb_for_phrase(pos_primary, llm_client, debug=debug)
    neg_rgb = _resolve_rgb_for_phrase(neg_primary, llm_client, debug=debug)

    palette = _get_known_palette_cached()
    positif = _bundle_palette(pos_primary, pos_rgb, pos_phrases, palette, top_k)
    negatif = _bundle_palette(neg_primary, neg_rgb, neg_phrases, palette, top_k)

    positif = _map_name_rgb_list(positif)
    negatif = _map_name_rgb_list(negatif)

    # Intra-lists: dedupe by normalized names
    positif = _drop_items_with_none_rgb(_unique_by_name(positif))
    negatif = _drop_items_with_none_rgb(_unique_by_name(negatif))

    # Cross-lists: remove negatives that are the same (normalized) as positives
    negatif = _dedupe_cross_lists(positif, negatif)
    return {"positif": positif, "negatif": negatif}


def analyze_colors(
    text: str, debug: bool = False, *, top_k: int = 12
) -> dict[str, object]:
    """Legacy “single block” mode (primary + alternatives).
    Also returns tones/phrases/rgb_map for integration/debug.
    """
    llm_client = get_llm_client(debug=debug)

    tones, phrases, rgb_map = aggregate_color_phrase_results(
        segments=[text],
        known_modifiers=known_modifiers,
        known_tones=known_tones,
        expression_map=expression_map,
        all_webcolor_names=all_webcolor_names,
        llm_client=llm_client,
        cache=None,
        debug=debug,
    )

    def _apply_dedup(items: set[str]) -> set[str]:
        singles = {t for t in items if " " not in t}
        comps = {t for t in items if " " in t}
        singles = _dedup_singles_against_compounds(
            singles, comps, known_modifiers, known_tones
        )
        return singles | comps

    tones = _apply_dedup(set(tones))
    phrases = _apply_dedup(set(phrases))
    rgb_map = {k: _coerce_rgb(v) for k, v in rgb_map.items() if k in phrases}

    primary_phrase = _pick_primary(tones, phrases)
    primary_rgb = _resolve_rgb_for_phrase(primary_phrase, llm_client, debug=debug)

    alternatives: list[dict[str, tuple[int, int, int]]] = []
    if primary_rgb:
        palette = _get_known_palette_cached()
        exclude = set(phrases) | ({primary_phrase} if primary_phrase else set())
        alternatives = _top_k_nearest(primary_rgb, palette, exclude=exclude, k=top_k)

    if primary_rgb:
        primary_rgb = _coerce_rgb(primary_rgb)
    alternatives = _map_name_rgb_list(alternatives)

    return {
        "primary": {"phrase": primary_phrase, "rgb": primary_rgb},
        "alternatives": alternatives,
        "tones": tones,
        "phrases": list(phrases),
        "rgb_map": rgb_map,
    }


# =============================================================================
# Demo / Runtime entry
# =============================================================================
if __name__ == "__main__":
    # Load env + basic context logs only at runtime (no import side effects)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    logger.info("Python: %s", sys.executable)
    logger.info(
        "OpenRouter API key present: %s",
        "yes" if os.getenv("OPENROUTER_API_KEY") else "no",
    )

    _enable_glued_guard_if_requested()

    # Demo — preferences
    print(
        analyze_colors_with_sentiment(
            "I love bright red but I hate purple", top_k=10, debug=True
        )
    )
    # Demo — legacy
    # print(analyze_colors("Soft dusty rose glow lipstick", debug=True))
