# ── Boot / ENV ────────────────────────────────────────────────────────────────
import os, sys
print("[PYTHON]", sys.executable)
from dotenv import load_dotenv
load_dotenv()
print("API Key found:", bool(os.getenv("OPENROUTER_API_KEY")))

# ── Vocab & Config ───────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color.vocab import (
    all_webcolor_names,
    known_tones as _known_tones,
)
from color_sentiment_extractor.extraction.general.utils.load_config import load_config
from typing import Set, Tuple, Optional, List, Dict

known_modifiers: Set[str] = load_config("known_modifiers", mode="set")
known_tones: Set[str] = set(_known_tones or [])

# ── Pipelines / LLM client ───────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color.logic.color_pipeline import (
    aggregate_color_phrase_results,
)
from color_sentiment_extractor.extraction.color.llm.llm_api_client import get_llm_client
from color_sentiment_extractor.extraction.general.token.base_recovery import recover_base

# ── RGB utils ────────────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.color.utils.rgb_distance import (
    _try_simplified_match,
    rgb_distance,
)
from color_sentiment_extractor.extraction.color.logic.rgb_pipeline import (
    resolve_rgb_with_llm,
)

# ── Sentiment ────────────────────────────────────────────────────────────────
from color_sentiment_extractor.extraction.general.sentiment_core import (
    analyze_sentence_sentiment,
)


# =============================================================================
# Monkey-patch: guard pour split_glued_tokens (évite les stalls sur tokens courts)
# =============================================================================
def _force_guard_split_glued_tokens():
    try:
        import importlib
        m = importlib.import_module("extraction.color.token.split")
    except Exception:
        print("[patch] split_glued_tokens module NOT found")
        return

    if not hasattr(m, "split_glued_tokens"):
        print("[patch] split_glued_tokens symbol NOT found in module")
        return

    _orig = m.split_glued_tokens

    def _patched(token: str, *a, **kw):
        # Court-circuite les tokens très courts
        if not token or len(token) <= 2:
            return []
        return _orig(token, *a, **kw)

    m.split_glued_tokens = _patched
    print(f"[patch] split_glued_tokens guard ACTIVE in {m.__name__}")



# =============================================================================
# Helpers
# =============================================================================

def _coerce_rgb(rgb) -> Optional[Tuple[int, int, int]]:
    """Convertit IntegerRGB/objets similaires en tuple natif (r, g, b)."""
    if rgb is None:
        return None
    if isinstance(rgb, tuple) and len(rgb) == 3:
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    r = getattr(rgb, "red", None)
    g = getattr(rgb, "green", None)
    b = getattr(rgb, "blue", None)
    if None not in (r, g, b):
        return (int(r), int(g), int(b))
    try:
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:
        return None


def _map_name_rgb_list(items: List[Dict[str, object]]) -> List[Dict[str, Tuple[int, int, int]]]:
    """Applique _coerce_rgb sur une liste [{'name':..., 'rgb':...}, ...]."""
    out: List[Dict[str, Tuple[int, int, int]]] = []
    for d in items:
        out.append({"name": d["name"], "rgb": _coerce_rgb(d["rgb"])})
    return out


def _unique_by_name(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Déduplique en préservant l'ordre d'apparition (clé = 'name')."""
    seen: Set[str] = set()
    out: List[Dict[str, object]] = []
    for it in items:
        name = (it.get("name") if isinstance(it, dict) else None) or ""
        if name not in seen:
            seen.add(name)
            out.append(it)
    return out


def _drop_items_with_none_rgb(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Supprime les entrées dont le rgb est None (ou mal formé)."""
    out: List[Dict[str, object]] = []
    for it in items:
        rgb = it.get("rgb") if isinstance(it, dict) else None
        if isinstance(rgb, tuple) and len(rgb) == 3 and all(isinstance(c, int) for c in rgb):
            out.append(it)
    return out


def _dedupe_cross_lists(
    positives: List[Dict[str, object]],
    negatives: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Supprime des négatifs les couleurs déjà présentes en positifs (priorité au like).
    """
    pos_names = {p["name"] for p in positives if isinstance(p, dict) and "name" in p}
    return [n for n in negatives if (isinstance(n, dict) and n.get("name") not in pos_names)]


def _dedup_singles_against_compounds(
    singles: Set[str],
    compounds: Set[str],
    known_mods: Set[str],
    known_tns: Set[str],
) -> Set[str]:
    """
    Supprime les singletons couverts par au moins un composé (surface + recover_base).
    """
    covered: Set[str] = set()
    for comp in compounds:
        for w in comp.split():
            covered.add(w)
            base = recover_base(
                w,
                known_modifiers=known_mods,
                known_tones=known_tns,
                fuzzy_fallback=False
            )
            if base:
                covered.add(base)
    return {s for s in singles if s not in covered}


def _build_known_palette(names: Set[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Construit une palette {name -> (r,g,b)} à partir des noms CSS/XKCD connus.
    """
    palette: Dict[str, Tuple[int, int, int]] = {}
    for nm in names:
        rgb = _try_simplified_match(nm, debug=False)
        if rgb:
            palette[nm] = _coerce_rgb(rgb)  # force tuple
    return palette

from functools import lru_cache

@lru_cache(maxsize=1)
def _get_known_palette_cached() -> Dict[str, Tuple[int,int,int]]:
    return _build_known_palette(set(all_webcolor_names))

def _top_k_nearest(
    target_rgb: Tuple[int, int, int],
    palette: Dict[str, Tuple[int, int, int]],
    exclude: Set[str],
    k: int,
) -> List[Dict[str, Tuple[int, int, int]]]:
    """
    Renvoie les k couleurs les plus proches (name + rgb), sans distances.
    """
    scored = []
    for name, rgb in palette.items():
        if name in exclude:
            continue
        d = rgb_distance(target_rgb, rgb)
        scored.append((d, name, rgb))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [{"name": name, "rgb": rgb} for _, name, rgb in scored[:k]]


def _pick_primary(tones: Set[str], phrases: Set[str]) -> Optional[str]:
    """
    Choisit une « primary »: priorité aux composés, sinon premier trié.
    """
    if tones:
        comps = [t for t in tones if " " in t]
        return (sorted(comps, key=len, reverse=True)[0] if comps else sorted(tones)[0])
    if phrases:
        comps = [p for p in phrases if " " in p]
        return (sorted(comps, key=len, reverse=True)[0] if comps else sorted(phrases)[0])
    return None


def _resolve_rgb_for_phrase(phrase: Optional[str], llm_client, debug: bool = False) -> Optional[Tuple[int, int, int]]:
    """
    Résout un RGB robuste (CSS/XKCD d’abord, puis fallback LLM en DB-first).
    """
    if not phrase:
        return None
    rgb = _try_simplified_match(phrase, debug=debug)
    if rgb:
        return _coerce_rgb(rgb)
    rgb = resolve_rgb_with_llm(
        phrase,
        llm_client=llm_client,
        cache=None,
        debug=debug,
        prefer_db_first=True
    )
    return _coerce_rgb(rgb)


def _extract_group_colors(
    group_segments: List[str],
    llm_client,
    debug: bool = False
) -> Tuple[Set[str], Set[str], Dict[str, Tuple[int, int, int]]]:
    """
    Exécute le color pipeline sur une liste de segments, applique la dédup, et
    garde un rgb_map aligné sur les phrases retenues.
    """
    if not group_segments:
        return set(), set(), {}

    tones, phrases, rgb_map = aggregate_color_phrase_results(
        segments=group_segments,
        known_modifiers=known_modifiers,
        all_webcolor_names=all_webcolor_names,
        llm_client=llm_client,
        cache=None,
        debug=debug,
    )

    def _apply_dedup(items: Set[str]) -> Set[str]:
        singles = {x for x in items if " " not in x}
        comps   = {x for x in items if " " in x}
        singles = _dedup_singles_against_compounds(singles, comps, known_modifiers, known_tones)
        return singles | comps

    tones   = _apply_dedup(set(tones))
    phrases = _apply_dedup(set(phrases))
    # Normalise rgb_map et ne garde que les phrases retenues
    rgb_map = {k: _coerce_rgb(v) for k, v in rgb_map.items() if k in phrases}
    return tones, phrases, rgb_map


def _bundle_palette(
    primary_phrase: Optional[str],
    primary_rgb: Optional[Tuple[int, int, int]],
    phrases_set: Set[str],
    palette: Dict[str, Tuple[int, int, int]],
    top_k: int,
) -> List[Dict[str, Tuple[int, int, int]]]:
    """
    Construit la liste: [primary(name+rgb), alt1(name+rgb), ...] sans distances.
    """
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
) -> Dict[str, List[Dict[str, Tuple[int, int, int]]]]:
    """
    Mode PRÉFÉRENCES (toujours dispo) :
    Retourne deux listes:
      {
        "positif": [{ "name": <primary_like>, "rgb": (...) }, { "name": <alt1>, "rgb": (...) }, ...],
        "negatif": [{ "name": <primary_dislike>, "rgb": (...) }, { "name": <alt1>, "rgb": (...) }, ...]
      }
    """
    # 1) Clauses + polarités
    clauses = analyze_sentence_sentiment(text)  # [{clause, polarity, ...}]
    pos_clauses = [c["clause"] for c in clauses if c.get("polarity") == "positive"]
    neg_clauses = [c["clause"] for c in clauses if c.get("polarity") == "negative"]

    # 2) Extraction couleurs par polarité
    llm_client = get_llm_client(debug=debug)
    pos_tones, pos_phrases, _ = _extract_group_colors(pos_clauses, llm_client, debug)
    neg_tones, neg_phrases, _ = _extract_group_colors(neg_clauses, llm_client, debug)

    # 3) Primary + RGB
    pos_primary = _pick_primary(pos_tones, pos_phrases)
    neg_primary = _pick_primary(neg_tones, neg_phrases)

    pos_rgb = _resolve_rgb_for_phrase(pos_primary, llm_client, debug=debug)
    neg_rgb = _resolve_rgb_for_phrase(neg_primary, llm_client, debug=debug)

    # 4) Alternatives proches (name+rgb), pas de distances
    palette = _get_known_palette_cached()
    positif = _bundle_palette(pos_primary, pos_rgb, pos_phrases, palette, top_k)
    negatif = _bundle_palette(neg_primary, neg_rgb, neg_phrases, palette, top_k)

    # 5) Normalise
    positif = _map_name_rgb_list(positif)
    negatif = _map_name_rgb_list(negatif)

    # 6) Dédupe intra-listes + drop None
    positif = _drop_items_with_none_rgb(_unique_by_name(positif))
    negatif = _drop_items_with_none_rgb(_unique_by_name(negatif))

    # 7) ✅ Dédupe inter-listes (positif > négatif)
    pos_names = {p["name"].strip().lower() for p in positif}
    negatif = [n for n in negatif if n["name"].strip().lower() not in pos_names]


    return {"positif": positif, "negatif": negatif}


def analyze_colors(
    text: str, debug: bool = False, *, top_k: int = 12
) -> Dict[str, object]:
    """
    Mode legacy « un seul bloc » (primary + alternatives).
    Retourne aussi tones/phrases/rgb_map pour debug/intégration.
    """
    llm_client = get_llm_client(debug=debug)

    tones, phrases, rgb_map = aggregate_color_phrase_results(
        segments=[text],
        known_modifiers=known_modifiers,
        all_webcolor_names=all_webcolor_names,
        llm_client=llm_client,
        cache=None,
        debug=debug,
    )

    def _apply_dedup(items: Set[str]) -> Set[str]:
        singles = {t for t in items if " " not in t}
        comps   = {t for t in items if " " in t}
        singles = _dedup_singles_against_compounds(singles, comps, known_modifiers, known_tones)
        return singles | comps

    tones   = _apply_dedup(set(tones))
    phrases = _apply_dedup(set(phrases))
    rgb_map = {k: _coerce_rgb(v) for k, v in rgb_map.items() if k in phrases}

    primary_phrase = _pick_primary(tones, phrases)
    primary_rgb = _resolve_rgb_for_phrase(primary_phrase, llm_client, debug=debug)

    alternatives: List[Dict[str, Tuple[int, int, int]]] = []
    if primary_rgb:
        palette = _get_known_palette_cached()
        exclude = set(phrases) | ({primary_phrase} if primary_phrase else set())
        alternatives = _top_k_nearest(primary_rgb, palette, exclude=exclude, k=top_k)

    # Normalisation finale
    if primary_rgb:
        primary_rgb = _coerce_rgb(primary_rgb)
    alternatives = _map_name_rgb_list(alternatives)

    return {
        "primary": {"phrase": primary_phrase, "rgb": primary_rgb},
        "alternatives": alternatives,   # juste name + rgb (tuples)
        "tones": tones,
        "phrases": list(phrases),
        "rgb_map": rgb_map,
    }


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    # Démo préférences
    print(analyze_colors_with_sentiment("I love bright red but I hate purple", top_k=10, debug=True))
    # Démo legacy
    # print(analyze_colors("Soft dusty rose glow lipstick", debug=True))
