from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Chatbot.extraction.orchestrator import analyze_colors_with_sentiment

def test_smoke():
    out = analyze_colors_with_sentiment("I love bright red but I hate purple", top_k=3, debug=False)
    assert isinstance(out, dict)
    assert "positif" in out and "negatif" in out
    for bucket in ("positif", "negatif"):
        assert isinstance(out[bucket], list)
        for item in out[bucket]:
            assert "name" in item and "rgb" in item
            rgb = item["rgb"]
            assert isinstance(rgb, (list, tuple)) and len(rgb) == 3
