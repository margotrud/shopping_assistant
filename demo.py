# demo.py
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Chatbot.extraction.orchestrator import analyze_colors_with_sentiment

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python demo.py "I love bright red but I hate purple" [--debug]')
        raise SystemExit(2)
    debug = "--debug" in sys.argv
    text = " ".join(a for a in sys.argv[1:] if a != "--debug")
    out = analyze_colors_with_sentiment(text, top_k=10, debug=debug)
    print(json.dumps(out, indent=2, ensure_ascii=False))
