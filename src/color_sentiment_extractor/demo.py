# src/color_sentiment_extractor/demo.py
import json
import argparse

def main():
    # 👉 adapte l'import si besoin selon l’emplacement réel
    from .extraction.orchestrator import analyze_colors_with_sentiment

    parser = argparse.ArgumentParser(
        prog="cse-demo",
        description="Analyze text: extract color mentions, split by sentiment, resolve RGB."
    )
    parser.add_argument("text", nargs="*", help='Text to analyze (e.g. I love bright red but I hate purple)')
    parser.add_argument("--debug", action="store_true", help="Verbose debug logs")
    parser.add_argument("--top-k", type=int, default=10, dest="top_k", help="Max color phrases to return")

    args = parser.parse_args()
    text = " ".join(args.text) or "I love bright red but I hate purple"

    out = analyze_colors_with_sentiment(text, top_k=args.top_k, debug=args.debug)
    print(json.dumps(out, indent=2, ensure_ascii=False))
