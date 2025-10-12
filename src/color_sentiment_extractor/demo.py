# src/color_sentiment_extractor/demo.py
import argparse
import json
import sys


def main():
    """CLI demo: analyze text, extract color mentions, split by sentiment, and resolve RGB."""
    from .extraction.orchestrator import analyze_colors_with_sentiment

    parser = argparse.ArgumentParser(
        prog="cse-demo",
        description="Analyze text: extract color mentions, split by sentiment, resolve RGB.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to analyze (e.g. I love bright red but I hate purple)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debug logs")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top_k",
        help="Max color phrases to return",
    )

    args = parser.parse_args()
    text = " ".join(args.text) or "I love bright red but I hate purple"

    try:
        result = analyze_colors_with_sentiment(text, top_k=args.top_k, debug=args.debug)
        print("\n🎨 Color Sentiment Extraction Result:\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
