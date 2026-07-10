"""
run_pipeline.py
────────────────
PERSON 4'S FILE

Command-line entry point for running the Voxa pipeline without the UI.

Usage:
    python run_pipeline.py data/input/my_audio.wav
    python run_pipeline.py data/input/my_audio.wav --output results.json
"""

import sys
import os
import argparse
import json

from core.orchestrator import VoxaOrchestrator
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Voxa 3-agent pipeline on an audio file."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file (.wav, .mp3, .flac, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom path to save the JSON result"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.audio_path):
        print(f"✗ Error: File not found: {args.audio_path}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  VOXA — 3-AGENT PIPELINE (Command Line)")
    print("═" * 60)

    # Build the orchestrator (loads all 3 models once)
    orchestrator = VoxaOrchestrator()

    # Run the pipeline
    result = orchestrator.run(args.audio_path)

    if not result["success"]:
        print(f"\n✗ Pipeline failed: {result['error']}")
        sys.exit(1)

    # Print final results to terminal
    print("\n📝 TRANSCRIPT:")
    print("-" * 60)
    print(result["transcript"])

    print("\n✨ SIMPLIFIED TEXT:")
    print("-" * 60)
    print(result["simplified_text"])

    print("\n📊 SUMMARY:")
    print("-" * 60)
    print(result["summary"])

    # Save to custom path if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Result also saved to: {args.output}")

    print(f"\n⏱  Total time: {result['elapsed_seconds']}s")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
