"""
core/orchestrator.py
─────────────────────
PERSON 4'S FILE

The Orchestrator connects all 3 agents and runs them in order:
  Audio → TranscriptionAgent → SimplificationAgent + SummarizationAgent

It does NOT do any AI work — it only coordinates the 3 agents built by
Person 1, Person 2, and Person 3.
"""

import os
import json
import time
import sys

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.transcription_agent import TranscriptionAgent
from agents.simplification_agent import SimplificationAgent
from agents.summarization_agent import SummarizationAgent


class VoxaOrchestrator:
    """
    Coordinates the 3-agent Voxa pipeline.

    Usage:
        orch   = VoxaOrchestrator()
        result = orch.run("data/input/my_audio.wav")

        print(result["transcript"])
        print(result["simplified_text"])
        print(result["summary"])
    """

    def __init__(self):
        print("\n" + "═" * 58)
        print("  Initializing Voxa Orchestrator")
        print("═" * 58)

        # Load all 3 agents once, at startup (not on every run)
        self.transcription_agent  = TranscriptionAgent()
        self.simplification_agent = SimplificationAgent()
        self.summarization_agent  = SummarizationAgent()

        self.verbose = config.ORCHESTRATOR_CONFIG["verbose"]

        print("═" * 58)
        print("  ✓ All 3 agents ready")
        print("═" * 58 + "\n")

    def run(self, audio_path: str) -> dict:
        """
        Run the complete 3-agent pipeline on one audio file.

        Args:
            audio_path (str): Path to the audio file to process

        Returns:
            dict: {
                "transcript":       str,
                "simplified_text":  str,
                "summary":          str,
                "transcription_meta":  dict,   # confidence, language, etc.
                "simplification_meta": dict,   # word counts
                "summarization_meta":  dict,   # word counts, compression
                "elapsed_seconds":  float,
                "success": bool,
                "error": str or None,
            }
        """
        start = time.time()

        try:
            # ── Stage 1: Transcription ────────────────────────────────
            self._log("Stage 1/3 → TranscriptionAgent")
            transcription_result = self.transcription_agent.transcribe(audio_path)
            transcript = transcription_result["transcript"]

            if not transcript.strip():
                return self._error_result("Transcription produced empty text.")

            # ── Stage 2: Simplification ───────────────────────────────
            self._log("Stage 2/3 → SimplificationAgent")
            simplification_result = self.simplification_agent.simplify(transcript)

            # ── Stage 3: Summarization ────────────────────────────────
            self._log("Stage 3/3 → SummarizationAgent")
            summarization_result = self.summarization_agent.summarize(transcript)

            elapsed = round(time.time() - start, 2)

            # ── Consolidate results ───────────────────────────────────
            final_result = {
                "success": True,
                "error": None,
                "transcript":       transcript,
                "simplified_text":  simplification_result["simplified_text"],
                "summary":          summarization_result["summary"],
                "transcription_meta": {
                    "language":   transcription_result.get("language"),
                    "confidence": transcription_result.get("confidence"),
                    "word_count": transcription_result.get("word_count"),
                },
                "simplification_meta": {
                    "original_words":   simplification_result["original_words"],
                    "simplified_words": simplification_result["simplified_words"],
                },
                "summarization_meta": {
                    "original_words":    summarization_result["original_words"],
                    "summary_words":     summarization_result["summary_words"],
                    "compression_ratio": summarization_result["compression_ratio"],
                },
                "elapsed_seconds": elapsed,
            }

            self._print_summary(final_result)
            self._save_result(final_result, audio_path)

            return final_result

        except Exception as exc:
            return self._error_result(str(exc))

    # ── Internal Helpers ──────────────────────────────────────────────

    def _log(self, message: str):
        if self.verbose:
            print(f"\n[Orchestrator] {message}")

    def _error_result(self, error_message: str) -> dict:
        print(f"\n[Orchestrator] ✗ ERROR: {error_message}")
        return {
            "success": False,
            "error": error_message,
            "transcript": "",
            "simplified_text": "",
            "summary": "",
        }

    def _print_summary(self, result: dict):
        print("\n" + "═" * 58)
        print(f"  PIPELINE COMPLETE — {result['elapsed_seconds']}s")
        print("═" * 58)
        print(f"  Transcript words   : {result['transcription_meta']['word_count']}")
        print(f"  Simplified words   : {result['simplification_meta']['simplified_words']}")
        print(f"  Summary words      : {result['summarization_meta']['summary_words']}")
        print(f"  Confidence         : {result['transcription_meta']['confidence']:.1%}")
        print("═" * 58 + "\n")

    def _save_result(self, result: dict, audio_path: str):
        """Save the consolidated result as JSON in data/output/."""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        filename = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = os.path.join(config.OUTPUT_DIR, f"{filename}_result.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[Orchestrator] ✓ Full result saved → {save_path}")


# ══════════════════════════════════════════════════════════════
# TEST FUNCTION — PERSON 4 can run this standalone
# ══════════════════════════════════════════════════════════════

def test_orchestrator():
    """
    Test the full pipeline end-to-end.

    Usage:
        python core/orchestrator.py
    """
    print("\n" + "="*60)
    print("TESTING FULL ORCHESTRATOR PIPELINE")
    print("="*60)

    orch = VoxaOrchestrator()

    test_audio = os.path.join(config.INPUT_DIR, "test_audio.wav")

    if not os.path.exists(test_audio):
        print(f"\n⚠ Test audio not found: {test_audio}")
        print("Please place a test .wav file there.")
        return

    result = orch.run(test_audio)

    if result["success"]:
        print("\n✓ Pipeline test passed!")
    else:
        print(f"\n✗ Pipeline test failed: {result['error']}")


if __name__ == "__main__":
    test_orchestrator()
