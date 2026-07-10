"""
agents/transcription_agent.py
──────────────────────────────
AGENT 1 — Converts audio to text using Whisper (or your fine-tuned model).

INPUT:  Audio file path
OUTPUT: Transcript text + metadata
"""

import os
import whisper
import librosa
import json

import config


class TranscriptionAgent:
    """
    Agent 1: Audio → Text

    Responsibilities:
      1. Load Whisper (base pretrained, or YOUR fine-tuned checkpoint)
      2. Accept an audio file path
      3. Transcribe the audio
      4. Return transcript + confidence scores
    """

    def __init__(self):
        self.config = config.TRANSCRIPTION_CONFIG
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load the transcription model.

        To use YOUR fine-tuned Whisper checkpoint:
          1. Set use_pretrained = False in config.py
          2. Place your model files in models/whisper_finetuned/
          3. whisper.load_model() also accepts a path to a .pt checkpoint file
        """
        print(f"[TranscriptionAgent] Loading model...")

        if self.config["use_pretrained"]:
            model_size = self.config["model_size"]
            self.model = whisper.load_model(model_size)
            print(f"[TranscriptionAgent] ✓ Loaded Whisper-{model_size}")
        else:
            model_path = self.config["model_path"]
            checkpoint_file = None

            if os.path.isdir(model_path):
                for f in os.listdir(model_path):
                    if f.endswith(".pt"):
                        checkpoint_file = os.path.join(model_path, f)
                        break

            if checkpoint_file and os.path.exists(checkpoint_file):
                self.model = whisper.load_model(checkpoint_file)
                print(f"[TranscriptionAgent] ✓ Loaded custom checkpoint: {checkpoint_file}")
            else:
                print(f"[TranscriptionAgent] ⚠ Custom model not found in {model_path}")
                print(f"[TranscriptionAgent]   Falling back to Whisper-{self.config['model_size']}")
                self.model = whisper.load_model(self.config["model_size"])

    def transcribe(self, audio_path):
        """
        Convert an audio file to text.

        Args:
            audio_path (str): Path to audio file (.wav, .mp3, etc.)

        Returns:
            dict: {
                "transcript": str,
                "segments": list,
                "language": str,
                "confidence": float,
                "word_count": int,
            }
        """
        print(f"\n[TranscriptionAgent] Processing: {audio_path}")

        audio = self._load_audio(audio_path)

        result = self.model.transcribe(
            audio,
            language=self.config["language"] if self.config["language"] != "auto" else None,
            fp16=False,   # fp32 for CPU safety; set True if you have a GPU
        )

        confidence = self._calculate_confidence(result.get("segments", []))

        output = {
            "transcript": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "confidence": confidence,
            "word_count": len(result["text"].split()),
        }

        print(f"[TranscriptionAgent] ✓ {output['word_count']} words, "
              f"language={output['language']}, confidence={confidence:.1%}")

        return output

    def _load_audio(self, audio_path):
        """Load audio and resample to 16kHz (required by Whisper)."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = librosa.load(audio_path, sr=self.config["sample_rate"])

        duration = len(audio) / sr
        print(f"[TranscriptionAgent]   Audio: {duration:.1f}s @ {sr}Hz")

        return audio

    def _calculate_confidence(self, segments):
        """
        Approximate overall confidence from Whisper's per-segment scores.
        Filters out likely-silence segments, then averages avg_logprob.
        """
        if not segments:
            return 0.0

        valid = [s for s in segments if s.get("no_speech_prob", 1.0) < 0.6]
        if not valid:
            return 0.0

        avg_logprob = sum(s.get("avg_logprob", -1) for s in valid) / len(valid)
        return max(0.0, min(1.0, avg_logprob + 1.0))

    def save_result(self, output, save_path=None):
        """Save the transcription result to a JSON file."""
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "transcript.json")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[TranscriptionAgent] ✓ Saved to {save_path}")


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

def test_agent():
    print("\n" + "=" * 60)
    print("TESTING TRANSCRIPTION AGENT")
    print("=" * 60)

    agent = TranscriptionAgent()
    test_audio_path = os.path.join(config.INPUT_DIR, "test_audio.wav")

    if not os.path.exists(test_audio_path):
        print(f"\n⚠ Test audio not found: {test_audio_path}")
        print("Please place a test .wav file there.")
        return

    result = agent.transcribe(test_audio_path)

    print("\n" + "-" * 60)
    print(f"Transcript: {result['transcript'][:200]}...")
    print(f"Word count: {result['word_count']}")
    print(f"Language:   {result['language']}")
    print(f"Confidence: {result['confidence']:.2%}")

    agent.save_result(result)
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_agent()
