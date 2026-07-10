"""
agents/summarization_agent.py
────────────────────────────────
PERSON 3'S FILE

This agent condenses transcript text into a short summary using YOUR trained model.

INPUT:  Transcript text
OUTPUT: Summary text
"""

import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import config


class SummarizationAgent:
    """
    Agent 3: Text → Summary

    Responsibilities:
      1. Load YOUR trained DistilBART (or similar) summarization model
      2. Accept transcript text
      3. Summarize text in chunks (for long text)
      4. Return summary
    """

    def __init__(self):
        self.config = config.SUMMARIZATION_CONFIG
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Load the summarization model.

        TODO FOR PERSON 3:
        If you have YOUR trained model:
          1. Set use_pretrained = False in config.py
          2. Place your model in models/summarization_model/
          3. The model loads from model_path below
        """
        print(f"[SummarizationAgent] Loading model...")

        if self.config["use_pretrained"]:
            print(f"[SummarizationAgent] Loading pretrained: {self.config['model_name']}")
            self.model = pipeline(
                "summarization",
                model=self.config["model_name"],
                framework="pt"
            )
            print(f"[SummarizationAgent] ✓ Loaded pretrained model")

        else:
            model_path = self.config["model_path"]

            if not os.path.exists(model_path):
                print(f"[SummarizationAgent] ⚠ Custom model not found at: {model_path}")
                print(f"[SummarizationAgent] Falling back to pretrained")
                self.model = pipeline(
                    "summarization",
                    model=self.config["model_name"],
                    framework="pt"
                )
                return

            print(f"[SummarizationAgent] Loading YOUR model from: {model_path}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model_raw = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                self.model = pipeline(
                    "summarization",
                    model=self.model_raw,
                    tokenizer=self.tokenizer,
                    framework="pt"
                )
                print(f"[SummarizationAgent] ✓ Loaded YOUR trained model")

            except Exception as e:
                print(f"[SummarizationAgent] ✗ Error loading custom model: {e}")
                print(f"[SummarizationAgent] Falling back to pretrained")
                self.model = pipeline(
                    "summarization",
                    model=self.config["model_name"],
                    framework="pt"
                )

    def summarize(self, transcript):
        """
        Main function: Summarize the transcript text.

        Args:
            transcript (str): Original transcript text

        Returns:
            dict: {
                "summary": str,             # The condensed summary
                "original_words": int,      # Word count before
                "summary_words": int,       # Word count after
                "compression_ratio": float, # summary_words / original_words
                "chunks_processed": int,
            }
        """
        print(f"\n[SummarizationAgent] Summarizing text...")

        chunks = self._split_into_chunks(transcript)
        print(f"[SummarizationAgent]   Split into {len(chunks)} chunk(s)")

        summary_parts = []
        for i, chunk in enumerate(chunks):
            summarized = self._summarize_chunk(chunk, i + 1, len(chunks))
            summary_parts.append(summarized)

        summary = " ".join(summary_parts)

        output = {
            "summary": summary,
            "original_words": len(transcript.split()),
            "summary_words": len(summary.split()),
            "compression_ratio": round(
                len(summary.split()) / max(len(transcript.split()), 1), 3
            ),
            "chunks_processed": len(chunks),
        }

        print(f"[SummarizationAgent] ✓ Summarized: {output['original_words']} → {output['summary_words']} words")

        return output

    def _split_into_chunks(self, text):
        """
        Split text into word-count chunks (summarization models handle
        longer input than simplification models, so chunks are bigger).
        """
        chunk_size = self.config["chunk_size"]
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ] or [text]

    def _summarize_chunk(self, chunk, chunk_num, total_chunks):
        """
        Summarize one chunk of text using your trained model.
        """
        # Skip chunks too short to meaningfully summarize
        if len(chunk.split()) < 30:
            print(f"[SummarizationAgent]   ✓ Chunk {chunk_num}/{total_chunks} too short — kept as-is")
            return chunk

        result = self.model(
            chunk,
            max_length=self.config["max_length"],
            min_length=self.config["min_length"],
            do_sample=False,
        )

        summarized = result[0]["summary_text"]
        print(f"[SummarizationAgent]   ✓ Chunk {chunk_num}/{total_chunks} summarized")

        return summarized

    def save_result(self, output, save_path=None):
        """
        Save summarization result to file.
        """
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "summary.json")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[SummarizationAgent] ✓ Saved to {save_path}")


# ══════════════════════════════════════════════════════════════
# TEST FUNCTION — PERSON 3 can run this standalone
# ══════════════════════════════════════════════════════════════

def test_agent():
    """
    Test the summarization agent with sample text.

    Usage:
        python -c "from agents.summarization_agent import test_agent; test_agent()"
    """
    print("\n" + "="*60)
    print("TESTING SUMMARIZATION AGENT")
    print("="*60)

    agent = SummarizationAgent()

    test_text = """
    Once upon a time, in a kingdom far away, there lived a young princess
    named Aria. She was known throughout the land for her kindness and
    curiosity. One day, while exploring the royal gardens, she discovered
    a hidden door behind a wall of roses. Despite the warnings from her
    guardians, she decided to open it, revealing a magical world full of
    talking animals and enchanted trees. Aria spent many days in this
    world, learning valuable lessons about courage and friendship before
    finally returning home, forever changed by her adventure.
    """

    print("\nOriginal text:")
    print(test_text)

    result = agent.summarize(test_text.strip())

    print("\n" + "-"*60)
    print("RESULT:")
    print("-"*60)
    print(f"Summary: {result['summary']}")
    print(f"\nWord count: {result['original_words']} → {result['summary_words']}")
    print(f"Compression ratio: {result['compression_ratio']}")

    agent.save_result(result)

    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_agent()
