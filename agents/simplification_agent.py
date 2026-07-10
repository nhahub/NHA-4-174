"""
agents/simplification_agent.py
────────────────────────────────
AGENT 2 — Simplifies complex text into plain language.

INPUT:  Transcript text
OUTPUT: Simplified text
"""

import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

import config


class SimplificationAgent:
    """
    Agent 2: Text → Simplified Text

    Responsibilities:
      1. Load FLAN-T5 (pretrained, or YOUR trained checkpoint)
      2. Accept transcript text
      3. Simplify text in chunks (for long text)
      4. Return simplified text
    """

    def __init__(self):
        self.config = config.SIMPLIFICATION_CONFIG
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Load the simplification model.

        To use YOUR trained model:
          1. Set use_pretrained = False in config.py
          2. Save your model with model.save_pretrained() into
             models/simplification_model/
        """
        print(f"[SimplificationAgent] Loading model...")

        if self.config["use_pretrained"]:
            print(f"[SimplificationAgent] Loading pretrained: {self.config['model_name']}")
            self.model = pipeline(
                "text2text-generation",
                model=self.config["model_name"],
                framework="pt"
            )
            print(f"[SimplificationAgent] ✓ Loaded pretrained model")
            return

        model_path = self.config["model_path"]

        if not os.path.exists(model_path) or not os.listdir(model_path):
            print(f"[SimplificationAgent] ⚠ Custom model not found at: {model_path}")
            print(f"[SimplificationAgent]   Falling back to pretrained")
            self.model = pipeline(
                "text2text-generation",
                model=self.config["model_name"],
                framework="pt"
            )
            return

        try:
            print(f"[SimplificationAgent] Loading YOUR model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_raw = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model = pipeline(
                "text2text-generation",
                model=model_raw,
                tokenizer=self.tokenizer,
                framework="pt"
            )
            print(f"[SimplificationAgent] ✓ Loaded YOUR trained model")
        except Exception as e:
            print(f"[SimplificationAgent] ✗ Error loading custom model: {e}")
            print(f"[SimplificationAgent]   Falling back to pretrained")
            self.model = pipeline(
                "text2text-generation",
                model=self.config["model_name"],
                framework="pt"
            )

    def simplify(self, transcript):
        """
        Simplify the transcript text.

        Args:
            transcript (str): Original transcript text

        Returns:
            dict: {
                "simplified_text": str,
                "original_words": int,
                "simplified_words": int,
                "chunks_processed": int,
            }
        """
        print(f"\n[SimplificationAgent] Simplifying text...")

        chunks = self._split_into_chunks(transcript)
        print(f"[SimplificationAgent]   Split into {len(chunks)} chunk(s)")

        simplified_chunks = [
            self._simplify_chunk(chunk, i + 1, len(chunks))
            for i, chunk in enumerate(chunks)
        ]

        simplified_text = " ".join(simplified_chunks)

        output = {
            "simplified_text": simplified_text,
            "original_words": len(transcript.split()),
            "simplified_words": len(simplified_text.split()),
            "chunks_processed": len(chunks),
        }

        print(f"[SimplificationAgent] ✓ {output['original_words']} → {output['simplified_words']} words")

        return output

    def _split_into_chunks(self, text):
        """Split text into ~chunk_size-word pieces at sentence boundaries."""
        chunk_size = self.config["chunk_size"]

        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

        chunks, current = [], ""
        for sentence in sentences:
            if len((current + " " + sentence).split()) > chunk_size and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    def _simplify_chunk(self, chunk, chunk_num, total_chunks):
        """Run one chunk through the simplification model."""
        prompt = self._build_prompt(chunk)

        result = self.model(
            prompt,
            max_length=self.config["max_length"],
            min_length=self.config["min_length"],
            num_beams=self.config["num_beams"],
            do_sample=False,
        )

        simplified = result[0]["generated_text"]
        print(f"[SimplificationAgent]   ✓ Chunk {chunk_num}/{total_chunks} simplified")

        return simplified

    def _build_prompt(self, text):
        """
        Build the instruction prompt.
        If your trained model expects a different prompt format, adjust here.
        """
        return (
            "Rewrite this text in very simple English that is easy to understand. "
            "Use short sentences. Use common everyday words. "
            "Keep the same meaning.\n\n"
            f"Text: {text}"
        )

    def save_result(self, output, save_path=None):
        """Save the simplification result to a JSON file."""
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "simplified.json")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[SimplificationAgent] ✓ Saved to {save_path}")


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

def test_agent():
    print("\n" + "=" * 60)
    print("TESTING SIMPLIFICATION AGENT")
    print("=" * 60)

    agent = SimplificationAgent()

    test_text = """
    The implementation of advanced machine learning algorithms necessitates
    a comprehensive understanding of statistical methodologies and computational
    paradigms. Furthermore, the optimization of hyperparameters requires
    iterative experimentation with various configurations to achieve
    satisfactory performance metrics.
    """

    print("\nOriginal text:")
    print(test_text)

    result = agent.simplify(test_text.strip())

    print("\n" + "-" * 60)
    print(f"Simplified: {result['simplified_text']}")
    print(f"\nWord count: {result['original_words']} → {result['simplified_words']}")

    agent.save_result(result)
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_agent()
