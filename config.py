"""
config.py
─────────
All configuration settings for the Voxa 3-Agent System.

Change settings here — nothing else in the project needs to be touched.
"""

import os

# ══════════════════════════════════════════════════════════════
# PROJECT PATHS
# ══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths — point these to YOUR trained models
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRANSCRIPTION_MODEL_PATH = os.path.join(MODELS_DIR, "whisper_finetuned")
SIMPLIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "simplification_model")
SUMMARIZATION_MODEL_PATH = os.path.join(MODELS_DIR, "summarization_model")

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


# ══════════════════════════════════════════════════════════════
# AGENT 1 — TRANSCRIPTION SETTINGS
# ══════════════════════════════════════════════════════════════

TRANSCRIPTION_CONFIG = {
    "model_type": "whisper",            # "whisper" or "custom"
    "model_size": "base",                # "tiny", "base", "small", "medium", "large"
    "use_pretrained": True,             # False if using YOUR fine-tuned model
    "model_path": TRANSCRIPTION_MODEL_PATH,
    "language": "en",                    # "en", "ar", "auto"
    "sample_rate": 16000,                # Audio sample rate (Whisper needs 16kHz)
    "output_format": "json",
}


# ══════════════════════════════════════════════════════════════
# AGENT 2 — SIMPLIFICATION SETTINGS
# ══════════════════════════════════════════════════════════════

SIMPLIFICATION_CONFIG = {
    "model_type": "flan-t5",             # "flan-t5" or "custom"
    "model_name": "google/flan-t5-base", # Pretrained fallback model
    "use_pretrained": True,              # Set False to use YOUR trained model
    "model_path": SIMPLIFICATION_MODEL_PATH,
    "chunk_size": 80,                    # Words per chunk
    "max_length": 130,
    "min_length": 30,
    "num_beams": 4,
}


# ══════════════════════════════════════════════════════════════
# AGENT 3 — SUMMARIZATION SETTINGS
# ══════════════════════════════════════════════════════════════

SUMMARIZATION_CONFIG = {
    "model_type": "distilbart",
    "model_name": "sshleifer/distilbart-cnn-12-6",  # Pretrained fallback model
    "use_pretrained": True,              # Set False to use YOUR trained model
    "model_path": SUMMARIZATION_MODEL_PATH,
    "chunk_size": 500,
    "max_length": 130,
    "min_length": 30,
}


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR SETTINGS
# ══════════════════════════════════════════════════════════════

ORCHESTRATOR_CONFIG = {
    "save_intermediate_results": True,
    "output_format": "json",
    "verbose": True,
}


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI SETTINGS
# ══════════════════════════════════════════════════════════════

STREAMLIT_CONFIG = {
    "title": "Voxa — AI Speech-to-Text for Deaf/Hard-of-Hearing",
    "max_upload_size_mb": 200,
    "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    "theme": "light",
}


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

LOGGING_CONFIG = {
    "level": "INFO",
    "log_file": os.path.join(BASE_DIR, "voxa.log"),
    "console_output": True,
}
