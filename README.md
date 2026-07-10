# 🎙️ Voxa — 3-Agent AI Pipeline with Streamlit UI

Converts spoken audio into accessible text for deaf and hard-of-hearing users.
Built with 3 independent AI agents + a simple web interface.

---

## Project Structure

```
voxa_team/
│
├── config.py                    ← All settings (model paths, chunk sizes, etc.)
├── requirements.txt              ← Exact pinned library versions
├── run_pipeline.py               ← Command-line runner (no UI)
│
├── models/                       ← Your trained models go here
│   ├── README.md                 ← Instructions for adding your models
│   ├── whisper_finetuned/
│   ├── simplification_model/
│   └── summarization_model/
│
├── agents/                       ← The 3 AI agents (one file each)
│   ├── transcription_agent.py    ← Agent 1: Audio → Text
│   ├── simplification_agent.py   ← Agent 2: Text → Simple Text
│   └── summarization_agent.py    ← Agent 3: Text → Summary
│
├── core/
│   └── orchestrator.py           ← Runs the 3 agents in order
│
├── ui/
│   └── streamlit_app.py          ← Web interface
│
└── data/
    ├── input/                    ← Put test audio files here
    └── output/                   ← Results are saved here automatically
```

---

## Setup — Step by Step

### 1. Install Python 3.10 or 3.11

Download from [python.org](https://www.python.org/downloads/) if you don't have it.
Check your version:
```bash
python --version
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked when you see `(venv)` at the start of your terminal line.

### 3. Install ffmpeg (required by Whisper)

Whisper needs `ffmpeg` installed on your system (not just pip):

```bash
# Windows (using Chocolatey)
choco install ffmpeg

# Mac (using Homebrew)
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg
```

Verify it worked:
```bash
ffmpeg -version
```

### 4. Install all Python dependencies

```bash
pip install -r requirements.txt
```

This installs torch, whisper, transformers, streamlit, and everything else
at tested, compatible versions. Takes 3-5 minutes (large downloads).

### 5. Download NLTK data (needed for text chunking)

```bash
python -c "import nltk; nltk.download('punkt')"
```

### 6. (Optional) Add your trained models

See `models/README.md` for instructions. If you skip this step, the
pipeline automatically uses public pretrained models instead.

---

## Running the Project

### Option A — Streamlit Web Interface (recommended)

```bash
streamlit run ui/streamlit_app.py
```

This opens a browser tab at `http://localhost:8501` where you can:
1. Upload an audio file
2. Click "Process Audio"
3. See the transcript, simplified text, and summary
4. Download results as `.txt` or `.json`

### Option B — Command Line

```bash
python run_pipeline.py data/input/my_audio.wav
```

With a custom output path:
```bash
python run_pipeline.py data/input/my_audio.wav --output my_results.json
```

### Option C — Test Individual Agents

Each agent can be tested on its own, without running the full pipeline:

```bash
# Test Agent 1 (place a test_audio.wav in data/input/ first)
python agents/transcription_agent.py

# Test Agent 2
python agents/simplification_agent.py

# Test Agent 3
python agents/summarization_agent.py

# Test the full orchestrator
python core/orchestrator.py
```

---

## How It Works

```
                    ┌─────────────────────┐
   Audio file  ───▶ │ TranscriptionAgent  │  Whisper: audio → text
                    └──────────┬──────────┘
                               │ transcript
                    ┌──────────┴──────────┐
                    ▼                     ▼
         ┌────────────────────┐ ┌──────────────────────┐
         │ SimplificationAgent│ │ SummarizationAgent   │
         │  plain language    │ │  short key points    │
         └──────────┬─────────┘ └───────────┬──────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    Displayed in Streamlit UI
                    Saved to data/output/*.json
```

The **Orchestrator** (`core/orchestrator.py`) loads all 3 agents once and
calls them in sequence. It does not contain any AI logic itself — its only
job is coordination.

---

## Configuration

All settings live in `config.py`. Key ones:

| Setting | File | Default | Description |
|---|---|---|---|
| `model_size` | Transcription | `"base"` | Whisper size: tiny/base/small/medium/large |
| `use_pretrained` | All 3 agents | varies | `True` = public model, `False` = your trained model |
| `chunk_size` | Simplification | `80` | Words per chunk sent to the model |
| `chunk_size` | Summarization | `500` | Words per chunk (larger, since summarization handles more context) |
| `max_upload_size_mb` | Streamlit | `200` | Max audio file size in the UI |

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'config'"**
→ Run all commands from the project root folder (`voxa_team/`), not from inside `agents/` or `core/`.

**"RuntimeError: No ffmpeg found"**
→ Install ffmpeg using step 3 above — it's a system program, not a pip package.

**Streamlit page is blank / stuck loading**
→ First run loads all 3 models (~1-2 minutes). Check your terminal for progress logs.

**"CUDA out of memory"**
→ You're running on GPU with a model too large. Either use `model_size="tiny"` in `config.py`, or install the CPU-only `torch` (already the default in `requirements.txt`).

**Custom model not loading**
→ Check `models/README.md`. Make sure `use_pretrained: False` is set in `config.py` and your model folder has all required files (`config.json`, `model.safetensors`, tokenizer files).

---

**Program:** Digital Egypt Pioneers Initiative (DEPI) — Generative AI Track
