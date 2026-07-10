# models/ — Where Your Trained Models Go

This folder holds the trained models used by the 3 agents.

```
models/
├── whisper_finetuned/        ← Person 1's fine-tuned Whisper (optional)
├── simplification_model/     ← Person 2's trained FLAN-T5
└── summarization_model/      ← Person 3's trained DistilBART
```

## How to add your trained model

If you trained a Hugging Face model (FLAN-T5, DistilBART, etc.) using
`model.save_pretrained("some_folder")`, copy the entire folder here.

A correctly saved model folder looks like this:

```
simplification_model/
├── config.json
├── model.safetensors        (or pytorch_model.bin)
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── spiece.model              (only for T5-based tokenizers)
```

## Enabling your trained model

Open `config.py` in the project root and for the agent you're updating,
set `use_pretrained` to `False`:

```python
SIMPLIFICATION_CONFIG = {
    ...
    "use_pretrained": False,   # ← change this to False
    "model_path": SIMPLIFICATION_MODEL_PATH,  # already points here
    ...
}
```

If `use_pretrained` stays `True`, the agent loads the public pretrained
model from Hugging Face instead (good for testing before your model is ready).

If your model folder is missing or invalid, each agent automatically
falls back to the pretrained model and prints a warning — so the pipeline
never crashes because of a missing model folder.
