# Exporting checkpoints

Install the fork's CPU dependencies and the exporter dependencies in a Python environment:

```sh
uv pip install 'transformers==5.24.0' tiktoken blobfile safetensors
python -m hf_export.convert CHECKPOINT_DIR TOKENIZER_DIR OUTPUT_DIR [--step STEP]
```

The default step is the latest checkpoint in the directory. Load the resulting folder with `AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, trust_remote_code=True)` and `AutoTokenizer.from_pretrained(OUTPUT_DIR)`. Checkpoint and tokenizer pickle files must come from a trusted source. The exporter checks that their vocabulary sizes match and that every model weight loads. Complete the generated model card with the actual training run and license before sharing.

The fast tokenizer recognizes literal special-token strings as special tokens. Nanochat's ordinary text encoder instead treats those strings as ordinary text; compare special tokens using `encode_special`, and ordinary text that does not contain literal special tokens using `encode`.

After the operator approves a repository name and a private repository exists, upload with `python -m hf_export.upload OUTPUT_DIR OWNER/REPO`. The uploader refuses public repositories and does not create a repository.
