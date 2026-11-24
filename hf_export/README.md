# Exporting riemann-nano checkpoints

Install the fork's CPU dependencies and the exporter dependencies in a Python environment:

```sh
uv pip install 'transformers==5.24.0' tiktoken blobfile safetensors
python -m hf_export.convert CHECKPOINT_DIR TOKENIZER_DIR OUTPUT_DIR [--step STEP]
```

The default step is the latest checkpoint in the directory. Load the resulting folder with `AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, trust_remote_code=True)` and `AutoTokenizer.from_pretrained(OUTPUT_DIR)`. Checkpoint and tokenizer pickle files must come from a trusted source. The exporter checks that their vocabulary sizes match and that every model weight loads. Complete the generated model card with the actual training run and license before sharing.

The fast tokenizer recognizes literal special-token strings as special tokens. Nanochat's ordinary text encoder instead treats those strings as ordinary text; compare special tokens using `encode_special`, and ordinary text that does not contain literal special tokens using `encode`.

The approved smoke-test repository is `lewtun/riemann-nano-smoke`. When a real checkpoint is available, create the repository as private and immediately add it to the private `lewtun/riemann-nano-6b059a7cb0adb6e1ef7acba2` collection with a one-line note. Then upload with `python -m hf_export.upload OUTPUT_DIR lewtun/riemann-nano-smoke`. The uploader refuses public repositories and does not create a repository. The final model is `lewtun/riemann-nano`; other sweep exports belong in `lewtun/riemann-nano-sweep`, one subfolder per run using `--path-in-repo RUN_NAME`.
