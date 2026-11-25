# Exporting riemann-nano checkpoints

Install the development dependencies using Python 3.11 or newer. Conversion requires `transformers==5.24.0`, `blobfile`, and the fork's tokenizer dependencies. Checkpoint and tokenizer pickles must come from trusted sources.

```sh
python -m hf_export.convert CHECKPOINT_DIR TOKENIZER_DIR OUTPUT_DIR
```

The converter chooses the latest numbered checkpoint and verifies `tokenizer.pkl` against `tokenizer_sha256` in its metadata. For legacy runs without that field, establish the training tokenizer's SHA256 independently and pass it as `--tokenizer-sha256 HASH`; matching vocabulary size alone is insufficient. Conversion writes to a temporary directory first. `--overwrite` replaces the entire prior export directory, including its card and assets; it refuses non-export directories and any directory containing an input.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
```

The default tokenizer call prepends BOS; the distinct EOS token is used for generation stopping and left padding. Set `add_special_tokens=False` to match nanochat's ordinary text encoder. Literal special-token strings are recognized as special tokens by the fast tokenizer; nanochat's ordinary text encoder treats them as ordinary text.

To include the results section and referenced plot images in the final card, supply both options:

```sh
python -m hf_export.convert CHECKPOINT_DIR TOKENIZER_DIR OUTPUT_DIR \
  --results-markdown RESULTS_SECTION.md --assets-dir PLOTS_DIR
```

The results Markdown is appended to the Apache 2.0 model card. An adjacent `selected_model.json` must match the checkpoint's run name and final validation BPB. Referenced images must use relative links such as `assets/scaling.png`; the converter checks and copies those images into `OUTPUT_DIR/assets/`. Complete the card's training split and limitations before sharing.

Upload to an existing private model repository with `python -m hf_export.upload OUTPUT_DIR OWNER/REPO`. The released model card is hand-edited: the uploader preserves an existing destination `README.md`, including on `--overwrite`. Edit the card separately when needed. The uploader refuses public repositories and pickle files, checks required export files, and refuses to replace an existing export unless `--overwrite` is passed. For a collection of runs, use `--path-in-repo RUN_NAME`; the uploader places matching remote-code files at the repository root. Load a run with `AutoModelForCausalLM.from_pretrained(REPO, subfolder=RUN_NAME, trust_remote_code=True)` and likewise for `AutoTokenizer`. Existing root modeling files must match, so use a separate repository or branch for incompatible code.

For optional real-data tests, set `RH_TEST_TOKENIZER` to the training tokenizer directory and `RH_TEST_LEAN` to a Lean source directory containing `Main.lean` and `Statements.lean`, then run `python -m pytest tests/test_hf_export.py`.
