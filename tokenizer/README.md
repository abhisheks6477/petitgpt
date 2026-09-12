# Tokenizer

The current public tokenizer is **[releases/tokenizer_v1/](releases/tokenizer_v1/)**.
Use `tokenizer/releases/tokenizer_v1/tokenizer.json` explicitly. There is no
unversioned alias or fallback. Its original bytes and [SHA256SUMS](releases/tokenizer_v1/SHA256SUMS)
are preserved; do not retrain, resave, or overwrite this release.

```text
tokenizer.json SHA256
 d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce
```

It is a 32,000-token byte-level BPE with no normalizer or postprocessor and
`add_prefix_space=false`. IDs are `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`,
`<|system|>=4`, `<|user|>=5`, `<|assistant|>=6`. Pretraining packing inserts
BOS/EOS explicitly; chat formatting masks roles and prompts and supervises
assistant content plus EOS. Literal control-token text must not inject role IDs.

From any working directory, set `REPO` to your checkout:

```bash
REPO=/path/to/petitgpt
(cd "$REPO/tokenizer/releases/tokenizer_v1" && sha256sum -c SHA256SUMS)
python "$REPO/recipes/research-v1/validate.py" \
  --synthetic-jsonl "$REPO/recipes/research-v1/examples/messages.synthetic.jsonl"
```

The first check uses system SHA256 tools; the second requires Python and
`tokenizers` for the optional message encoding. Neither constructs a model.
For corpus selection, training provenance, packing and the recorded tokenizer
launch, start at [Training and reproducibility](../TRAINING_AND_REPRODUCIBILITY.md).
[Preparation](data_preparation/) and [training/validation tools](tokenizer_training/)
remain available; their presence does not mean all frozen historical inputs ship here.

The unused four-special-token predecessors were removed from HEAD. Their exact
pre-removal commit/paths are described in [the history note](../legacy/tokenizers/README.md).
They are incompatible with the current chat contract and are not alternatives for research-v1.
