# Research-v1 recipes

Begin with [Training and reproducibility](../../TRAINING_AND_REPRODUCIBILITY.md).
It covers the exact release lineage, source identities, inputs, commands, effective
settings, checkpoint selection, evaluation and remaining reproducibility gaps.

- [validate.py](validate.py): model-free source/tokenizer/config/CLI-flag validation;
  optional [synthetic messages](examples/messages.synthetic.jsonl), never fake scores.
- [p2.py](p2.py): new public P2 path adapter with `--validate-only` and explicit future
  `--execute`; fixed original input hashes and unchanged recovered trainer semantics.
- [POSTTRAINING.md](POSTTRAINING.md): callable P3, fixed interpolation and native
  export interfaces, schemas, commands, validation evidence and replay limits.
- [sources/](sources/): recovered, independently bound source roots. Some scripts
  execute model work or I/O at import; static validation is the safe common entry point.
- [provenance/](provenance/): handoff, source map, integration comparison, adaptations
  and original legal notices. The handoff manifest describes input archive members;
  use INTEGRATION.json for their current repository locations.
- [configs/research-v1/](../../configs/research-v1/README.md): projected historical
  evidence, not a new generic training config loader.

P2/P3/fixed-interpolation/native-export adapters are implemented; model execution
has not been validated here. See POSTTRAINING.md for stage-specific checks and
private-input limits. F/I/M/A/B and public benchmark runtime portability remain
separate gaps.
No public source snapshot is a claim of full reproduction. The original handoff's
missing-public-baseline caveat is resolved by INTEGRATION.json's 318bc90 comparison.
