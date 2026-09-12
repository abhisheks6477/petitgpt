# Research-v1 recipes

Begin with [Training and reproducibility](../../TRAINING_AND_REPRODUCIBILITY.md).
It covers the exact release lineage, source identities, inputs, commands, effective
settings, checkpoint selection, evaluation and remaining reproducibility gaps.

- [validate.py](validate.py): model-free source/tokenizer/config/CLI-flag validation;
  optional [synthetic messages](examples/messages.synthetic.jsonl), never fake scores.
- [p2.py](p2.py): new public P2 path adapter with `--validate-only` and explicit future
  `--execute`; fixed original input hashes and unchanged recovered trainer semantics.
- [sources/](sources/): recovered, independently bound source roots. Some scripts
  execute model work or I/O at import; static validation is the safe common entry point.
- [provenance/](provenance/): handoff, source map, integration comparison, adaptations
  and original legal notices. The handoff manifest describes input archive members;
  use INTEGRATION.json for their current repository locations.
- [configs/research-v1/](../../configs/research-v1/README.md): projected historical
  evidence, not a new generic training config loader.

P2 source and path adaptation are present. F/I/M/A/B/P3/interpolation/export and
published evaluation source/evidence are recovered, but full portable launches
remain blocked where original frozen inputs or binding adapters are missing.
No public source snapshot is a claim of full reproduction. The original handoff's
missing-public-baseline caveat is resolved by INTEGRATION.json's 318bc90 comparison.
