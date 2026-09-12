# Research-v1 reader recipes

Start with [the practical manual](../../TRAINING_AND_REPRODUCIBILITY.md).

- [reader.py](reader.py): new-run pretrain A/B, P2, P3, fixed alpha075 blend, native export and local likelihood evaluation.
- [reader_prepare.py](reader_prepare.py): local message annotation, packed text format, P2/P3 deterministic plans; no acquisition.
- [PREPARED_INPUTS.md](PREPARED_INPUTS.md): exact supplied-input schemas and preparation coverage.
- [POSTTRAINING.md](POSTTRAINING.md): advanced historical adapters and immutable original identities.
- [sources/](sources/): recovered stage-specific source closures used by these adapters, separate from the shared trainer.
- [provenance/READER_V3.json](provenance/READER_V3.json): current reader source identities and explicit adaptations.
- [validate.py](validate.py): original snapshot/tokenizer checks with optional synthetic messages.

Use `--policy new-run` for compatible user data/checkpoints, or `--policy historical`
for the unchanged P2/P3/interpolation/export replay contracts. New-run validation
checks declared identities, schema and order; actual state checks happen when execution
loads weights. Small synthetic tests are not full-model execution evidence.
