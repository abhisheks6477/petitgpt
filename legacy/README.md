# Earlier research and history

Current training/evaluation starts in the [reader manual](../TRAINING_AND_REPRODUCIBILITY.md).
This directory keeps experiment configs, aggregate metrics and result summaries,
old diagnostic source and the immutable [migration record](MIGRATIONS.csv).
The migration CSV describes the tree at commit `5044497`; it is not a current
runtime file manifest and does not assert that every migrated artifact remains at HEAD.

Reader closeout removed 665 obsolete files from HEAD: 378 per-step sample texts,
281 per-step output/sample/benchmark snapshots, four old tokenizer/config JSONs,
and two old shard-sanity dumps. No active code/test/report reference to their current
or migration-original paths was found. Aggregate metrics/configs, meaningful experiment
summaries, negative findings, published tables and notices remain. These removals
change browsing, not the scientific results or Git history.

For removed content, use the same path at pre-removal commit
`e76cc03ad1a9f863911612defd2e07c977aef144`, for example
`git show e76cc03:legacy/samples/<run>/<file>`. For pre-migration names, consult
MIGRATIONS.csv. Original embedded paths in historical scripts/records are evidence;
those scripts are not the current supported launchers. No model assets or private
backups were deleted.
