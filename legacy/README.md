# Historical result snapshots

This directory groups nine already-public, passive SFT roundA evaluation JSON files under one historical namespace. See the [repository guide](../docs/REPOSITORY_GUIDE.md) for the current release and other directory roles.

## Scope and provenance

The files in [evaluation-results/](evaluation-results/) each identify an `outputs/sft_roundA.../latest.pt` checkpoint, `pretrain/bench_v1.jsonl`, and nine scored result rows. These are early experiment records, not the current release's public benchmark table. Their bytes, scores, generated text, and embedded paths are unchanged.

[MIGRATIONS.csv](MIGRATIONS.csv) maps every original repository-root-relative path to its new path, with the original Git blob identity, SHA256, reason, and known-consumer scope. Embedded checkpoint and benchmark paths still refer to the original repository/run context; do not resolve them relative to this archive directory or rewrite them to imply that missing weights are included.

At the migration baseline, `ece1ab7333710e148725e8c219105e1281b94c5a`, a bounded search of tracked Python, shell, configuration, JSON, Markdown, and other text found no literal old-path, basename, or stem references to these nine files. The evaluator family writes a caller-selected result path; the plotting helper reads explicit training-metrics inputs. Frozen documentation only describes the `eval/` directory generally, which remains present. No test, executable, or frozen document was changed to accommodate the move.

This is a local dependency review, not proof about external notebooks, private runs, or all dynamically constructed paths. Use the migration map if an external reference needs the new location. The snapshots do not identify an exact evaluator source revision; schema similarity alone does not establish one. Relocation does not make an experiment newly executed or fully reproducible.

Other configs, evaluation outputs, and samples remain in their original locations because their use or path relationships were not sufficiently resolved for relocation in this pass.
