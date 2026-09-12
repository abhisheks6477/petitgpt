# Historical implementations and artifacts

Superseded public artifacts live under this single root. Start with the
[current repository guide](../docs/REPOSITORY_GUIDE.md) and
[release recipe](../TRAINING_AND_REPRODUCIBILITY.md).

| Location | Evidence and interpretation |
|---|---|
| [tokenizers/](tokenizers/README.md) | Two old four-special-token BPE versions and their former root metadata. |
| [configs/](configs/) | v1–v6 SFT mixtures, including the old 137M token-budget specification and local code mixtures; none is the frozen P2 12,000/500-row selection. |
| [outputs/](outputs/) | Earlier `pretrain_140m_*` 12-layer/768-width runs and named SFT/distillation run records. Configs, metrics and evaluation records remain byte-identical. |
| [samples/](samples/) | Step-based text outputs paired with those historical pretraining run names. These are existing public evidence, not newly published data. |
| [evaluation-results/](evaluation-results/) | Earlier nine-row generated-answer benchmark and shard sanity snapshots, including the nine previously moved roundA files. Not the research-v1 likelihood protocol. |
| [tools/](tools/) | Fixed old bracket-role/MHA diagnostics and the superseded `eval_bench` v1–v5 family. Some execute model work at import; they are historical source, not supported CLIs. |

[MIGRATIONS.csv](MIGRATIONS.csv) records each original/new path, original Git blob,
SHA256 and classification evidence. The new migration baseline is
`318bc90a2b3afa4ef39c9f2c1a3d3da458271de3`; the first nine rows retain their earlier
baseline. GRPO's moves to `experiments/` are recorded in the same map.

Classification used the handoff lineage, run configs/checkpoint names, tokenizer
structure and Git history, evaluator subprocess behavior, caller-selected output
paths, imports, tests and CI. Generic training/sample/plot utilities write or read
caller-selected paths; they do not require the checked-in old output files to stay
in their original directories. The migration is not proof about external notebooks.

Embedded historical paths, scores and text were deliberately preserved. To replay
an old snapshot, use its original commit and original missing inputs; simply running
a relocated script against today's shared model is not an exact replay. The old
benchmark family refers to historical sampling interfaces. No compatibility shims
or archived test exclusions were introduced. Active GRPO tests were updated and
remain in the test suite. The frozen versioned reports and source/legal notices
remain at their established locations.
