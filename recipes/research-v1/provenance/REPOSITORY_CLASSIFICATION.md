# Repository classification at the 318bc90 integration base

This is a scoped implementation record, not a new experiment or source-license audit.
The [migration map](../../../legacy/MIGRATIONS.csv) contains per-file byte identities
and reasons. The [integration map](INTEGRATION.json) binds recovered source/configs.

| Group | Classification and dependency decision |
|---|---|
| `inference_native/` | Fixed release source closure; all bytes retained. Its `src` is independent of shared training source. |
| `tokenizer/releases/tokenizer_v1/` | Current bytes; checked manifest and seven-special-token contract; no duplicate alias. |
| `src/`, shared `pretrain/`, `sft/`, `dpo/`, `distill/`, tokenizer preparation/training, `scripts/` | Retained reusable implementations. Consumers include training imports, file/data utilities and active tests. Generic trainers are not substituted for recovered run-specific stages. Teacher-data utilities are not run by model-free validation. MoE remains implementation-only. |
| `recipes/research-v1/sources/` | Actual recovered stage roots, separately bound; same filenames do not imply equivalent implementations. P3/private runtime dependencies and sanitized-contract gaps are documented. |
| Six old mixture configs | Earlier SFT mixtures, including 137M-era settings, general/code budgets and local MBPP/code-fix paths. Frozen P2 uses reviewed 12k/500 rows, not these YAMLs. Archived without modifying recorded settings. |
| 291 tracked outputs + 378 samples | Named earlier pretrain_140m/SFT/distillation runs. Pretrain configs specify 12 layers/width768/12 heads and matching samples directories. Trainer and plot interfaces accept caller-selected output paths; they do not read the checked-in historical records as release inputs. |
| 33 additional evaluation snapshots | Old checkpoint/benchmark or shard identities and schemas, distinct from research-v1 protocol. Nine earlier roundA moves remain untouched. |
| Eight diagnostic/evaluator scripts | Two fixed old bracket-role/12-layer MHA tools and six earlier benchmark versions. Evaluators launch historical sampling CLIs and produce caller-selected nine-row results; old output configs record v5. No active release entry point consumes them; replay requires original source/sampling closure. |
| Four old tokenizer/metadata files | Four registered specials, pre-release versions established by artifact structure and Git history; root metadata not falsely assigned to one tokenizer. |
| Three GRPO modules | Moved to `experiments/grpo/`, with source bootstrap/imports, CLI examples, CI and active tests updated. Implementation-only; no measured research-v1 claim. |

720 historical files moved to `legacy/`, three implementation files to `experiments/`.
No source, report, score or output was deleted. Historical bytes stayed intact;
GRPO's path/import adaptations are deliberate exceptions to byte-identical moves.
No tests were archived or disabled. Literal searches supplemented source/CLI/config
and test review; they were not used as sole proof of absence of consumers.

Historical sources may refer to their original output/data layouts. No maintained
release command imports legacy modules or depends on archived artifacts. The shared
F builder retains its old frozen-contract path resolution helper; that helper does
not make the original F contract available, and is not a portable release recipe.
The recovered F source and precise missing binding requirements are separately indexed.
