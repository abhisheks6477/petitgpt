# Repository guide

Use this map to distinguish the released inference implementation, shared research tooling, and historical records. The [project homepage](../README.md) gives the release overview; the versioned documents below define its reported scope.

## Start with the released model

| Question | Reading entry point |
|---|---|
| What was released, and what did the experiments establish? | [Technical report](petitgpt-v1/TECHNICAL_REPORT.md) |
| How is the native bundle used? | [Run guide](petitgpt-v1/RUN_GUIDE.md) |
| What are the model's intended use and limitations? | [GitHub model-card document](petitgpt-v1/MODEL_CARD.md) |
| Where are the recorded comparisons and experiment lineage? | [Versioned tables](petitgpt-v1/tables/) |
| Where is the released implementation? | [Native inference](../inference_native/), beginning with [inference.py](../inference_native/inference.py) |

`inference_native/` contains a fixed released implementation and its own source closure. Its local `src/` is intentional: do not substitute the repository's shared `src/` for it. Follow the run guide for the complete bundle, requirements, and revision pins; this checkout is not a substitute for that bundle.

The Hugging Face model card is the root `README.md` in the separate `yqi0/petitgpt` model repository. The GitHub model-card document linked above is not automatically synchronized with it. This repository organization change does not synchronize or modify Hugging Face.

## Find the implementation by role

These are reading entry points, not an instruction to run every script or a claim that every historical run can be reproduced from public main.

| Directory | Role and useful starting point |
|---|---|
| [src/](../src/) | Shared model, chat/token contracts, optimization, tracking, and training utilities. Start with [model.py](../src/model.py) and [chat_template.py](../src/chat_template.py). Separate from the fixed inference source closure. |
| [tokenizer/](../tokenizer/) | Tokenizer corpus preparation, training, validation, and versioned release files. See [tokenizer_training/](../tokenizer/tokenizer_training/) and [releases/tokenizer_v1/](../tokenizer/releases/tokenizer_v1/). |
| [pretrain/](../pretrain/) | Data selection, shard/reference-validation contracts, training, and earlier diagnostics. Start with [train_pretrain.py](../pretrain/train_pretrain.py) and [run_plan_contract.py](../pretrain/run_plan_contract.py); input construction lives in [build_pretrain_shards.py](../pretrain/build_pretrain_shards.py). |
| [sft/](../sft/) | SFT data preparation and shared supervised training. Start with [train_sft.py](../sft/train_sft.py); [prepare_sft_mix_split_local.py](../sft/prepare_sft_mix_split_local.py) accepts an explicit mixture config. |
| [distill/](../distill/) | Retained distillation/KD research tooling, including teacher-data preparation and verification. [train_distill.py](../distill/train_distill.py) delegates text-response training to the SFT engine; it does not by itself reproduce the separate soft-logit lab. Some data-preparation scripts call external teachers. |
| [dpo/](../dpo/) | Retained preference-data preparation and DPO research implementation; see [dpo.py](../dpo/dpo.py). |
| [grpo/](../grpo/) | Retained GRPO research implementation, prompt preparation, and reward functions; see [grpo.py](../grpo/grpo.py) and [rewards.py](../grpo/rewards.py). |
| [configs/](../configs/) | Versioned SFT mixture specifications. Match each config to its data-preparation tool and historical run before use; the largest version number is not the release recipe. |
| [scripts/](../scripts/) | Supporting utilities. [plot_metrics.py](../scripts/plot_metrics.py) reads explicit run directories or metrics files; it is not the evaluator for every saved result JSON. |
| [tests/](../tests/) | Repository contract and unit tests. Read [pytest.ini](../pytest.ini), the [shared fixtures](../tests/conftest.py), and the [CI workflow](../.github/workflows/ci.yml) before choosing checks; CPU-only does not mean model-free. |
| [docs/](./) | This navigation guide and the unchanged [versioned release documents](petitgpt-v1/). The [historical README](petitgpt-v1/HISTORICAL_README.md) preserves an earlier narrative, not the current release specification. |

DPO, KD/distillation, and GRPO code is retained research tooling. Its presence does not imply inclusion in alpha075 or completion of every possible method. Use the technical report and experiment lineage to identify what was actually measured and which updates belong to the released weights.

## Read historical artifacts in context

| Location | What remains and how to interpret it |
|---|---|
| [eval/](../eval/) | Earlier benchmark and sanity-check snapshots that remain at their original paths. They are not interchangeable with the versioned public result tables. |
| [legacy/evaluation-results/](../legacy/evaluation-results/) | Nine early SFT roundA result snapshots, each recording an old checkpoint and nine benchmark rows. The [legacy index](../legacy/README.md) and [migration map](../legacy/MIGRATIONS.csv) preserve original paths and identities. |
| [outputs/](../outputs/) | Tracked historical run configs and evaluation/log records coexist with a path ignored for new local outputs. Embedded checkpoint paths do not mean those weights are distributed here. |
| [samples/](../samples/) | Tracked historical generated-text samples. Training configs and sampling code use run/step-based paths, so these remain in place with their context. |

The mixture configs, remaining evaluation files, output records, and samples have not been declared unused. Moving the nine roundA JSON files preserves their bytes, scores, embedded checkpoint/benchmark paths, and recorded results; it neither reruns those experiments nor establishes the exact evaluator revision or full reproducibility. No executable modules, training configs, fixtures, or frozen release files were relocated.

Private `runs/` records are not distributed with GitHub. Complete training data, optimizer checkpoints, and some run-specific implementations and evaluation evidence also remain outside this public tree. A versioned report can describe evidence that is not itself shipped here. Public main and a private training workspace are different scopes; this checkout is not a full project backup.

## A reading route for project discussion

1. Start with the technical report and model card to separate release scope, measured strengths, and limitations.
2. Trace the lineage in the versioned tables before discussing later SFT, DPO, or distillation experiments.
3. Read shared model/chat contracts, then the relevant preparation and training entry points. Keep the released inference implementation separate.
4. Use historical snapshots only with their recorded checkpoint, benchmark, and scoring context. Distinguish an implementation, a recorded experiment, and a released-weight contribution.
