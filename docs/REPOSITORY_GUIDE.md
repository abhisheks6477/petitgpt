# Repository guide

Use this map to distinguish the released inference implementation, shared research tooling, and historical records. The [project homepage](../README.md) gives the release overview; the versioned documents below define its reported scope.

## Start with the released model

| Question | Reading entry point |
|---|---|
| What was released, and what did the experiments establish? | [Technical report](petitgpt-v1/TECHNICAL_REPORT.md) |
| How is the native bundle used? | [Run guide](petitgpt-v1/RUN_GUIDE.md) |
| What are the model's intended use and limitations? | [GitHub model-card document](petitgpt-v1/MODEL_CARD.md) |
| How were the released weights trained and evaluated? | [Training and reproducibility](../TRAINING_AND_REPRODUCIBILITY.md) |
| Which tokenizer should I use? | [Canonical tokenizer](../tokenizer/README.md) |
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
| [pretrain/](../pretrain/) | Reusable data selection, shard/reference-validation contracts and training. Exact release source lives in the recipe roots. Start with [train_pretrain.py](../pretrain/train_pretrain.py) and [run_plan_contract.py](../pretrain/run_plan_contract.py); input construction lives in [build_pretrain_shards.py](../pretrain/build_pretrain_shards.py). |
| [sft/](../sft/) | SFT data preparation and shared supervised training. Start with [train_sft.py](../sft/train_sft.py); [prepare_sft_mix_split_local.py](../sft/prepare_sft_mix_split_local.py) accepts an explicit mixture config. |
| [distill/](../distill/) | Retained distillation/KD research tooling, including teacher-data preparation and verification. [train_distill.py](../distill/train_distill.py) delegates text-response training to the SFT engine; it does not by itself reproduce the separate soft-logit lab. Some data-preparation scripts call external teachers. |
| [dpo/](../dpo/) | Retained preference-data preparation and DPO research implementation; see [dpo.py](../dpo/dpo.py). |
| [experiments/](../experiments/README.md) | Measured research index; [GRPO](../experiments/grpo/README.md) is implementation-only with active tests, not an established research-v1 run. |
| [recipes/research-v1/](../recipes/research-v1/README.md) | Reader new-run training/evaluation entry point, prepared input schemas and separate historical adapters over recovered source roots. |
| [configs/research-v1/](../configs/research-v1/README.md) | Effective launch/parameter/input evidence; projected JSON is not an executable frozen contract. Old mixture YAMLs moved to legacy. |
| [scripts/](../scripts/) | Supporting utilities. [plot_metrics.py](../scripts/plot_metrics.py) reads explicit run directories or metrics files; it is not the evaluator for every saved result JSON. |
| [tests/](../tests/) | Repository contract and unit tests. Read [pytest.ini](../pytest.ini), the [shared fixtures](../tests/conftest.py), and the [CI workflow](../.github/workflows/ci.yml) before choosing checks; CPU-only does not mean model-free. |
| [docs/](./) | This navigation guide and the preserved [versioned release documents](petitgpt-v1/). The [historical README](petitgpt-v1/HISTORICAL_README.md) preserves an earlier narrative, not the current release specification. |

DPO, KD/distillation, and GRPO code is retained research tooling. Its presence does not imply inclusion in alpha075 or completion of every possible method. Use the technical report and experiment lineage to identify what was actually measured and which updates belong to the released weights.

## Read historical artifacts in context

[legacy/](../legacy/README.md) retains aggregate experiment records and historical
source that still helps interpretation. Obsolete per-step samples, output snapshots
and unused predecessor tokenizers were removed from HEAD; the short history note
points to their pre-removal commit. The immutable migration CSV describes its older
snapshot, not the current runtime. Published scientific evidence remains unchanged.

DPO/KD shared code remains in functional directories. GRPO moved to experiments
with its imports, direct-script bootstrap, CI and active tests updated. No test was
archived or disabled. `src/model_moe.py` remains a tested implementation with no
released-weight or completed research-v1 experiment claim.

The full [training/evaluation guide](../TRAINING_AND_REPRODUCIBILITY.md) separates
source recovery, syntax/config checks, data-path checks and model-work validation.
Prepared caller data can run through the new-run interfaces without private approvals.
Original data/approvals still limit exact historical replay; synthetic validation does
not establish full-model runtime or the published numerical results.

## A reading route for project discussion

1. Start with the technical report and model card to separate release scope, measured strengths, and limitations.
2. Trace the lineage in the versioned tables before discussing later SFT, DPO, or distillation experiments.
3. Read shared model/chat contracts, then the relevant preparation and training entry points. Keep the released inference implementation separate.
4. Use historical snapshots only with their recorded checkpoint, benchmark, and scoring context. Distinguish an implementation, a recorded experiment, and a released-weight contribution.
