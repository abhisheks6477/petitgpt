# Response-distillation tools

This directory retains reusable research code for preparing teacher answers,
verifying them, and training on the accepted text. File age is not a maintenance
status: shared dependencies and useful standalone data tools remain here; retired
helpers are listed in the [archive](../legacy/distill/README.md).

For the released model's training and evaluation workflow, start with
[Training and reproducibility](../TRAINING_AND_REPRODUCIBILITY.md). These generic
tools do not reproduce the recorded RKD1/RKD2 runs or the separate soft-logit lab
by themselves. Later response-distillation updates are not in alpha075; see the
[technical report](../docs/petitgpt-v1/TECHNICAL_REPORT.md).

## Choose a tool

| Purpose | Files and status |
|---|---|
| Train on teacher-written responses | [train_distill.py](train_distill.py) delegates to the shared [SFT engine](../sft/train_sft.py), using the `distill` stage identity for checkpoints/resume. Covered by the post-training resume tests. |
| Shared verification and data helpers | [code_utils.py](code_utils.py) supplies code extraction, AST checks and test execution, including the [GRPO code reward](../experiments/grpo/rewards.py). [general_utils.py](general_utils.py) supplies general-answer checks and teacher requests. Keep these with their consumers. |
| Export or normalize open sources | `export_hf_code_sources_v1.py`, `export_hf_general_sources_v1.py`, `export_hf_smol_source_v1.py`, [normalize_existing_smol_source_v1.py](normalize_existing_smol_source_v1.py). Exporters download datasets; the normalizer accepts local JSONL. |
| Prepare code prompts | `code_extract_apps_v1.py`, `code_extract_mbpp_v1.py`, `code_gen_core_families_v1.py`, `gen_core_family_prompts_v2.py`, `build_code_canonical_prompts_v1.py`. These cover extraction, task-family generation and canonicalization; version numbers alone do not establish interchangeability. |
| Generate and verify code answers | `code_teacher_generate_v1.py`, `code_verify_v1.py`, `code_build_bank_v1.py`. Teacher generation supports raw answers and repair; bank building selects verified candidates. |
| Prepare general prompts | `general_extract_open_v1.py`, `general_classify_canonicalize_v1.py`, `general_gen_template_seeds_v1.py`, `general_paraphrase_templates_v1.py`. Paraphrasing uses a teacher endpoint. |
| Generate and verify general answers | `general_teacher_generate_v1.py`, `general_verify_v1.py`, `general_build_bank_v1.py`. Retained research utilities with task-specific format/content checks. |
| Assemble datasets | [build_targeted_distill_mix_v1.py](build_targeted_distill_mix_v1.py) combines code/general train and validation banks; [build_smoke_manifests_v1.py](build_smoke_manifests_v1.py) selects small prompt subsets. |
| Alternative teacher-data preparation | [data_preparation_tools/](data_preparation_tools/README.md) contains the task-type/metadata-based workflow and the remaining stage-2 prompt variants. Its schemas differ from the top-level canonical banks. |

## Running and validation scope

Run top-level CLIs from the repository root as `python distill/<script>.py ...`;
several import sibling helpers and are intended for direct-script invocation.
Inspect arguments before choosing inputs, for example:

```bash
python distill/train_distill.py --help
python distill/general_extract_open_v1.py --help
python distill/build_targeted_distill_mix_v1.py --help
```

Teacher generation needs an explicitly configured model/service. Dataset export
needs the `datasets` package; semantic selection may need embedding/TF-IDF
dependencies. The trainer consumes chat-message examples; intermediate prompt and
candidate banks require the corresponding conversion/verification steps first.

The repository tests exercise the training wrapper and GRPO's shared code verifier.
Retention here does not imply that every data source, teacher endpoint or full
historical pipeline is continuously tested. A passing `--help` check establishes
only CLI startup, not successful data generation or training.

## Archived helpers

Seven scripts moved to [legacy/distill/](../legacy/distill/README.md): the already
applied Qwen nonthinking patch, two model/service smoke scripts, three earlier
stage-2 prompt generators, and the older Smol normalizer. Their contents are
preserved, with original paths and reasons in the archive index. The remaining
35 Python files keep their existing paths.
