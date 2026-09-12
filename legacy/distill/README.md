# Archived distillation helpers

These seven scripts moved from `distill/` on 2026-09-12 after checking their source,
successor differences and repository references. Their file contents are unchanged.
No other code imports or launches these scripts in the reviewed tree. Current
research entry points are documented in [distill/README.md](../../distill/README.md).

Paths in the first column are relative to the former `distill/` directory; the
archive preserves that subdirectory layout. This is a later cleanup and does not
modify the older immutable [migration CSV](../MIGRATIONS.csv).

| Original relative path / archived file | Reason and retained alternative |
|---|---|
| [patch_general_utils_qwen_nonthinking_v1.py](patch_general_utils_qwen_nonthinking_v1.py) | One-time source patch. Its `disable_thinking`, request kwargs and response cleanup already exist in [general_utils.py](../../distill/general_utils.py). Its default target still points to `distill/general_utils.py`; this is historical source, not a setup step. |
| [data_preparation_tools/qwen35_gptq_smoke_test.py](data_preparation_tools/qwen35_gptq_smoke_test.py) | Standalone teacher-endpoint trial with a fixed example prompt, not a pytest test or pipeline dependency. |
| [data_preparation_tools/test_qwen35_hf_9b.py](data_preparation_tools/test_qwen35_hf_9b.py) | Standalone Qwen 9B/4-bit loading and generation trial, not a pytest test or PetitGPT runtime component. |
| [data_preparation_tools/make_stage2_prompts_v1.py](data_preparation_tools/make_stage2_prompts_v1.py) | Initial fixed general/code/math prompt lists. Later stage-2 recipes remain in [data_preparation_tools/](../../distill/data_preparation_tools/README.md); preserve this version for its distinct original examples. |
| [data_preparation_tools/make_stage2_basic_math_prompts_v1_1.py](data_preparation_tools/make_stage2_basic_math_prompts_v1_1.py) | Earlier fixed math mixture. The retained [v1.2](../../distill/data_preparation_tools/make_stage2_basic_math_prompts_v1_2.py) changes task proportions, answer formats and row budget; it is not an exact historical-data replacement. |
| [data_preparation_tools/make_stage2_general_code_prompts_v1_1.py](data_preparation_tools/make_stage2_general_code_prompts_v1_1.py) | Earlier fixed general/code mixture. The retained [v1.2](../../distill/data_preparation_tools/make_stage2_general_code_prompts_v1_2.py) revises prompts and sampling; archive the old recipe instead of conflating their outputs. |
| [data_preparation_tools/normalize_existing_smol_source_v1.py](data_preparation_tools/normalize_existing_smol_source_v1.py) | Earlier narrow-schema converter. Prefer the [top-level normalizer](../../distill/normalize_existing_smol_source_v1.py), which adds conversation aliases, instruction/input joining, source metadata, empty-prompt filtering and sampling. The two were not identical copies. |

Some archived scripts execute immediately when imported, including model loading,
service requests or writes to their original relative dataset paths. They remain
outside pytest's `tests/` collection scope. This archive preserves research context;
it is not part of the supported training or inference launch path.
