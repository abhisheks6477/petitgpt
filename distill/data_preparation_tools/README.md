# Alternative teacher-data preparation

These retained research tools use prompt rows with `id`, `task_type`, `prompt`
and optional `meta`. Check each script's input/output handling before connecting
it to the [top-level canonical-bank tools](../README.md); similarly named stages
do not guarantee matching schemas.

| Task | Tools |
|---|---|
| Generate local prompts with explicit output paths | [make_teacher_general_prompts_v2.py](make_teacher_general_prompts_v2.py), [make_teacher_code_prompts_v2.py](make_teacher_code_prompts_v2.py). Code prompts carry entrypoints and tests in metadata. |
| Expand prompts through a teacher | [bootstrap_general_prompts_with_teacher.py](bootstrap_general_prompts_with_teacher.py), [expand_code_prompts_with_teacher.py](expand_code_prompts_with_teacher.py). |
| Generate candidate answers | [stage2_teacher_generate_open.py](stage2_teacher_generate_open.py) accepts an explicit compatible endpoint and preserves prompt metadata. [stage2_teacher_generate.py](stage2_teacher_generate.py) retains a different request/formatting path; the two are not identical replacements. |
| Verify code and build chat data | [verify_code_candidates_with_tests.py](verify_code_candidates_with_tests.py), [build_general_code_teacher_dataset.py](build_general_code_teacher_dataset.py). |
| Verify stage-2 answers and build a split | [stage2_verify_and_build.py](stage2_verify_and_build.py), including optional math answer keys. |
| Retain the later fixed stage-2 prompt recipes | [make_stage2_basic_math_prompts_v1_2.py](make_stage2_basic_math_prompts_v1_2.py), [make_stage2_general_code_prompts_v1_2.py](make_stage2_general_code_prompts_v1_2.py). These run at module scope and write to `dataset/stage2/prompts_v1_2` relative to the working directory; they have no `--help` CLI. |

Use [../normalize_existing_smol_source_v1.py](../normalize_existing_smol_source_v1.py)
for local Smol/SmolTalk normalization. The older same-named script here is
[archived](../../legacy/distill/README.md), along with the stage-2 v1/v1.1
generators and Qwen smoke scripts. The retained normalizer handles more schemas
and filters empty prompts, so its output is not byte-identical to the old version.

The parameterized scripts expose `--help`, for example:

```bash
python distill/data_preparation_tools/make_teacher_code_prompts_v2.py --help
python distill/data_preparation_tools/stage2_verify_and_build.py --help
```

Teacher clients require the `openai` package and a configured service. CLI checks
do not validate a remote teacher. The fixed v1.2 generators retain their original
mixtures and output paths; their presence is not a released-model recipe claim.
