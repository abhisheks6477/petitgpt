# Training and reproducibility: research-v1

The released **alpha075** lineage is **Base step_049590 → P2 step750 → P3 step320**,
then `P2 + 0.75 * (P3_step320 - P2)`, followed by native FP32 export. The model has
124,635,456 unique parameters: 30 layers, width 576, FFN 1536, 9 Q / 3 KV heads,
32,000 vocabulary, context 2,048, tied embeddings, RoPE/RMSNorm/SwiGLU/GQA.
Later DPO, response KD, LoRA and unified Base-SFT are experiments, not ancestors.
The soft-logit KD lab uses separate external models.

This entry point integrates recovered run-specific source and distinguishes three
things: **recorded historical launches**, **a public P2 path adapter**, and **new
synthetic usage checks**. The full historical recipe is not currently runnable from
public inputs alone. Missing frozen contracts and data are listed below, not replaced
with a generic SFT invocation. Existing [results](docs/petitgpt-v1/tables/) and
[methodology](docs/petitgpt-v1/TECHNICAL_REPORT.md) remain unchanged.

## Start here

| Need | Entry point |
|---|---|
| Exact current tokenizer | [tokenizer/README.md](tokenizer/README.md) |
| Inspect release source without executing model work | [recipes/research-v1/](recipes/research-v1/README.md) |
| Effective settings, argv, input identities, recorded revisions | [configs/research-v1/](configs/research-v1/README.md) |
| Original/candidate file hashes and resolved public comparison | [INTEGRATION.json](recipes/research-v1/provenance/INTEGRATION.json), [SOURCE_MAP.json](recipes/research-v1/provenance/SOURCE_MAP.json) |
| Recovered stage descriptions and execution caveats | [handoff](recipes/research-v1/provenance/HANDOFF.md) |
| Released inference | [run guide](docs/petitgpt-v1/RUN_GUIDE.md), separate [native source closure](inference_native/) |
| Non-release measured and implementation-only tracks | [experiments/](experiments/README.md) |
| Earlier superseded versions | [legacy/](legacy/README.md) |

The integration base is `318bc90a2b3afa4ef39c9f2c1a3d3da458271de3`. The handoff had
compared against `157ef969de38167dbf0dce04e9095344a4a4b177` because it lacked 318bc90.
That gap is now resolved: 81 mapped public-source comparisons include 37 matching
originals, 10 differing originals, and 34 paths absent at the base. In particular,
the recovered P2 trainer, chat encoding, canonical loss and schedule must not be
replaced by similarly named current modules. Some candidate sources are already
path-sanitized and differ from originals; both hashes remain recorded.

Every `recipes/research-v1/sources/<stage>/` is a separate source root. Empty package
initializers prevent Python from accidentally selecting the checkout's shared `src`.
Recovered implementation bytes are unchanged from the handoff. These are source
closures, not importable stages in one combined package. `src/` and
`inference_native/src/` remain distinct. No frozen inference bytes changed.

## Validation and environments

Use absolute roots to avoid depending on the current directory:

```bash
REPO=/path/to/petitgpt
DATA_ROOT=/path/to/approved-local-inputs
OUTPUT_ROOT=/path/to/new-local-outputs
python "$REPO/recipes/research-v1/validate.py"
python "$REPO/recipes/research-v1/validate.py" \
  --synthetic-jsonl "$REPO/recipes/research-v1/examples/messages.synthetic.jsonl"
python "$REPO/recipes/research-v1/p2.py" --help
```

`validate.py` checks source/config hashes, tokenizer manifest/contract, Python syntax
and recorded flag correspondence without importing trainers. Optional synthetic
encoding needs `tokenizers`; it does not import torch. P2 `--help` is also free of
trainer imports. P2 `--validate-only` additionally requires NumPy, CPU-capable PyTorch
and tokenizers for recovered parser/data/plan utilities; it does not construct a
model, deserialize a checkpoint, perform a forward pass or inspect CUDA. It hashes
provided input files, including the Base checkpoint, as opaque bytes.

Historical observations are compatibility evidence, not dependencies installed by
these commands:

| Stage | Recorded environment / requirements |
|---|---|
| F/G tokenizer | Python 3.10.12, tokenizers 0.22.2; exact trainer/corpus manifests and byte ordering. |
| Pretraining A/B | Python 3.10.12, torch 2.11.0+cu126, NumPy 2.2.6, CUDA 12.6, RTX 4090, driver 580.126.20; BF16 and compile. |
| P2/P3 | Recovered Torch/NumPy/tokenizers training closure; single BF16-capable CUDA GPU. P2 explicitly requires one GPU/process. Exact complete P2 launch environment is not independently pinned by the projected argv alone. |
| Native export/inference | Python 3.10.12, torch 2.11.0+cu126, NumPy 2.2.6, tokenizers 0.22.2, safetensors 0.8.0; [tested requirements](inference_native/requirements-inference-tested.txt). |
| Published benchmark | Above plus transformers 5.15.0, datasets 5.0.1, huggingface_hub 1.27.0, pyarrow 25.0.1; exact guards in recovered `common.py`. Its sanitized executable/prefix guards require a reviewed port, not suppression. |

Validation levels are separate: **source-recovered**, **syntax/config checked**,
**data-path checked**, **training-smoke run**, **full historical reproduction**.
The first two are established for the supplied source/configs. Synthetic schema
checks are established separately. Exact historical data-path validation, any
training smoke, export/score execution and full historical reproduction are **not
established** by this repository maintenance change.

## Tokenizer corpus, tokenizer training, selection and packing

These stages contribute to the released weights through tokenizer/data identities.
The canonical tokenizer is
`tokenizer/releases/tokenizer_v1/tokenizer.json`, SHA256
`d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce`.
Never resave it to make a new path or overwrite it with a synthetic training result.

| Stage | Actual implementation under `recipes/research-v1/sources/` | Inputs and outputs |
|---|---|---|
| F corpus | `tokenizer/tokenizer/data_preparation/build_tokenizer_training_corpus.py` | D2/D3 eligible occurrences, frozen E allocation and L1 reserve exclusions → five canonical JSONL buckets; 2,511,569 occurrences, 10,000,043,658 UTF-8 text bytes. Exact original builder hash recovered (`81fc2a63…`); file membership at recorded 90e14a34 is not established. |
| G tokenizer | `tokenizer/tokenizer/tokenizer_training/train_tokenizer.py`; original commit `e0408a3487c0d6244ffab7ce579a278ca1a0cd8e`, hash `7bd6b759…` | Five F files in recorded order plus exclusion/release manifests → the shipped 32k BPE. Verifier source is recovered separately without an invented historical Git binding. |
| I selection | `data_selection/pretrain/stage_i_realize_v1.py` and local module family | Frozen graph/provenance/H token accounting and tokenizer → immutable selected plaintext; A 10,000,003,234 / B 3,000,004,240 serialized tokens. Sources matched to original 3e6994a8 implementation identities. |
| M packing | `packing/pretrain/stage_m_realize_v1.py` and local module family | Exact accepted I order → uint16 arrays; A 10,000,003,073 / B 3,000,002,561 packed tokens. Original producer identity ea081327. |

F records contain `text`, `canonical_source`, `canonical_release_id`,
`physical_row_index`, `cleaned_text_sha256`; text is lossless, ranking uses the
frozen BLAKE2b128 tuple and no new deduplication. A **schema illustration only** is:

```json
{"text":"Hello.","canonical_source":"synthetic","canonical_release_id":"illustration-only","physical_row_index":0,"cleaned_text_sha256":"<SHA256 of UTF-8 text bytes>"}
```

This placeholder is not a valid frozen F row or input release. Read the recorded
source identities/revisions in [tokenizer_corpus_selection.json](configs/research-v1/tokenizer_corpus_selection.json),
[data_selection_plan.json](configs/research-v1/data_selection_plan.json) and
[packing_plan.json](configs/research-v1/packing_plan.json). DCLM, FineWeb, Python,
structured/tutorial and Wikipedia identities are bound there; missing upstream
revisions must not be replaced with current defaults. G argv and file order are
in [tokenizer_environment_argv.json](configs/research-v1/tokenizer_environment_argv.json).

Packing uses `[BOS] content [EOS]`, empty-text separators, no padding, 2048-position
blocks and a final lookahead per stage. It drops 1,840 tail tokens; there are
6,347,659 sequences. Packed totals and optimizer-consumed positions are different.
The reference reserve exclusions/G2 validation release are not a random split
that may be regenerated with an arbitrary seed.

The maintained generic [shard builder](pretrain/build_pretrain_shards.py) accepts
explicit `--source path:weight`, `--tokenizer_path` and `--out_dir`; it is not Stage M.
The [tokenizer trainer](tokenizer/tokenizer_training/train_tokenizer.py) accepts the
real flags shown by its `--help`. Safe interface inspection:

```bash
python "$REPO/tokenizer/tokenizer_training/train_tokenizer.py" --help
python "$REPO/pretrain/build_pretrain_shards.py" --help
```

The recorded I and M command shapes are `python pretrain/stage_i_realize_v1.py run
--plan <original-plan> --expected-plan-sha256 <original-sha> --out-dir <new-output>
--repo-root <matching-source-root>` and the corresponding `stage_m_realize_v1.py`.
I's plan hash is `3effe6300c3f383dd788d22b4146064b98f620bed85e5073a5229f5fb26d5e56`;
M's is `6c7af2bdaf00b12fd62e8af0f26960ed2c12317cd87ce290f9b1de055e7b1d21`.
These are **recorded shapes, not supported public replay commands**: the projected
JSON files have different bytes and cannot satisfy the original hashes. F's complete
historical argv is unknown. Exact raw parents, D2/D3 outputs, L1 exclusions, accepted
I data and immutable path/source bindings remain unavailable. Selection/packing
must fail on mismatches; no public adapter bypasses these contracts.

## Pretraining A → B → Base49590

Use the separate [A source root](recipes/research-v1/sources/pretrain_stage_a/) and
[B source root](recipes/research-v1/sources/pretrain_stage_b/). Both contain
`pretrain/train_pretrain_with_bench.py`, their dataset, launch/run contracts, model,
optimizer and schedule. A's original execution closure at
`6d80423adc16d4a160a7fe42660020c585b5185d` matches historical bundle `bbd49b9d…`
even though run metadata said `git_dirty=true`. B's closure at
`7686fd811642dd6246ca3a3c21a4bf43bc28cd3b` matches `1086af0b…`.
See [execution bindings](configs/research-v1/TRAINER_EXECUTION_BINDINGS.json) for
complete file hashes; relocated candidate bytes are separately identified.

A starts without resume and stops at 38,146. **Use
[pretrain_stage_a_initial_effective.json](configs/research-v1/pretrain_stage_a_initial_effective.json)
for the initial launch.** The similarly named `pretrain_stage_a_effective.json`
records endpoint recovery with resume/stop at 38,146, not another training pass.
B consumes the recorded successor-head compatibility bridge with `--resume_full`,
`--resume_step 38146`, `--strict_resume_contract`, `data_stage_start_step=38146`
and absolute stop 49,590. Preserve optimizer/scaler/RNG/sampler state. Weight-only
initialization is not the transition. Missing state cannot be reconstructed from logs.

Effective parameters: microbatch 8 × accumulation 16 × sequence 2048 = 262,144
positions/update; seed 20260831, A/B sampler seeds 20260832/20260833, validation
seed 20260834. Muon matrix groups plus auxiliary AdamW use LR 6e-4, weight decay
0.1, clip 1, 500 warmup, one absolute WSD schedule to 49,590 with decay
44,631–49,590 and minimum ratio 0.1. `muon_lr=0` resolves to main LR. See
[contract semantics](configs/research-v1/pretrain_contract_semantics.json) for
betas, eps, grouping, masking and checkpoint/evaluation milestones.

Inputs are `toks[:-1]`, labels `toks[1:]`; BOS and later consecutive EOS labels are
masked, valid final labels are retained, EOS weight is 1. No schedule or optimizer
reset at A/B. Across A/B the optimizer consumed 12,999,720,960 positions.
The selected Base is `step_049590.pt`, SHA256
`95ae1201a5ce8a6f838e8e131c59b8779d639f4ec4cf4dbc553928b9b754b86e`.

The recorded complete `cmd`/`args` in A initial and
[B effective launch](configs/research-v1/pretrain_stage_b_effective.json) are checked
against source flags. There is **no supported portable full A/B launch yet**:
original M/G2 arrays, exact launch/authorization/source contracts and Stage A/bridge
full state are external. The public [shared trainer](pretrain/train_pretrain.py)
remains useful reusable code, but its generic invocation is not this historical run.

## P2: released ancestor, portable path adapter

[P2 source](recipes/research-v1/sources/posttraining/sft/train_sft.py) is the recovered
`997201f1…` trainer with its local `p2_plan.py`, `p2_loss.py` (`ce269097…`) and
`p2_evaluation.py`. The data preparation/finalization scripts and their earlier
pilot helper are under the same source root's `runs/`. Top-level builders match
recorded original digests; the pilot helper's execution-era digest is unresolved.

P2 uses 12,000 train / 500 validation reviewed smol-smoltalk-derived conversations,
pinned collection revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, default/train.
Seven source labels are labels within that collection, not independently pinned
repository configs. Frozen selection/review/defer records and rows are not public.
See [input identities](configs/research-v1/p2_input_identities.json).
Rows preserve `messages`, `audit_id` and `shifted_supervised_tokens`; the two small
[synthetic examples](recipes/research-v1/examples/messages.synthetic.jsonl) illustrate
only messages, not a replacement 12k/500 audited dataset or benchmark.

Every assistant turn's content and EOS are supervised; prompts, roles and BOS are
masked. No injected system or truncation in source-preserving mode. This is what the
recovered implementation establishes; terse “final-assistant-turn” descriptions in
frozen report tables do not override the source. Those historical tables are preserved.

P2 initializes from Base **weights only**, inheriting tied model config, with fresh
AdamW (betas .9/.95, eps 1e-8), LR 5e-5, weight decay .1 by ndim≥2, warmup 38,
clip 1, microbatch 2 × accumulation 16, context 2048, seed 20260906, two workers,
750 updates. The effective-batch loss divides summed CE by all shifted assistant
targets across microbatches. Forward is BF16; selected-logit CE is FP32. The frozen
plan covers two epoch permutations: 24k row exposures / 1,994,854 target observations.
Evaluation and saves at 375/750 cover the full validation set. Selected output:
`step_000750.pt`, SHA256 `20afc40096568343c59d8a55e9606667c279f5f1c976908c57de18a0107f4e4e`.

This **new public adapter**, not the original executable, supports configurable roots:

```bash
python "$REPO/recipes/research-v1/p2.py" \
  --train-jsonl "$DATA_ROOT/P2_TRAIN.jsonl" \
  --validation-jsonl "$DATA_ROOT/P2_VALIDATION.jsonl" \
  --base-checkpoint "$DATA_ROOT/step_049590.pt" \
  --original-consumption-plan "$DATA_ROOT/CONSUMPTION_PLAN.json" \
  --out-dir "$OUTPUT_ROOT/p2" --validate-only
```

It verifies original train/validation/Base/tokenizer/plan SHA256, row counts and
supervised target counts, disjoint audit IDs and complete deterministic plan equality.
Only `train_path` in a temporary copy of the verified plan changes; all ordering,
row identities, exposures and target counts remain exact. Output roots must be fresh.
It exits before model initialization. Missing inputs are errors, not skipped contracts.
A future authorized training run would replace `--validate-only` with `--execute`;
the unchanged recovered trainer then performs 750 updates, writes its checkpoints,
metrics/preflight/status plus `CONSUMPTION_PLAN.public.json` and
`PUBLIC_PATH_BINDINGS.json`. There is no new resume mode. Local path strings affect
provenance/checkpoint serialization, so byte-identical checkpoint files are not promised.

Validation here establishes CLI/source/config and synthetic contract behavior only;
the exact-data branch could not run without the omitted input files. It does not
certify full historical input authorization/review or claim reproduced P2 scores.

## P3 and fixed interpolation

[P3 runtime](recipes/research-v1/sources/posttraining/runs/p3_basic_instruction_generalization_20260907/runtime/)
contains recovered `prepare.py`/`execute.py`; original/candidate hashes differ by
handoff path redaction. The recorded launch in [p3_launch.json](configs/research-v1/p3_launch.json)
is `python -u .../runtime/execute.py train` with frozen `RUN_CONFIG`/`UPDATE_PLAN`.
This is **not a portable public launch**: imports of private `checkers`/`curriculum`,
immutable file/source hashes and their 10,240-row plan remain required. No substitute
curriculum is provided. Frozen dependency names/hashes are in
[p3_frozen_bindings.json](configs/research-v1/p3_frozen_bindings.json).

P3 uses P2 weights only and fresh AdamW (.9/.95, eps1e-8, weight decay0), LR5e-5,
32 warmup, cosine over 640 updates, min ratio .1; micro2×accum16, sequence512 inside
the unchanged 2048 architecture, seed20260907, four CPU threads, zero workers,
two passes and 7200s process cap with no automatic restart. Each block has seven
procedural updates then three replay updates (7,168 procedural + 3,072 replay rows).
Loss is effective-update assistant-target mean including EOS, BF16 forward with
selected FP32 CE. Checkpoints include model/optimizer/config/step/scheduler/RNG
and data/plan hashes. Step320 is the blend parent; adverse step640 remains reported.

[Interpolation source](recipes/research-v1/sources/posttraining/runs/p3_retention_two_point_interpolation_20260907/runtime/)
fixes parents P2step750 and P3step320
(`b11cb018d9f7382e46088f587dc62db9bc25c7d696d399e6b8a001f7ca9cb4e2`).
It computes CPU FP32 `A + alpha*(B-A)` for exactly .50/.75, checks configurations,
finite values, fixed buffers and tied parameter aliases, with no optimization.
Recorded launch: [interpolation_launch.json](configs/research-v1/interpolation_launch.json).
Selected `weights/alpha075.pt` hash is
`1da85cc329d55e92dacf51c36623779558c4c6c9a39d78a34a064f61fcddbe97`.
It is an inference weight artifact, not a resumable optimizer state.
The recovered script still needs parents and preparation manifests/directories;
a public path port is not completed for those unavailable bindings. Its `.50`
control and P3step640 negative evidence have not been removed or rescored.

## Native export

The recovered [export scripts](recipes/research-v1/sources/posttraining/runs/alpha075_native_inference_export_v1/evidence/scripts/)
consume alpha075 and [export bindings](configs/research-v1/export_bindings.json).
`export_weights.py` runs work at module startup: **do not invoke it with `--help`**.
It constructs/strictly loads the source model and safetensors reload, checks all
213 named state entries and tied storage, then emits `model.safetensors`, config,
metadata and equality evidence. Historical weight SHA256 is
`4396efb7a52b047e7fdf513e46d1b401dfc70582d3aca1f9cb5a07e97d426ef1`.
Eight recorded source/export precision-profile pairs matched; no new parity run
was performed. `build_bundle.py` is explicitly a source excerpt, not complete
publication orchestration. A runnable public export port remains blocked on the
frozen INPUT_BINDINGS/evidence layout and source checkpoint; do not pass the
projected config as the original binding. Existing released inference requires
the complete published bundle in the [run guide](docs/petitgpt-v1/RUN_GUIDE.md).

## Published evaluation, separate from generated-answer review

The recovered [benchmark runtime](recipes/research-v1/sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/)
contains `prepare_v2.py`, `evaluate_v2.py`, `likelihood_checks.py`, `common.py` and
`aggregate_report.py`. Licensed pinned harness helper sources are adjacent under
`source/harness/`. It extracts specific helper methods via AST from harness commit
`b954108c9baaaa934b4ad842033b31a97ee30816`; it is **not a full installed-harness run**.

[Protocol](configs/research-v1/benchmark_protocol.json) binds:

| Task | Source revision and split | Aggregation |
|---|---|---|
| ARC-Easy | `allenai/ai2_arc`, `210d026faf9955653af8916fad021475a3f00453`, ARC-Easy/test, 2,376 physical rows | `acc`, `acc_norm` |
| PIQA | `baber/piqa`, `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1`, validation, 1,838 physical rows | `acc`, `acc_norm` |

The protocol JSON includes parquet/canonical row hashes; [model identities](configs/research-v1/benchmark_models.json)
pin alpha075 and both external baselines. FP32 parameters/forward/log-softmax,
autocast/TF32 off, MATH SDPA, unpadded batch1, zero-shot raw completion, no chat/BOS
or scored EOS, first-maximum ties. `acc_norm` divides by the original candidate's
Unicode-character length, not token count. Boundary and per-token likelihood
checks accompany aggregate outcomes. The unchanged
[public benchmark table](docs/petitgpt-v1/tables/PUBLIC_BENCHMARK_RESULTS.csv)
contains the historical results; no synthetic or modern-harness scores were added.

The actual evaluator parser accepts positional `phase` (`checks` or `full`) and
`model`; it has no generic `--tasks`/`--model_path` harness CLI. Full historical argv
is not recovered. Requests, `runtime/CODE_FREEZE.json`,
`preparation/FROZEN_V2_FILES.json`, forward events, outcomes and other runtime inputs
are external; aggregation preserves incomplete status and protocol versions.
Source identity of aggregation is recorded without claiming an unavailable
historical freeze binding. Public benchmark replay/path portability remains blocked.
Do not import the runtime for help: imported code performs setup and other scripts
have startup I/O. Use `validate.py` for static inspection.

Complete generated-answer review is separate:
[ASSISTANT_RESULTS_VERSIONED.csv](docs/petitgpt-v1/tables/ASSISTANT_RESULTS_VERSIONED.csv)
and report methodology distinguish content, formatting, interface and whole-answer
correctness. Exact prompts, rubric freezes, labels and raw outcomes are not all public.
A Python interface pass or teacher-forced NLL cannot stand in for complete-answer
correctness. The fixed ARC/PIQA diagnostics were not untouched final tests.

## Remaining blockers and subsequent validation

1. **F/I/M and A/B:** exact raw releases, D2/D3/L1/H/G2 artifacts, original immutable
   plans/launch contracts, and full Stage A/bridge state. Path-sanitized projections
   cannot satisfy original byte hashes. No bypass/optimizer reset is acceptable.
2. **P2:** exact reviewed 12k/500 rows, original consumption plan and Base are external.
   The path adapter is implemented; real-data validation and historical review-input
   restoration remain unexecuted. Public redistribution needs a separate explicit
   distribution review; no run-specific data were added here.
3. **P3:** private curriculum/checkers, frozen config/update/data/evaluation contracts,
   missing runtime helpers and exact parent. Source recovery does not close these gaps.
4. **Interpolation/export/benchmark:** original runtime input/code/parity/request
   manifests and weights, plus reviewed public path adapters for their fixed layouts.
   Some handoff redactions invalidate frozen hashes/environment path guards.
5. **Generated answers:** exact held-out rows and versioned adjudication artifacts are
   not fully public. The frozen scores remain evidence, not newly reproducible output.

No model construction, training, inference, scoring, checkpoint loading/conversion,
GPU use, teacher API or model/data download was part of these checks. No full dataset
preparation was run. A later training smoke would need a separate authorized recipe
with explicit update/time/memory bounds after its inputs and contract are available;
this change does not weaken the fixed historical P2/other stage contracts to create
one. Full historical reproduction requires substantially more than a smoke test.
