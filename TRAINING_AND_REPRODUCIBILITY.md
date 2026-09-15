# Train and evaluate PetitGPT from prepared inputs

If you are interested in the realization details, this manual describes how to **reproduce the research-v1 method from the beginning**
with your own prepared local data. The `reader.py` command-line tool connects the
recovered research-v1 trainers and fixed native model through this workflow:

| Stage | Purpose |
|---|---|
| A → B pretraining | Learn next-token prediction from two packed text corpora; B continues A's full training state. |
| P2 supervised fine-tuning | Train concise instruction responses from conversations. |
| P3 procedural fine-tuning | Practice exact copying (COPY), field lookup (FIELD), set membership (MEMBERSHIP), and structured JSON output (JSON), mixed with instruction replay examples to help retain earlier skills. |
| Fixed interpolation | Combine P2 step 750 and P3 step 320 as `P2 + 0.75 * (P3 - P2)`; `alpha075` names this coefficient. |
| Export → evaluation | Package native weights and inference code, then score local ARC-Easy/PIQA multiple-choice tasks by answer likelihood. |

New data/checkpoints have new hashes and results. Published measurements for the
released alpha075 remain in the [technical report](docs/petitgpt-v1/TECHNICAL_REPORT.md).

Read [model/tokenizer](tokenizer/README.md) → [prepared input schemas](recipes/research-v1/PREPARED_INPUTS.md)
→ [pretrain](#pretrain-ab) → [posttrain](#posttraining) → [likelihood evaluation](#export-and-evaluate). The shared `pretrain/train_pretrain.py`
is useful research tooling but is not the trainer used by this route.

## Setup and conventions

The execution reference is Python 3.10.12, torch 2.11.0+cu126, NumPy 2.2.6,
tokenizers 0.22.2 and safetensors 0.8.0. Training and the recorded likelihood profile
require a BF16-capable CUDA GPU. See
[recorded requirements](recipes/research-v1/runtime-support-v2/frozen_native/requirements-inference-tested.txt).

### Resource scale

The historical run used one RTX 4090. Its recorded Stage A log span was about
29.8 hours; a Stage B window covered 2.994 billion positions in 32,403.5 seconds
(about 9 hours). These are historical elapsed windows, not measured GPU hours for
this new-run workflow; see [throughput and duration](docs/petitgpt-v1/TECHNICAL_REPORT.md#44-observed-results).
Peak training VRAM and end-to-end new-run GPU hours have not been measured here,
so there is no verified minimum VRAM or monetary cost estimate.

The fixed A/B schedule processes `49,590 × 128 × 2,048 = 12,999,720,960` token
positions. The minimum packed A/B payload alone is about **26.0 GB (24.2 GiB)**,
derived from the required block counts and two bytes per token. Allow additional
space for raw data, validation, optimizer checkpoints, intermediate weights and
export archives; this payload figure is not a total disk-space budget.

### Commands and output directories

```bash
export REPO=/path/to/your/petitgpt-checkout
export DATA=/path/to/prepared-inputs
export OUT=/path/to/new-runs
export TOK="$REPO/tokenizer/releases/tokenizer_v1/tokenizer.json"
export PYTHONDONTWRITEBYTECODE=1
cd /tmp
python -B "$REPO/recipes/research-v1/reader.py" --help
python -B "$REPO/recipes/research-v1/reader.py" p3 --policy new-run --help
```

Use absolute paths for `REPO`, `DATA` and `OUT`. `cd /tmp` is an optional working
directory choice: the scripts find sibling modules from their own location, so
another directory works too. `PYTHONDONTWRITEBYTECODE=1` and `-B` suppress Python
bytecode caches (`__pycache__`); the environment variable also applies to child
Python processes. Neither changes the training method.

All stage examples below use `--validate-only`: they check inputs without loading
weights or creating the requested output directory. To run a stage, use its same
arguments with `--execute` instead, then proceed to the next stage. A downstream
validation needs the checkpoint and metadata actually produced by its predecessor;
running all examples with `--validate-only` cannot create that chain.
Each stage output directory must be nonexistent, even for validation.
If you see `Requires a fresh nonexistent output directory`, choose a new stage
directory (and update downstream paths), or a new `OUT` for a complete rerun.
Do not pre-create the per-stage directories; their parent may already exist.
`reader_prepare.py --write` creates prepared files only; it does not train.
No command downloads data/models or calls a teacher/API.

`--policy new-run` accepts your own schema-compatible inputs. `--policy historical`
uses the original input identities and flags for P2, P3, interpolation and export;
see [advanced historical material](recipes/research-v1/POSTTRAINING.md).
V2 and V3 in repository records refer to the historical adapters and the newer
`reader.py` interfaces, respectively. A checkpoint's adjacent `.reader.json` file
is its metadata sidecar (also called a receipt or declaration): it records its
hash, configuration, kind and step for downstream checks.

## What you supply

| Input | Exact new-run requirement |
|---|---|
| Tokenizer | Canonical `tokenizer/releases/tokenizer_v1/tokenizer.json`; 32k vocabulary, IDs 0–6 PAD/UNK/BOS/EOS/system/user/assistant |
| Packed A/B/validation | Pass split directories: `$DATA/A/train`, `$DATA/B/train`, `$DATA/reference/val`. Each contains `*.bin` shards and has `meta.json` in its parent directory. Little-endian uint16, exact sorted shard inventory/hashes/counts and tokenizer identity; see prepared-input schema |
| P2 | Original-sized default: 12,000 train and 500 validation message objects. Unique IDs, disjoint IDs/exact messages across splits, ≤ 2,048 encoded tokens and exact shifted assistant/EOS target counts |
| P3 | 7,168 supplied procedural records (1,792 each COPY/FIELD/MEMBERSHIP/JSON) and 3,072 replay records; disjoint IDs/exact messages, ≤ 512 tokens. You supply task content; no private template bank or held-out labels needed |
| Evaluation | Explicit local ARC-Easy/PIQA JSONL, task revisions/splits/row counts/file hashes, and scope synthetic/subset/full-local |
| Existing checkpoint | Compatible complete native config and adjacent `.reader.json` declaration. Actual tensors/step are checked when loaded; metadata validation alone does not prove them |

P2 derives a two-pass plan: concatenate two deterministic shuffled passes through
the supplied rows, group them into 32-row updates, and cap the run at 1,000 updates.
Thus 12,000 rows give 750 updates and 38 warmup updates; other valid populations
can produce a different-length run. The P3/interpolation workflow below requires
**P2 step 750**, so use 12,000 training rows for this route.
P3 retains 640 updates and fixed population/batch geometry. These size requirements
are public method settings. The 500-row P2 validation split matches the original
size; the new-run validator accepts other nonempty validation sizes.

### Prepare messages and the P3 plan

**Raw message JSONL must be annotated before training or plan preparation.** Each
row needs an exact `shifted_supervised_tokens` count, computed by the canonical
tokenizer for assistant content and EOS targets. The commands below annotate all
four splits; if your files already satisfy the prepared schema, you can use them
directly. P3's 512-token limit is checked when preparing its plan and validating
training inputs; the general `messages` writer checks a 2,048-token limit.

```bash
python -B "$REPO/recipes/research-v1/reader_prepare.py" messages \
  --input "$DATA/p2-train.raw.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p2-train-prepared" --write
python -B "$REPO/recipes/research-v1/reader_prepare.py" messages \
  --input "$DATA/p2-val.raw.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p2-val-prepared" --write
python -B "$REPO/recipes/research-v1/reader_prepare.py" messages \
  --input "$DATA/p3-train.raw.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p3-train-prepared" --write
python -B "$REPO/recipes/research-v1/reader_prepare.py" messages \
  --input "$DATA/p3-replay.raw.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p3-replay-prepared" --write

python -B "$REPO/recipes/research-v1/reader_prepare.py" p3-plan \
  --input "$OUT/p3-train-prepared/messages.jsonl" \
  --replay "$OUT/p3-replay-prepared/messages.jsonl" \
  --tokenizer "$TOK" --out-dir "$OUT/p3-plan" --write
```

Separate plan preparation is optional: training derives and checks its plan
automatically. This example prepares P3's plan explicitly and passes it with
`--plan` below to require exact equality.

For A/B inputs starting from selected text JSONL, use `reader_prepare.py packed
--split train`; use `--split val` for validation. Its `--out-dir` is the release
root (for example, `$DATA/A`), and training consumes the generated split directory
(`$DATA/A/train`). See the [packed preparation example](recipes/research-v1/PREPARED_INPUTS.md#packed-pretraining).

Preparation methods, schemas and synthetic format examples are in
[PREPARED_INPUTS.md](recipes/research-v1/PREPARED_INPUTS.md). Existing F/G/I/M source
methods (tokenizer corpus selection/training, corpus selection, and packing) are
also linked there.

## Pretrain A/B

```bash
python -B "$REPO/recipes/research-v1/reader.py" pretrain --policy new-run \
  --stage stage_a --stage-a-dir "$DATA/A/train" --stage-b-dir "$DATA/B/train" \
  --validation-dir "$DATA/reference/val" --tokenizer "$TOK" \
  --out-dir "$OUT/A" --validate-only

# After actually executing A, hand over its complete checkpoint:
python -B "$REPO/recipes/research-v1/reader.py" pretrain --policy new-run \
  --stage stage_b --stage-a-dir "$DATA/A/train" --stage-b-dir "$DATA/B/train" \
  --validation-dir "$DATA/reference/val" --tokenizer "$TOK" \
  --resume "$OUT/A/step_038146.pt" --out-dir "$OUT/B" --validate-only
```

A runs steps 1–38,146; B continues 38,147–49,590 with **full optimizer, RNG and committed
sampler state**.
The source's strict resume checks remain: same config/optimizer/schedule/runtime,
sampler commitments and exact stage boundary. New-run plan validation substitutes
only the old private provenance/approval layer. Its separate schema binds both
stage inventories and shared validation; the sampler-seed change is allowed only
from A's seed 20260832 to B's seed 20260833 at step 38,146. Same-stage restart uses
`--resume` and a fresh output directory, with unchanged data/plan/environment. Checkpoints get adjacent
receipts when actually saved, so completed checkpoints survive a later interruption.
The recovered data contract also records file mtimes: preserve the prepared data
files when resuming; relocation that changes this fingerprint is rejected.

| Pretraining setting | Value |
|---|---|
| Architecture | 30 layers; width 576; feed-forward width 1,536; grouped-query attention with 9 query heads and 3 key/value heads |
| Context | 2,048 tokens |
| Batch | Microbatch 8 × gradient accumulation 16 = 128 sequences/update |
| Seeds | Model 20260831; validation 20260834; samplers A 20260832 / B 20260833 |
| Optimizer | Fresh Muon plus the source's AdamW groups at A initialization; full state carried into B |
| Peak learning rate / weight decay / gradient clip | 0.0006 / 0.1 / 1 |
| Precision / compilation | BF16 / enabled |
| Warmup–stable–decay (WSD) schedule | Absolute horizon 49,590; warmup 500; stable through 44,631; decay to 49,590; floor 0.1 × peak learning rate |
| Minimum unique blocks | A: 4,882,688; B: 1,464,832; validation: at least 1 |

The source dataset traverses global contiguous
2,049-token windows at stride 2,048, across shard boundaries. BOS and repeated EOS
masking, final-label coverage, position-weighted loss and no-replacement sampler
are source-derived. Incomplete tails are unused.

Optional in-run validation/generation hooks are disabled in this workflow and
recorded as omitted. Explicit checkpoint milestones and final checkpoint retention
are preserved. Outputs include
`READER_INPUTS.json`, source metrics/run/data metadata, `step_*.pt`, `latest.pt`,
adjacent checkpoint receipts and a completion status only after the trainer returns.

## Posttraining

```bash
python -B "$REPO/recipes/research-v1/reader.py" p2 --policy new-run \
  --parent "$OUT/B/step_049590.pt" \
  --train-jsonl "$OUT/p2-train-prepared/messages.jsonl" \
  --validation-jsonl "$OUT/p2-val-prepared/messages.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/P2" --validate-only

python -B "$REPO/recipes/research-v1/reader.py" p3 --policy new-run \
  --parent "$OUT/P2/step_000750.pt" \
  --train-jsonl "$OUT/p3-train-prepared/messages.jsonl" \
  --replay-jsonl "$OUT/p3-replay-prepared/messages.jsonl" \
  --plan "$OUT/p3-plan/PLAN.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/P3" --validate-only

python -B "$REPO/recipes/research-v1/reader.py" interpolate --policy new-run \
  --p2-parent "$OUT/P2/step_000750.pt" --p3-parent "$OUT/P3/train/step_000320.pt" \
  --tokenizer "$TOK" --out-dir "$OUT/blend" --validate-only
```

P2 writes checkpoints directly in `$OUT/P2/`; P3 writes them in `$OUT/P3/train/`.
The extra `train/` in the interpolation command is intentional.

| Posttraining setting | P2 (12,000 training rows) | P3 |
|---|---|---|
| Initialization | B step 49,590 weights | P2 step 750 weights |
| Updates / passes | 750 / 2 | 640 / 2 |
| Context | 2,048 tokens | 512 tokens |
| Microbatch × gradient accumulation | 2 × 16 = 32 rows/update | 2 × 16 = 32 rows/update |
| Seed | 20260906 | 20260907 |
| Optimizer | Fresh AdamW; betas 0.9/0.95; epsilon 1e-8 | Fresh AdamW; betas 0.9/0.95; epsilon 1e-8 |
| Peak learning rate | 5e-5 | 5e-5 |
| Weight decay / gradient clip | 0.1 (matrix parameters) / 1 | 0 / 1 |
| Warmup updates | 38 | 32 |
| Checkpoints | Steps 375 and 750 | Steps 320 and 640 |

P2 uses the [recovered effective settings](configs/research-v1/p2_effective_launch.json)
and deterministic two-pass plan. P3 uses a cosine schedule with horizon 640 and
floor 0.1 × peak learning rate. Each shuffled ten-update block contains seven
procedural and three replay updates, with two exposures per row. Both use
BF16 forward; loss is FP32 selected-logit assistant/EOS cross-entropy (CE),
normalized by the **whole effective update's**
target count. Both stages check the actual loaded architecture,
state keys/shapes/FP32 values, tied aliases and parent step before training.

The interpolation workflow selects **P3 step 320 of the 640-update schedule**;
rescheduling cosine to end at 320 changes the method. New-run P3 omits the private
baseline, development, 500-row validation and final evaluation hooks.
P2 retains evaluation on the supplied public-format validation split. Materialized
P2 and P3 do not expose resume; start a fresh run if interrupted.

New-run P3 writes each adjacent `.reader.json` immediately after that checkpoint's
finalized save. Step 320 can therefore pass downstream metadata validation even if
later work fails; this does not mark the run complete or enable resume. Receipt
write errors stop the run.

Plans are derived and checked during validation; execution writes `PLAN.json` (P2)
or `PLAN.jsonl` (P3). Pass `--plan` to require exact equality with a separately
prepared plan. P3 records its new `INPUTS.json` identity in checkpoint metadata.

Interpolation defaults to the fixed 0.75 formula, yielding
`weights/new_alpha075.pt`. `--include-alpha050` additionally records the fixed 0.50
control. The complete recovered CPU FP32 `P2 + alpha * (P3 - P2)`
implementation checks configurations, shapes/dtypes, aliases, fixed/nonfloating and
native buffers, finite values, endpoints, FP64 oracle tolerance, strict save/reload
and parent non-mutation. New tensors are compared to their actual parents and
serialized output.

## Export and evaluate

```bash
python -B "$REPO/recipes/research-v1/reader.py" export --policy new-run \
  --parent "$OUT/blend/weights/new_alpha075.pt" --tokenizer "$TOK" \
  --out-dir "$OUT/native" --validate-only

python -B "$REPO/recipes/research-v1/reader.py" evaluate --policy new-run \
  --model "$OUT/native/bundle" --tasks "$DATA/tasks.json" --tokenizer "$TOK" \
  --out-dir "$OUT/likelihood" --validate-only
```

Export validates the source's full native tensor schema (213 FP32 state entries),
writes 212 unique safetensors entries with tied alias metadata, reloads and compares
every source/output dtype/shape/value. Frozen inference assets and the canonical
tokenizer are copied byte-for-byte. Outputs:
`bundle/`, new provenance/evaluation index/manifest, verified tar.gz and checksum.

Likelihood uses the pinned harness `_encode_pair` method, raw
`Question: ...\nAnswer:` plus one leading space before each exact candidate,
no chat/BOS/EOS insertion, complete causal continuation alignment, FP32
parameters/forward/logsoftmax/sum, MATH SDPA and unpadded batch 1. `acc` selects the
candidate with the highest summed log-likelihood; `acc_norm` selects using that
score divided by Python Unicode-character length of the original answer (excluding
the delimiter). Ties select the first candidate. The evaluator accepts local
checkpoint files or native bundles.
Outputs: `INPUTS.json`, per-row `ROWS.jsonl` scores/predictions and per-task
`RESULTS.json`. Revisions, original file hashes, normalized row identities and
scope are recorded. `full-local` means all supplied rows.

For a compatible pre-existing local checkpoint, create its declaration with
`reader.py bind-checkpoint --policy new-run --checkpoint PATH --kind pretrain
--step 49590 --tokenizer "$TOK"`. The sidecar records the actual file hash and a
compatible config declaration, **not tensor verification or approval**. Actual
loaded state and step must pass execution checks. Automatically produced reader
outputs already include this sidecar.

## Validation status and interpretation

<!-- The new interfaces are implemented.  -->
Targeted CLI/import/schema/encoding/plan and small synthetic CPU tensor/serialization
tests have run; see the [input and stage tests](tests/test_reader_v3.py),
[P3 boundary tests](tests/test_p3_reader_boundary.py) and
[checkpoint loader tests](tests/test_reader_numpy_loader.py).

<!-- The subsequent loader repair passed 13 synthetic regression cases plus the 40
accepted reader/P3 boundary cases. These include both fixed NumPy reconstruction
pickle names, restricted-object rejection and unchanged schema/step checks. -->

<!-- Only NumPy 1.26.4 was actually tested for this repair; simulated serialized names
do not establish a NumPy 2.x environment test or universal version support.

The accepted RunPod P2 validation PASS and P3 original data/640-plan checks are
supplied historical input evidence. They are not this new-run implementation's
training results. The published negative/retention findings remain unchanged.
The new local likelihood interface closes the missing public callable entrypoint;
full historical benchmark/private-answer-review orchestration remains separate. -->

<!-- On 2026-09-12 the first bounded CPU export attempt failed before `torch.load`
because the loader accessed an unloaded NumPy compatibility submodule. After the
explicit-import/scoped-allowlist repair, one separately authorized additional
attempt passed through the public `export --policy new-run --execute` entrypoint.
It used the existing 498,608,319-byte historical alpha075, not a new reader-trained
blend. The prior failure remains a separate result. The staged blend's step0 is
the model-only container convention, not zero historical training updates. -->

<!-- The tested stack was Linux x86_64, Python 3.11.7, torch 2.11.0+cu130, NumPy 1.26.4,
tokenizers 0.22.1 and safetensors 0.6.2, with two CPU threads and no dependency
changes.  -->

<!-- This differs from the historical execution reference above. The successful
attempt took 36.49 seconds; measured peak RSS was 1,838,408 KiB. Limits were
180 seconds, 8 GiB virtual memory and less than 4 GiB of new working files. -->

In a CPU re-export of the existing historical alpha075, the source passed complete
native config and 213 FP32 state-entry checks.
The exporter stored 212 entries plus tied-alias metadata, reloaded its own output,
and compared source/output dtypes, shapes and values. Tokenizer/frozen asset hashes,
all 17 bundle-manifest entries and all 18 native archive members also matched.

<!-- This verifies a temporary re-export of existing alpha075 in this environment;
it does not replace the release or claim historical safetensors file-byte equality.
No full GPT was constructed. Reader training, P3 real runtime, forward/backward,
optimization, inference/generation parity, scoring and GPU work were not run.
CPU conversion is not CPU inference support or end-to-end training validation. -->

This verifies source-to-export tensor equality for that existing checkpoint.
End-to-end new-run training, real P3 execution, GPU scoring and generation parity
remain unverified here. The P3 boundary checks use synthetic files and stub
callbacks; they do not establish runtime validation of the separate V2 interface.

New-run metadata describes your supplied data and newly produced checkpoints.
Historical approvals and benchmark orchestration remain in the separate historical
interfaces. Changed data and omitted evaluation hooks can change the training
trajectory; matching settings or using `full-local` evaluation does not establish
the published result. Use the technical report's scores as historical observations,
not an expected range for a new dataset.

## What a completed run produces

After executing each stage successfully, the examples above leave these key files:

| Location under `$OUT` | Contents |
|---|---|
| `A/step_038146.pt`, `B/step_049590.pt` | Full-state pretraining checkpoints for handover/resume |
| `P2/step_000750.pt` | Instruction-tuned P2 checkpoint |
| `P3/train/step_000320.pt`, `P3/train/step_000640.pt` | Selected interpolation parent and final P3 checkpoint |
| `blend/weights/new_alpha075.pt` | Your fixed 0.75 blend |
| `native/bundle/` | `model.safetensors`, tokenizer, frozen inference code and provenance; archive and SHA256 checksum in `native/` |
| `likelihood/` | `INPUTS.json` input identities, `ROWS.jsonl` per-row candidate scores/predictions, and `RESULTS.json` per-task accuracy |

Checkpoints also have adjacent `.reader.json` sidecars. Training, interpolation
and export write `READER_STATUS.json` with `execution: "COMPLETED"` after returning
successfully; likelihood evaluation records completion in `RESULTS.json`.
File presence alone, especially a P3 step-320 checkpoint, does not mean the whole
run finished.

`RESULTS.json` contains `results.<task>.rows`, `acc` and `acc_norm`; accuracies are
fractions from 0 to 1. `ROWS.jsonl` lets you inspect which candidates won under raw
and length-normalized likelihood. Report the measured values together with the
task revision, split and scope recorded in `INPUTS.json`.
