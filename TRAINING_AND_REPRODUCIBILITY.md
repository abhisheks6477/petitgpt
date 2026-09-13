# Train and evaluate PetitGPT from prepared inputs

For interested readers, this manual describes a way to **reproduce the research-v1 method from the beginning**. One provides prepared
local data; the adapters reuse the recovered research-v1 trainers and fixed native
model. New data/checkpoints have new hashes and new results. These are not the released
alpha075. Published results remain in the unchanged [technical report](docs/petitgpt-v1/TECHNICAL_REPORT.md).

Read [model/tokenizer](tokenizer/README.md) → [prepared input schemas](recipes/research-v1/PREPARED_INPUTS.md)
→ [pretrain](#pretrain-ab) → [posttrain](#posttraining) → [likelihood evaluation](#export-and-evaluate). The shared `pretrain/train_pretrain.py`
is useful research tooling but is not the trainer used by this route.

## Setup and conventions

Use a fresh Python process per command, from outside the checkout. The recorded
execution reference is Linux, Python 3.10.12, torch 2.11.0+cu126, NumPy 2.2.6,
tokenizers 0.22.2 and safetensors 0.8.0. Training and the recorded likelihood profile
require a BF16-capable CUDA GPU; export and interpolation use CPU tensors. See
[recorded requirements](recipes/research-v1/runtime-support-v2/frozen_native/requirements-inference-tested.txt).
Dependencies are not installed by the scripts. Metadata/encoding validation uses
installed Python/tokenizers/NumPy; P2 plan utilities also import torch. The bounded
CPU export result below covers a different laptop environment; reader training,
inference/parity and scoring in that environment remain unverified.

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

All stage examples below use `--validate-only`: they check inputs without loading
weights or creating the requested output directory. To run a stage, use its same
arguments with `--execute` instead. Each stage output directory must be nonexistent.
`reader_prepare.py --write` creates prepared files only; it does not train.
No command downloads data/models or calls a teacher/API.

`--policy new-run` is the common reader policy. `--policy historical` delegates P2,
P3, interpolation and export to the unchanged V2 contracts, with their original
flags. See [advanced historical material](recipes/research-v1/POSTTRAINING.md).
Historical A/B governance and the original benchmark launch remain frozen source
interfaces, not part of the new-run dispatcher. No historical approval is minted.

## What you supply

| Input | Exact new-run requirement |
|---|---|
| Tokenizer | Canonical `tokenizer/releases/tokenizer_v1/tokenizer.json`, SHA256 `d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce`; 32k vocabulary, IDs0–6 PAD/UNK/BOS/EOS/system/user/assistant |
| Packed A/B/validation | Separate stage roots with `meta.json` and `train/*.bin` or `val/*.bin`; little-endian uint16, exact sorted shard inventory/hashes/counts and tokenizer identity; see prepared-input schema |
| P2 | Original-sized default: 12,000 train and 500 validation message objects. Unique IDs, disjoint IDs/exact messages across splits, ≤2048 encoded tokens and exact shifted assistant/EOS target counts |
| P3 | 7,168 supplied procedural records (1,792 each COPY/FIELD/MEMBERSHIP/JSON) and 3,072 replay records; disjoint IDs/exact messages, ≤512 tokens. You supply task content; no private template bank or held-out labels needed |
| Evaluation | Explicit local ARC-Easy/PIQA JSONL, task revisions/splits/row counts/file hashes, and scope synthetic/subset/full-local |
| Existing checkpoint | Compatible complete native config and adjacent `.reader.json` declaration. Actual tensors/step are checked when loaded; metadata validation alone does not prove them |

P2 derives the recovered two-pass plan for the supplied population (32 rows/update,
cap1000). Thus 12k gives 750 updates and warmup38; other valid populations produce
a different-length run. The mainline requires **P2step750**, so use 12k for this route.
P3 retains 640 updates and fixed population/batch geometry. These size requirements
are public method settings, not private identities. They do not imply identical data.

Input preparation is separate and explicit:

```bash
python -B "$REPO/recipes/research-v1/reader_prepare.py" messages \
  --input "$DATA/p2-train.raw.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p2-train-prepared" --write
python -B "$REPO/recipes/research-v1/reader_prepare.py" p3-plan \
  --input "$DATA/p3-train.jsonl" --replay "$DATA/p3-replay.jsonl" \
  --tokenizer "$TOK" --out-dir "$OUT/p3-plan" --write
```

Preparation methods, schemas and synthetic format examples are in
[PREPARED_INPUTS.md](recipes/research-v1/PREPARED_INPUTS.md). Existing F/G/I/M source
methods remain discoverable there. There is no new acquisition/cleaning project.

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

A runs steps1–38146; B continues38147–49590 with **full optimizer, RNG and committed
sampler state**. No weights-only stage reset or fabricated bridge checkpoint is used.
The source's strict resume checks remain: same config/optimizer/schedule/runtime,
sampler commitments and exact stage boundary. New-run plan validation substitutes
only the old private provenance/approval layer. Its separate schema binds both
stage inventories and shared validation; the sampler-seed change is allowed only
from A20260832 to B20260833 at38146. Same-stage restart uses `--resume` and a fresh
output directory, with unchanged data/plan/environment. Checkpoints get adjacent
receipts when actually saved, so completed checkpoints survive a later interruption.
The recovered data contract also records file mtimes: preserve the prepared data
files when resuming; relocation that changes this fingerprint is rejected.

Both use 30×576/FFN1536/GQA9:3, context2048, micro8×accum16, model seed20260831,
validation seed20260834, fresh Muon plus the source's AdamW groups, LR0.0006,
weight-decay0.1, clip1, BF16/compile. Absolute WSD horizon49590, warmup500, stable
through44631, decay to49590/floor0.1. The source dataset traverses global contiguous
2049-token windows at stride2048, across shard boundaries. BOS and repeated EOS
masking, final-label coverage, position-weighted loss and no-replacement sampler
are source-derived. Minimum unique blocks: A4,882,688 and B1,464,832; unused tails
are not silently padded into training. Validation must have at least one full block.

Optional in-run validation/generation hooks are disabled in the reader route and
recorded as omitted. Their timing/RNG effects and private governance differ from
history; this does not claim the same training trajectory. Explicit checkpoint
milestones and final checkpoint retention are preserved. Outputs include
`READER_INPUTS.json`, source metrics/run/data metadata, `step_*.pt`, `latest.pt`,
adjacent checkpoint receipts and a completion status only after the trainer returns.

## Posttraining

```bash
python -B "$REPO/recipes/research-v1/reader.py" p2 --policy new-run \
  --parent "$OUT/B/step_049590.pt" --train-jsonl "$DATA/p2-train.jsonl" \
  --validation-jsonl "$DATA/p2-val.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/P2" --validate-only

python -B "$REPO/recipes/research-v1/reader.py" p3 --policy new-run \
  --parent "$OUT/P2/step_000750.pt" --train-jsonl "$DATA/p3-train.jsonl" \
  --replay-jsonl "$DATA/p3-replay.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/P3" --validate-only

python -B "$REPO/recipes/research-v1/reader.py" interpolate --policy new-run \
  --p2-parent "$OUT/P2/step_000750.pt" --p3-parent "$OUT/P3/train/step_000320.pt" \
  --tokenizer "$TOK" --out-dir "$OUT/blend" --validate-only
```

P2 uses the recovered trainer, original effective optimizer/loss/batch settings and
its deterministic two-pass plan. P3 consumes the supplied ordered records/plan,
seed20260907, micro2×accum16, fresh AdamW (betas .9/.95, eps1e-8, weight-decay0),
LR5e-5, warmup32 and cosine horizon640/floor0.1. Each shuffled ten-update block
contains seven procedural and three replay updates, two exposures per row. Loss is
FP32 selected-logit assistant/EOS CE normalized by the **whole effective update's**
target count, with BF16 forward. Both stages check the actual loaded architecture,
state keys/shapes/FP32 values, tied aliases and parent step before training.

P3 saves **320 and 640**, before the corresponding historical hook points. The mainline
selects **step320 of this 640-update schedule**; stopping or rescheduling cosine at
320 changes the method. No adaptive checkpoint selection is offered. New-run P3
omits private baseline/development/val500/final hooks and never writes
`BASELINE_COMPLETE`. The V2 historical-schedule route remains available separately.
P2 retains evaluation on the supplied public-format validation split. Materialized
P2 and P3 do not expose resume; start a fresh run if interrupted. Pretrain's full
resume mechanism must not be confused with posttraining's fresh optimizers.

New-run P3 writes each adjacent `.reader.json` immediately after that checkpoint's
finalized save. Step320 can therefore pass downstream metadata validation even if
later work fails; this does not mark the run complete or enable resume. Receipt
write errors stop the run. The private hook calls (including their arguments) are
omitted before execution. These boundaries are tested with synthetic files and
stub callbacks only; real P3 runtime remains unverified. This fix does not repair
or runtime-validate the separate V2 `training-only` interface.

Plans are derived and checked during validation; execution writes `PLAN.json` (P2)
or `PLAN.jsonl` (P3). Pass `--plan` to require exact equality with a separately
prepared plan. P3 records its new `INPUTS.json` identity in checkpoint metadata;
it is explicitly not an original FROZEN.json or an approval record.

Interpolation defaults to the fixed .75 formula, yielding
`weights/new_alpha075.pt`. `--include-alpha050` additionally records the fixed .50
control; no alpha search is exposed. The complete recovered CPU FP32 `A+alpha*(B-A)`
implementation checks configurations, shapes/dtypes, aliases, fixed/nonfloating and
native buffers, finite values, endpoints, FP64 oracle tolerance, strict save/reload
and parent non-mutation. New tensors are compared to their actual parents and
serialized output, **not** the old alpha075 tensor digest. Outputs are distinctly named.

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
writes212 unique safetensors entries with tied alias metadata, reloads and compares
every source/output dtype/shape/value. It does not instantiate the full model:
nonpersistent RoPE buffers are derived later by identical native code/config.
Frozen inference assets and canonical tokenizer are copied byte-for-byte. Outputs:
`bundle/`, new provenance/evaluation index/manifest, verified tar.gz and checksum.
A new artifact is never labelled the original released alpha075. Source-to-export
tensor equality is different from generation parity, which stays NOT_RUN.

Likelihood uses the pinned harness `_encode_pair` method, raw
`Question: ...\nAnswer:` plus one leading space before each exact candidate,
no chat/BOS/EOS insertion, complete causal continuation alignment, FP32
parameters/forward/logsoftmax/sum, MATH SDPA, unpadded batch1, first maximum, and
`acc_norm` divided by Python Unicode-character length of the original answer
(excluding the delimiter). It accepts local checkpoint files or native bundles.
Outputs: `INPUTS.json`, per-row `ROWS.jsonl` scores/predictions and per-task
`RESULTS.json`. Revisions, original file hashes, normalized row identities and
scope are recorded. No baseline models, datasets or private reviews are downloaded.
`full-local` means all supplied rows, not automatic equivalence to published tasks.

For a compatible pre-existing local checkpoint, create its declaration with
`reader.py bind-checkpoint --policy new-run --checkpoint PATH --kind pretrain
--step 49590 --tokenizer "$TOK"`. The sidecar records the actual file hash and a
compatible config declaration, **not tensor verification or approval**. Actual
loaded state and step must pass execution checks. Automatically produced reader
outputs already include this sidecar.

## Validation status and interpretation

The new interfaces are implemented. Targeted CLI/import/schema/encoding/plan and
small synthetic CPU tensor/serialization tests have run; see the V3 review logs.
The subsequent loader repair passed 13 synthetic regression cases plus the 40
accepted reader/P3 boundary cases. These include both fixed NumPy reconstruction
pickle names, restricted-object rejection and unchanged schema/step checks.
Only NumPy 1.26.4 was actually tested for this repair; simulated serialized names
do not establish a NumPy 2.x environment test or universal version support.

The accepted RunPod P2 validation PASS and P3 original data/640-plan checks are
supplied historical input evidence. They are not this new-run implementation's
training results. The published negative/retention findings remain unchanged.
The new local likelihood interface closes the missing public callable entrypoint;
full historical benchmark/private-answer-review orchestration remains separate.

On 2026-09-12 the first bounded CPU export attempt failed before `torch.load`
because the loader accessed an unloaded NumPy compatibility submodule. After the
explicit-import/scoped-allowlist repair, one separately authorized additional
attempt passed through the public `export --policy new-run --execute` entrypoint.
It used the existing 498,608,319-byte historical alpha075, not a new reader-trained
blend. The prior failure remains a separate result. The staged blend's step0 is
the model-only container convention, not zero historical training updates.

The tested stack was Linux x86_64, Python 3.11.7, torch 2.11.0+cu130, NumPy 1.26.4,
tokenizers 0.22.1 and safetensors 0.6.2, with two CPU threads and no dependency
changes. This differs from the historical execution reference above. The successful
attempt took 36.49 seconds; measured peak RSS was 1,838,408 KiB. Limits were
180 seconds, 8 GiB virtual memory and less than 4 GiB of new working files.

The real source passed complete native config and 213 FP32 state-entry checks.
The exporter stored 212 entries plus tied-alias metadata, reloaded its own output,
and compared source/output dtypes, shapes and values. Tokenizer/frozen asset hashes,
all 17 bundle-manifest entries and all 18 native archive members also matched.
This verifies a temporary re-export of existing alpha075 in this environment;
it does not replace the release or claim historical safetensors file-byte equality.
No full GPT was constructed. Reader training, P3 real runtime, forward/backward,
optimization, inference/generation parity, scoring and GPU work were not run.
CPU conversion is not CPU inference support or end-to-end training validation.
