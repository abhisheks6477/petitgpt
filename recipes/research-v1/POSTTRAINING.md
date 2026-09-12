# Advanced historical posttraining entry points V2

For ordinary supplied-data training and newly trained checkpoint linkage, start with
[the new-run manual](../../TRAINING_AND_REPRODUCIBILITY.md). The V2 entry points below
remain original-artifact replay interfaces; `reader.py STAGE --policy historical`
delegates to them without relaxing their checks.


Start here for P2 → P3 → fixed interpolation → native export. These are small
public adapters over the recovered source closures, with explicit local bindings.
The frozen snapshots, tokenizer, inference implementation and published report are
unchanged. No execution command below was run in this maintenance task.

| Stage | Supported interface / implementation | Checks actually performed | Full historical replay / remaining limits |
|---|---|---|---|
| P2: [p2.py](p2.py) | Original reviewed Base, train/validation JSONL and consumption plan; fresh output; unchanged recovered 750-update trainer | Supplied RunPod real 12k/500 validation PASS; local help/error/source/plan tests | No training here; original private data and checkpoint remain external |
| P3: [p3.py](p3.py) | Explicit parent/tokenizer/train/replay/update-plan/original-freeze/output; training-only and historical-schedule implementations | Local CLI/error/import/AST checks, canonical tokenizer with synthetic messages, full synthetic ID-only plan (640 updates); supplied support's real-input checks are external evidence | Neither execution mode run here; private original inputs required. Historical-schedule includes baseline and 320/640 hooks, excludes final/Part A/dev100/likelihood. Full historical replay not implemented |
| Fixed interpolation: [interpolate.py](interpolate.py) | Exact original P2step750/P3step320 files, tokenizer, fresh output; original .50/.75 arithmetic and checks | Local help/error/hash/source/AST checks; original parents unavailable locally | No parent tensor loaded/blended here; new P3 checkpoints are not automatically accepted as the original step320 |
| Native export: [export_native.py](export_native.py) | Original alpha075 or fixed-interpolation receipt, frozen inference root, tokenizer, fresh output; complete conversion and packaging implementation | Local help/error/source/asset checks and synthetic opaque-file receipt validation, including tampered-hash rejection | No conversion, tensor equality or parity run here; weights remain external. Receipt-declared tensor identity is independently checked only on execute |

P2 real validation is specifically the unmodified adapter SHA256
`bbbb02e276cce9f0797ab68cab032181f939151cd31c728ff944e272007fe3db` on RunPod.
The [supplied result](provenance/runtime-support-v2/P2_VALIDATION.json) records
Python 3.10.12, torch 2.11.0+cu126, NumPy 2.2.6, tokenizers 0.22.2 and safetensors
0.8.0. It checked Base's opaque hash, all input/source identities, 12,000/500 rows,
disjoint IDs, assistant target encoding and the entire two-pass plan; no model
work or output-directory creation. The initial failed observer comparison happened
before invoking P2; only the observer's absolute/projected path comparison changed.
There was no failed adapter validation rewritten into a success and no adapter fix.
This evidence was supplied, not independently rerun on the laptop.

Local model-free tests use the existing Python 3.11.7 environment: tokenizers
0.22.1, NumPy 1.26.4, torch 2.11.0, safetensors 0.6.2, pytest 7.4.0. This is not
a validated execution environment. No dependencies were installed or upgraded.
Use the recorded RunPod versions above as the execution dependency reference;
[requirements](runtime-support-v2/frozen_native/requirements-inference-tested.txt)
records them. P2/P3 execution requires a BF16-capable CUDA GPU; interpolation/export
perform CPU model work. Validation for interpolation/export needs only Python;
P3 encoding validation needs tokenizers; P2 imports the recovered torch/NumPy trainer
for parser/preflight functions but does not construct a model or deserialize state.

## Commands and local bindings

Run from a working directory outside the checkout, with bytecode disabled. Set the
variables to actual owner-supplied files. Each OUT variable must name a nonexistent
directory, including for validation; validation does not create it. Relative input
paths resolve against the caller's working directory. No input is downloaded.

```bash
export REPO=/path/to/your/petitgpt-checkout
export PYTHONDONTWRITEBYTECODE=1
cd /tmp
python -B "$REPO/recipes/research-v1/p2.py" --help
python -B "$REPO/recipes/research-v1/p3.py" --help
python -B "$REPO/recipes/research-v1/interpolate.py" --help
python -B "$REPO/recipes/research-v1/export_native.py" --help
TOK="$REPO/tokenizer/releases/tokenizer_v1/tokenizer.json"

python -B "$REPO/recipes/research-v1/p2.py" \
  --train-jsonl "$P2_TRAIN" --validation-jsonl "$P2_VAL" \
  --base-checkpoint "$BASE_049590" --original-consumption-plan "$P2_PLAN" \
  --out-dir "$P2_OUT" --validate-only

python -B "$REPO/recipes/research-v1/p3.py" --mode training-only \
  --parent-checkpoint "$P2_STEP750" --tokenizer "$TOK" \
  --train-jsonl "$P3_TRAIN" --replay-jsonl "$P3_REPLAY" \
  --update-plan "$P3_PLAN" --freeze-json "$P3_ORIGINAL_FREEZE" \
  --out-dir "$P3_OUT" --validate-only

python -B "$REPO/recipes/research-v1/p3.py" --mode historical-schedule \
  --parent-checkpoint "$P2_STEP750" --tokenizer "$TOK" \
  --train-jsonl "$P3_TRAIN" --replay-jsonl "$P3_REPLAY" \
  --update-plan "$P3_PLAN" --freeze-json "$P3_ORIGINAL_FREEZE" \
  --historical-bindings "$P3_LOCAL_MAP" --out-dir "$P3_HISTORICAL_OUT" --validate-only

python -B "$REPO/recipes/research-v1/interpolate.py" \
  --p2-step750 "$P2_STEP750" --p3-step320 "$ORIGINAL_P3_STEP320" \
  --tokenizer "$TOK" --out-dir "$BLEND_OUT" --validate-only

python -B "$REPO/recipes/research-v1/export_native.py" \
  --source-checkpoint "$ALPHA075" \
  --frozen-inference-root "$REPO/recipes/research-v1/runtime-support-v2/frozen_native" \
  --tokenizer "$TOK" --out-dir "$EXPORT_OUT" --validate-only
```

**Future execution only, requiring separate owner-authorized bounded validation:**
replace the final `--validate-only` token in any complete command above with
`--execute`. These are full implementation branches, not smoke tests: P2 runs 750
updates; P3 runs 640 (not a newly scheduled 320-update run); interpolation constructs
models and writes two checkpoints; export deserializes/converts/reloads and packages
weights. Do not use Python `-O`, which disables historical assertions. No automatic
restart, overwrite, alpha search or checkpoint selection is exposed.

For a future newly materialized interpolation output, export's complete invocation is:

```bash
python -B "$REPO/recipes/research-v1/export_native.py" \
  --source-checkpoint "$BLEND_OUT/weights/alpha075.pt" \
  --interpolation-receipt "$BLEND_OUT/INTERPOLATION_RECEIPT.json" \
  --frozen-inference-root "$REPO/recipes/research-v1/runtime-support-v2/frozen_native" \
  --tokenizer "$TOK" --out-dir "$EXPORT_OUT" --validate-only
# Future authorized conversion: use the same arguments with --execute instead.
```

The original-alpha075 mode always requires its original file hash. Receipt mode
requires the two fixed tags/alphas, original parent identities, canonical tokenizer,
recorded tensor digests and the supplied file's new SHA256. A receipt is a local
declaration, not signed approval or tensor evidence. Only execute recomputes the
actual alpha075 state digest and enforces strict conversion equality. Neither mode
loads interpolation parents during export. A new P3 training checkpoint cannot be
silently substituted for the frozen downstream parent: its identity/equivalence
needs separately authorized validation.

## File contracts and outputs

P2 keeps its fixed identities in
[p2_effective_launch.json](../../configs/research-v1/p2_effective_launch.json).
Train/validation are original UTF-8 JSONL, 12,000/500 objects with `audit_id`,
`messages` (role/content objects) and `shifted_supervised_tokens`; all original
metadata survives unchanged. Base and plan must match the recorded SHA256 values.
The plan schema is `p2_two_pass_materialized_v1`, seed20260906, 750 updates,
2×16 microbatch/accumulation, two exposures per row, 1,994,854 supervised targets.
Only `train_path` is relocated in a separately recorded plan. Outputs include
checkpoints, trainer preflight/metrics/status, `CONSUMPTION_PLAN.public.json` and
`PUBLIC_PATH_BINDINGS.json`. P2's tokenizer is bound to the repository canonical
path; it is not a configurable alternative tokenizer.

P3 requires original `FROZEN.json` (SHA256 `d9e63373a2ea1cc946fba1a5384916a736b9c31133b60cced6764a0d5cd3881d`),
not its public projection. The [freeze projection](runtime-support-v2/contracts/P3_FREEZE.projection.json)
contains required data hashes; [config](runtime-support-v2/contracts/P3_RUN_CONFIG.adapter.json)
records every training setting. Train has 7,168 rows (1,792 each COPY/FIELD/MEMBERSHIP/JSON),
replay 3,072. Both require unique/disjoint string `audit_id`, original `messages`,
`formatted_tokens`, `shifted_targets`, `input_unsupervised_tokens` and the remaining
required fields/types in the [observed schema](provenance/runtime-support-v2/SUPPLIED_INPUT_SCHEMAS_AND_CHECKS.json).
Optional metadata is checked when present. Input bytes and record order are immutable.
Encoding preserves messages, no default system, sequence≤512; shifted assistant
content plus EOS are supervised, role/input/padding positions are masked.

The original plan is 640 JSONL objects with exactly `update:int`, `epoch:int`,
`block:int`, `stream:string`, `row_ids:list[string]` (32 ordered IDs). Seven procedural
and three replay updates are **shuffled within each block**, as are each family's
input-ordered stream and the update's row IDs, using one Random(20260907) across two
passes. The complete supplied plan must equal the recovered builder's result;
20,480 exposures, exactly twice per row. Validation derives IDs only, never data.
Training consumes the supplied plan, not the derived copy.

P3 execution binds exact frozen `load_model`, `encoded_rows`, `checkpoint`, `train`
function bodies via an explicit AST function allowlist; it never executes the
startup of prepare.py/curriculum.py. AdamW, seed/reset order, 640-horizon cosine
schedule, 32 warmup, LR5e-5, BF16 selected-logit FP32 assistant/EOS loss normalized
by whole-update targets, gradient clip1 and checkpoint320/640 remain source-derived.
The training-only transform removes exactly the initial baseline assertion and
binds explicit omitted evaluation callbacks. The missing hooks change timing and
possibly RNG state; relocated checkpoint metadata also changes serialization.
It never writes `BASELINE_COMPLETE.json`.

Historical-schedule requires `--historical-bindings` JSON of this shape (all values
are absolute local paths; populate **every** original freeze key, no wildcards):

```json
{
  "files": {"runtime/checkers.py": "/local/original/checkers.py", "preparation/train.jsonl": "/local/original/train.jsonl"},
  "accepted_source_files": {"src/model.py": "/local/original/src/model.py"},
  "p2_validation": "/local/original/P2_VALIDATION.jsonl"
}
```

The example illustrates structure only and intentionally fails until complete.
Every file/source hash in the original freeze is checked, including original runtime,
config, development512, train128 and final files (final bytes are hashed, not scored).
Explicit train/replay/plan flags must agree with this map. Original config equality
is checked after changing only `source_checkpoint` to its logical placeholder.
Baseline runs in a separate spawned process: P2 development, val500, train128.
Only successful actual baseline work creates the new baseline marker. Training then
runs in another process, checks that marker, saves before development/val500 hooks
at 320 and 640, and counts these hooks toward the 7200-second training wall cap.
Original marker identities remain historical evidence, never a new run's approval.
The supplied exact pure checker function is used with private row payloads; no new
procedural/evaluation examples are substituted. Final/Part A/dev100/likelihood and
original outer launch/review orchestration are not implemented by this entry point.

P3 outputs `PUBLIC_BINDINGS.json`, optional `HISTORICAL_LOCAL_BINDINGS.json`,
`ADAPTER_STATUS.json`, `train/` checkpoints (including optimizer/scheduler/RNG),
exclusive launch marker, status/metrics/identities and `evaluation/` load records.
Historical-schedule additionally writes real baseline/development/val500 results;
training-only has no evaluation-success records. Output bindings/logs may contain
private paths and row IDs; they are local run outputs, not automatically public files.

Interpolation contracts pin full opaque parent hashes/sizes and tokenizer in
[identities](runtime-support-v2/contracts/INTERPOLATION_EXPORT_IDENTITIES.json).
Validation creates nothing and establishes no tensor equality. Execute retains CPU
FP32 `A + alpha*(B-A)` order, complete config/shape/dtype guards, finite checks,
parameter alias grouping, fixed/nonfloating equality/copy, matching nonpersistent
RoPE buffers, endpoint/FP64-oracle checks, strict save/reload and parent non-mutation.
New outputs: `PUBLIC_BINDINGS.json`, `preparation/` launch/per-parameter/materialization
checks, `weights/alpha050.pt`, `weights/alpha075.pt`, `INTERPOLATION_RECEIPT.json`.
Only model/config/derivation are serialized. Candidate tensor digests must equal
recorded digests before emitting a receipt; file hashes may differ with relocated
parent paths. Original artifacts remain separately identified as historical.

Export takes `--frozen-inference-root` pointing to the supplied small frozen-native
subset or an exact complete frozen source root. The four core `src` files absent
from the subset come from the already recovered posttraining source closure; every
asset has a pinned hash. Tokenizer comes only from the explicit canonical binding.
No frozen inference module is imported or changed. The new adapter implements the
actual export_weights checks: no compiled prefixes, complete config, 213 FP32 state
entries/212 stored tensors, tied parameter/storage, strict load/save/reload,
124,635,456 unique parameters, 60 native buffers, full dtype/shape/value comparisons.
It creates `bundle/` with native weights, config, tokenizer, inference sources,
requirements, notices, new provenance/evaluation index/README/manifest;
`TENSOR_EQUALITY.json` outside the bundle; verified tar.gz and SHA256 sidecar.
Copied historical parity is not a new test result: new parity stays `NOT_RUN`.
The nine-line build_bundle.py remains labelled evidence and is never invoked.

## Provenance and scope

[RELOCATION_V2.json](provenance/RELOCATION_V2.json) separately records original,
transport/support and adapter hashes, function selection, path bindings and narrow
semantic differences. It does not overwrite any original manifest or hash.
The supplied public archive was externally SHA256-checked, regular-member/path
inspected, extracted to a fresh sibling directory outside Git and its 19 manifest
entries checked. Only public support contracts/helpers/frozen assets and public
support provenance were integrated; no private retention report was imported.
The large source/schema evidence is linked here for provenance, not a config to
invoke blindly. [Runtime support](provenance/runtime-support-v2/RUNTIME_SUPPORT.md)
and [source map](provenance/runtime-support-v2/SOURCE_HASH_MAP.json) retain original
source boundaries, omissions and environment records.

F/I/M/A/B preprocessing, acquisition and governance portability remain at their
previous documented coverage. Public benchmark runtime/path portability remains a
separate open gap. No new data acquisition, approvals, historical results, parity,
training completion or release hash was invented. Checkpoint/data acquisition is
owner-local; redistribution rights and original reviews cannot be inferred from
public source availability. Full historical replay is distinct from an implemented
interface and from validation with specified real or synthetic inputs.
