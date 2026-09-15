# Prepared inputs for a new run

This is the input reference for `reader.py --policy new-run`. It explains how to
prepare your own local text, conversations, and likelihood tasks. Follow
[Reproducibility](../../TRAINING_AND_REPRODUCIBILITY.md) for the full training and
evaluation sequence. These public preparation interfaces were assembled after the
historical runs; the recovered preparation sources are linked at the end.

You supply the data and task content. The commands below do not download datasets,
call a teacher model, or reconstruct the original corpus. All inline records are
synthetic format examples; reproducing the published scores requires more than
matching their schema.

## Setup and output conventions

Use absolute paths, with the dependencies listed in the
[training manual](../../TRAINING_AND_REPRODUCIBILITY.md#setup-and-conventions):

```bash
export REPO=/path/to/your/petitgpt-checkout
export DATA=/path/to/local-inputs
export OUT=/path/to/prepared-outputs
export TOK="$REPO/tokenizer/releases/tokenizer_v1/tokenizer.json"
```

The canonical tokenizer's **file hash** is checked, as well as its 32,000-token
vocabulary and control IDs. Another tokenizer with the same vocabulary size is not
interchangeable. Use UTF-8 JSONL with one object per line and no blank lines.

Every `reader_prepare.py` command requires either `--validate-only` or `--write`.
Both run on CPU, without loading model weights or training:

| Mode | Result |
|---|---|
| `--validate-only` | Run that preparation method's checks and print a JSON summary; create no output directory |
| `--write` | Run the same checks and create the prepared files |

`--out-dir` must be **nonexistent in both modes**; its parent may already exist.
If you see `Requires a fresh nonexistent output directory`, choose a new directory.
To validate an existing prepared file, pass it as an input to the appropriate plan
or training check, with a fresh output path.

| Preparation method | Input | Files created by `--write` |
|---|---|---|
| `messages` | Raw conversation JSONL | `messages.jsonl`, `PREPARATION.json` |
| `p2-plan` | Annotated P2 training JSONL | `PLAN.json` |
| `p3-plan` | Annotated P3 procedural and replay JSONL | `PLAN.jsonl` |
| `packed` | Selected document JSONL | `meta.json`, `<split>/shard_00000.bin`, `PREPARATION.json` |

`PREPARATION.json` records the input hash, record count, and preparation method.
The checks are specific to each method: passing `messages` or `packed` preparation
does not establish that the population meets a later training stage's requirements.

## Messages and plans

### Conversation schema

Each raw row needs a nonempty string `audit_id`, unique within its file, and a
nonempty `messages` array. Use `role` values `system`, `user`, or `assistant` and
nonempty string `content`. The conversation may start with one system turn, then
must alternate user/assistant turns and **end in assistant**. This is training
data; inference prompts instead end in user.

P3 procedural rows also need a `family` value: `COPY`, `FIELD`, `MEMBERSHIP`, or
`JSON`. Replay rows do not require it. The checks validate the family labels and
population sizes, not whether the examples correctly implement their tasks.

```json
{"audit_id":"synthetic-copy-1","family":"COPY","messages":[{"role":"user","content":"Copy: pear"},{"role":"assistant","content":"pear"}]}
```

Training and plan preparation require an additional integer field,
`shifted_supervised_tokens`. It counts assistant-content tokens plus each assistant
turn's EOS among the next-token prediction targets. System/user content, BOS, and
role tokens are masked. All assistant turns contribute, not just the last one.
The canonical formatter preserves content and encodes literal special-token strings
as ordinary text.

### Annotate before training

Save raw rows to your own input file and run:

```bash
python -B "$REPO/recipes/research-v1/reader_prepare.py" messages \
  --input "$DATA/p2-train.raw.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p2-train-prepared" --write
```

Pass **`$OUT/p2-train-prepared/messages.jsonl`** to subsequent commands. The writer
computes `shifted_supervised_tokens`, replacing any existing value, and preserves
other row metadata such as `family`, `source`, or `p2_task`. Do not hand-estimate
the count. Files already satisfying the annotated schema can be used directly.

Repeat with distinct input/output paths for P2 validation, P3 procedural training,
and P3 replay. The [four-file example](../../TRAINING_AND_REPRODUCIBILITY.md#prepare-messages-and-the-p3-plan)
shows these paths together.

| Input | Encoded length limit, including control tokens | Population |
|---|---:|---|
| P2 training | 2,048 | Use 12,000 rows for the complete P2 → P3 → blend workflow |
| P2 validation | 2,048 | Any nonempty set; the historical split had 500 rows |
| P3 procedural training | 512 | 7,168 rows: exactly 1,792 per family |
| P3 replay | 512 | Exactly 3,072 rows |

The general `messages` writer checks **2,048 tokens**, even for rows carrying a P3
family. P3's stricter 512-token limit and population sizes are checked by `p3-plan`
and `reader.py p3`. These limits reject overlength conversations rather than
truncating them.

Plan/training validation recomputes the target count and rejects a missing or wrong
value. P2 training/validation must be disjoint by ID and exact complete message
array; P3 procedural/replay must likewise be disjoint. P2 checks this when validating
the training stage; P3 also checks it during plan preparation. These are exact-match
checks across each paired input, not semantic deduplication or a contamination audit.

### Optional materialized plans

Training derives its plan automatically. Preparing a plan separately lets you inspect
the ordered batches and require exact equality with `--plan`:

```bash
python -B "$REPO/recipes/research-v1/reader_prepare.py" p2-plan \
  --input "$OUT/p2-train-prepared/messages.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p2-plan" --write

python -B "$REPO/recipes/research-v1/reader_prepare.py" p3-plan \
  --input "$OUT/p3-train-prepared/messages.jsonl" \
  --replay "$OUT/p3-replay-prepared/messages.jsonl" --tokenizer "$TOK" \
  --out-dir "$OUT/p3-plan" --write
```

- **P2:** `PLAN.json` uses schema `p2_two_pass_materialized_v1` and seed 20260906.
  It concatenates two shuffled passes, groups them into 32-row updates, and keeps
  at most 1,000 updates. The incomplete final batch and any rows beyond the cap
  are unused, so arbitrary input sizes do not guarantee two exposures for every row.
  With 12,000 rows, the plan has 750 updates and 38 warmup updates. A standalone P2
  run needs at least one full update; downstream P3 and interpolation require
  **P2 step 750**.
- **P3:** `PLAN.jsonl` has 640 updates, each with `update`, `epoch`, `block`,
  `stream`, and 32 ordered `row_ids`. Seed 20260907 determines the shuffles.
  Every 10-update block contains seven procedural updates and three replay updates;
  each procedural update has eight rows from each family. Every supplied row is
  used twice over two epochs.

Pass `--plan "$OUT/p2-plan/PLAN.json"` to P2 or
`--plan "$OUT/p3-plan/PLAN.jsonl"` to P3. Supplied plans must exactly equal the
deterministic derivation. P2's plan also records the training file's absolute path,
hash, and target counts; regenerate it after moving or editing that input.

## Packed pretraining

### Directory layout and inventory

Each release root describes **one split** in its `meta.json`. Keep A, B, and reference
validation in separate roots:

```text
A/meta.json                 B/meta.json                 reference/meta.json
A/train/shard_00000.bin      B/train/shard_00000.bin      reference/val/shard_00000.bin
```

Pass the split directories to training: `--stage-a-dir "$DATA/A/train"`,
`--stage-b-dir "$DATA/B/train"`, and `--validation-dir "$DATA/reference/val"`.
The loader reads `meta.json` from each split directory's parent.

Illustrative `meta.json` for a two-block shard:

```json
{"schema":"petitgpt-packed-new-v3","dtype":"uint16","tokenizer_sha256":"d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce","split":"train","shards":[{"name":"shard_00000.bin","tokens":4097,"sha256":"REPLACE_WITH_ACTUAL_FILE_HASH"}]}
```

The hash placeholder is intentionally invalid. Prefer the writer below for small
inputs; it computes the inventory. Existing large packed corpora can supply this
schema directly without rewriting their shards.

Shards contain little-endian `uint16` IDs in 0–31,999. The inventory must list unique,
sorted, simple `.bin` filenames matching exactly the directory's `.bin` files, with
positive token counts, byte sizes of twice those counts, and SHA-256 file hashes.
Training validation checks those properties, the tokenizer identity, and all token
ranges.

The loader concatenates shards without resetting at boundaries. Each window contains
**2,049 tokens**: 2,048 input tokens and their right-shifted targets. Windows advance
by 2,048 tokens, so a stream of `T` tokens provides `(T - 1) // 2048` full blocks;
the final incomplete tail is unused.

| Split | Minimum full blocks for the new-run training contract |
|---|---:|
| Stage A | 4,882,688 |
| Stage B | 1,464,832 |
| Reference validation | 1 |

The A/B budgets follow the fixed schedule of 128 sequences per update. Validation
requires a full block even though this workflow omits the historical validation
hooks. Both A and B inventories are checked when starting Stage A and bound into
the resume plan. Changing either inventory or the validation inventory invalidates
that binding.

Training and validation may not share byte-identical shards. This does not detect
overlapping text inside different shards. A and B content overlap is not rejected;
you control that data choice. No-replacement sampling describes block traversal,
not corpus deduplication.

### Prepare already selected text

Each raw document needs a nonempty string `id`, unique within the input, and nonempty
string `text`:

```json
{"id":"synthetic-document-1","text":"A small format example."}
```

```bash
python -B "$REPO/recipes/research-v1/reader_prepare.py" packed \
  --input "$DATA/selected-text.jsonl" --tokenizer "$TOK" --split train \
  --out-dir "$OUT/packed-example" --write
```

Here `--out-dir` is the release root; the generated split directory is
`$OUT/packed-example/train`. Use a separate output root and `--split val` for
validation. Each document becomes `[BOS] + canonical content IDs + [EOS]`, with no
extra text separator.

`packed --validate-only` checks the supplied document fields and IDs. It does not
write a shard or establish that the encoded corpus meets the training block budget.
The one-document example above is too small for pretraining. The writer loads the
input JSONL into memory and produces one shard; it is a small preparation utility,
not a streaming corpus-selection or sharding pipeline.

## Local likelihood tasks

The new-run `evaluate` command currently supports **`arc_easy` and `piqa` only**.
ARC-Challenge, HellaSwag, and IFEval results in the technical report came from
separate evaluation runs; they are not additional task names for this interface.

`tasks.json` contains schema `petitgpt-tasks-new-v3`, a `scope` of `synthetic`,
`subset`, or `full-local`, and a nonempty `tasks` list. Each task entry needs:

| Field | Meaning |
|---|---|
| `task` | `arc_easy` or `piqa`, with each task named at most once |
| `revision`, `split` | Explicit nonempty dataset-version and split labels |
| `path` | Local JSONL path; relative paths resolve against the manifest's directory |
| `rows` | Positive integer matching the loaded record count |
| `sha256` | SHA-256 of the exact JSONL file bytes |

PIQA row:

```json
{"goal":"Keep a page dry.","sol1":"Place it under a cover.","sol2":"Place it in water.","label":0}
```

ARC-Easy row:

```json
{"id":"synthetic-arc-1","question":"Which is a fruit?","choices":{"label":["A","B"],"text":["Pear","Rock"]},"answerKey":"A"}
```

PIQA's `label` is integer 0 or 1. ARC's choice labels must be unique and parallel
the choice texts; `answerKey` must identify one of them. Questions and answers must
be nonempty strings. IDs must be unique within each task; when absent, the loader
derives them from task name and zero-based row index. Each encoded question/candidate
pair must fit within 2,048 tokens.

For example, after saving the PIQA row to `$DATA/piqa.synthetic.jsonl`, create a
manifest with its real hash (the destination must not already exist):

```bash
python - "$DATA/piqa.synthetic.jsonl" "$DATA/tasks.synthetic.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data, manifest = map(Path, sys.argv[1:])
payload = data.read_bytes()
rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
task = {
    "task": "piqa", "revision": "synthetic-v1", "split": "illustration",
    "path": str(data.resolve()), "rows": len(rows),
    "sha256": hashlib.sha256(payload).hexdigest(),
}
with manifest.open("x", encoding="utf-8") as stream:
    json.dump({"schema": "petitgpt-tasks-new-v3", "scope": "synthetic",
               "tasks": [task]}, stream, indent=2)
    stream.write("\n")
PY
```

Pass that manifest as `--tasks "$DATA/tasks.synthetic.json"` to the
[evaluation command](../../TRAINING_AND_REPRODUCIBILITY.md#export-and-evaluate),
along with the new-run checkpoint or export it requires. Preparation of task files
alone does not create a model to evaluate.

For dataset experiments, preserve the official text and physical row order. The
evaluator adds the fixed question/answer prompt and leading answer delimiter; do not
add chat formatting or control tokens yourself. Gold labels are required for
accuracy. Scope labels describe your supplied population, not verified reproduction
of published results. The [recorded benchmark protocol](../../configs/research-v1/benchmark_protocol.json)
gives the original ARC-Easy/PIQA dataset revisions and task definitions.

## Preparation coverage

| Method | Implementation and scope |
|---|---|
| Supplied conversations, annotation, packed text, deterministic P2/P3 plans | [reader_prepare.py](reader_prepare.py); local format preparation with explicit paths |
| Tokenizer corpus selection and tokenizer training (historical stages F/G) | [Recovered tokenizer sources](sources/tokenizer/); the canonical released tokenizer is already included |
| Pretraining corpus selection and order (historical stage I) | [Recovered selection sources](sources/data_selection/); original corpora and exclusion artifacts are not republished |
| Packing and reference validation (historical stage M) | [Recovered packing sources](sources/packing/) and [pretrain modules](../../pretrain/); new-run consumes the bin/meta schema above |
| Original P2 selection and P3 procedural population construction | [Recovered post-training preparation](sources/posttraining/runs/); the private generator bank and original populations are not supplied |
| Local likelihood input encoding | [reader_evaluate.py](reader_evaluate.py); raw ARC-Easy/PIQA JSONL to prompt/continuation encodings |

New-run preparation needs your own inputs, not private historical approvals or
held-out templates. Preparation and small synthetic checks do not establish a full
training run or its scores. See [validation status](../../TRAINING_AND_REPRODUCIBILITY.md#validation-status-and-interpretation)
for the current execution evidence and remaining limits.
