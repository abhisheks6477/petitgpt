# Prepared input formats and methods

These are new-run contracts. Data rights, acquisition, cleaning and task content
are supplied separately. Different data cannot reproduce the same measured results
by configuration alone. All examples below are **synthetic format illustrations**.

## Messages and plans

UTF-8 JSONL, one object per line, no blank lines. For P2 train/validation and P3
train/replay: unique nonempty `audit_id`, `messages` with standard system/user/assistant
roles and string content, and `shifted_supervised_tokens`. Preserve the entire message
sequence; role/input tokens are masked, assistant content and EOS supervised.
Literal special-token strings in content use the canonical chat formatter's safe
content encoding. P3 train additionally requires `family` in
COPY/FIELD/MEMBERSHIP/JSON. Metadata such as replay `source`/`p2_task` is retained.

```json
{"audit_id":"synthetic-copy-1","family":"COPY","messages":[{"role":"user","content":"Copy: pear"},{"role":"assistant","content":"pear"}]}
```

This is raw format; do not guess the target count. Run `reader_prepare.py messages
--input RAW --tokenizer TOK --out-dir NEW --write` to produce annotated
`messages.jsonl`. The validator rejects wrong counts, over-context rows and split
overlap by ID or complete exact messages. It does not claim a semantic contamination
audit. P2 fits≤2048 tokens; P3≤512. P3 requires exactly1792 records per family and
3072 replay rows, so a one-row example cannot masquerade as the release-sized input.
No private procedural/development/final templates or approvals are necessary.

`reader_prepare.py p2-plan --input TRAIN --tokenizer TOK --out-dir NEW --write`
emits the recovered `p2_two_pass_materialized_v1` plan: two epoch permutations,
32 rows/update, cap1000, actual target counts and train hash/path. `p3-plan` additionally
requires `--replay REPLAY`, emits640 JSONL updates with `update`, `epoch`, `block`,
`stream`, and32 ordered `row_ids`. It reuses the seed20260907 family/replay shuffle
method and validates two exposures and7:3 blocks. Both support `--validate-only`.
Training optionally takes `--plan` and rejects any differing order/targets.

## Packed pretraining

Each release root has one `meta.json`, with `train/` or `val/` containing exactly the
listed sorted shards. The loader consumes little-endian uint16 token IDs0–31999,
concatenates shards without resetting the transition at boundaries, and extracts
2049-token windows at stride2048. The final incomplete tail is unused.

```json
{"schema":"petitgpt-packed-new-v3","dtype":"uint16","tokenizer_sha256":"d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce","split":"train","shards":[{"name":"shard_00000.bin","tokens":4097,"sha256":"REPLACE_WITH_ACTUAL_FILE_HASH"}]}
```

The hash placeholder is invalid and will be rejected. Inventory count/byte hashes,
all token ID ranges, dtype, tokenizer identity and available full block budget are
checked. Training and validation must not share identical shards. Both A and B
inventories enter one new plan; they cannot be changed during a resume/handover.
A and B may share content only as an explicitly prepared data choice, never an
implicit sampler replay. New-run does not certify historical exclusion/selection.

A small local writer is available for already selected text JSONL:

```json
{"id":"synthetic-document-1","text":"A small format example."}
```

```bash
python -B "$REPO/recipes/research-v1/reader_prepare.py" packed \
  --input "$DATA/selected-text.jsonl" --tokenizer "$TOK" --split train \
  --out-dir "$OUT/packed-example" --write
```

It serializes `[BOS] + canonical content IDs + [EOS]` per supplied document into one
shard and writes the real inventory. It does not select/clean/filter/acquire the
original corpus. Use `--validate-only` to inspect schema without writing. For large
already packed corpora, provide the same inventory format directly; no full rewrite
is necessary. Match the release-sized budgets in the practical manual.

## Local likelihood tasks

`tasks.json` names local JSONL files, explicit revisions/splits/counts and SHA256.
Relative paths resolve against tasks.json, not a hidden original machine path.

```json
{"schema":"petitgpt-tasks-new-v3","scope":"synthetic","tasks":[{"task":"piqa","revision":"synthetic-v1","split":"illustration","path":"piqa.synthetic.jsonl","rows":1,"sha256":"REPLACE_WITH_ACTUAL_FILE_HASH"}]}
```

PIQA row schema:

```json
{"goal":"Keep a page dry.","sol1":"Place it under a cover.","sol2":"Place it in water.","label":0}
```

ARC-Easy row schema:

```json
{"id":"synthetic-arc-1","question":"Which is a fruit?","choices":{"label":["A","B"],"text":["Pear","Rock"]},"answerKey":"A"}
```

Use exact local official strings and physical row order for dataset experiments.
No trimming, answer cleanup or private evaluation labels are fetched. PIQA IDs are
derived from row index when absent. Gold labels are required for accuracy. Manifest
scope must be synthetic, subset or full-local; none alone claims published-result
reproduction. ARC/PIQA recorded dataset revisions and task definitions remain in
[benchmark_protocol.json](../../configs/research-v1/benchmark_protocol.json).

## Preparation coverage

| Method | Python implementation / coverage |
|---|---|
| Local supplied messages, annotation, packed text, deterministic P2/P3 plans | [reader_prepare.py](reader_prepare.py); directly callable with explicit paths |
| F tokenizer corpus selection and G tokenizer training | [recovered tokenizer preparation/training](sources/tokenizer/); original data identities and exclusions remain external. Canonical released tokenizer is already supplied |
| I corpus selection and order | [recovered data selection](sources/data_selection/); exact D2/D3/L1/H exclusion artifacts and licensed corpora not republished |
| M packing and reference validation | [recovered packing](sources/packing/) and [pretrain preparation modules](../../pretrain/); historic manifests are original governance records. New-run uses the explicit bin/meta schema above |
| P2 original selection and P3 procedural population construction | [recovered posttraining preparation](sources/posttraining/runs/); no private generator bank or frozen original population replacement is supplied |
| User-local likelihood input normalization | [reader_evaluate.py](reader_evaluate.py); raw ARC/PIQA JSONL to exact prompt/continuation encodings, no acquisition |

These data-access limitations are distinct from code availability. Current reader
training/conversion/scoring execution branches exist, but full-model execution has
not been validated here. Original governance, source acquisition and private final
review are not prerequisites for an ordinary new run.
