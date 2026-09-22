"""Fast Stage B shard builder: same core guarantees as the repo's
pretrain/build_pretrain_shards.py (weighted per-source token targets,
BOS/EOS-wrapped documents, uint16-packed fixed-size shards, hash-based
validation holdout to prevent train/val leakage) but batch-tokenizes
(tok.encode_batch) instead of one document at a time, and drops the
per-document SHA-256 audit trail / exclusion-manifest machinery the repo's
script carries (irrelevant to our reproduction, and the actual bottleneck:
measured single-doc encode() at ~37K tok/s vs this approach).

Not a general-purpose replacement for the repo's script -- a scoped,
logged deviation for this one Stage B build. See PROJECT_LOG.md.
"""
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

BOS_ID = 2
EOS_ID = 3
_HASH_SPACE = 1 << 64
BATCH_SIZE = 25000


def is_validation_holdout(text: str, *, train_target: int, val_target: int, seed: int) -> bool:
    train_target = max(0, int(train_target))
    val_target = max(0, int(val_target))
    if val_target == 0:
        return False
    if train_target == 0:
        return True
    h = hashlib.blake2b(digest_size=8, person=b"PetitGPT-val-v1")
    h.update(str(int(seed)).encode("ascii"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    bucket = int.from_bytes(h.digest(), "big", signed=False)
    return bucket * (train_target + val_target) < val_target * _HASH_SPACE


def iter_texts(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t:
                yield t


class SourceQueue:
    """Pulls texts from one source, batch-tokenizes BATCH_SIZE at a time,
    and hands out (text, ids) pairs one at a time to the interleaver."""

    def __init__(self, path: Path, weight: float, tok: Tokenizer):
        self.path = path
        self.weight = weight
        self.tok = tok
        self._it = iter_texts(path)
        self._queue: deque[tuple[str, list[int]]] = deque()
        self._exhausted = False

    def _refill(self):
        batch_texts = []
        for t in self._it:
            batch_texts.append(t)
            if len(batch_texts) >= BATCH_SIZE:
                break
        if not batch_texts:
            self._exhausted = True
            return
        encs = self.tok.encode_batch(batch_texts)
        self._queue.extend(
            (t, [BOS_ID] + e.ids + [EOS_ID]) for t, e in zip(batch_texts, encs)
        )

    def next(self):
        if not self._queue:
            self._refill()
        if not self._queue:
            return None
        return self._queue.popleft()


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {name: w * total for name, w in weights.items()}
    floors = {name: int(v) for name, v in raw.items()}
    remainder = total - sum(floors.values())
    fracs = sorted(raw.items(), key=lambda kv: kv[1] - floors[kv[0]], reverse=True)
    for i in range(remainder):
        floors[fracs[i % len(fracs)][0]] += 1
    return floors


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="path:weight")
    ap.add_argument("--target_train_tokens", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--shard_tokens", type=int, default=10_000_000)
    ap.add_argument("--val_shard_tokens", type=int, default=2_000_000)
    ap.add_argument("--val_ratio", type=float, default=0.002)
    ap.add_argument("--min_val_tokens_per_source", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sources = []
    for s in args.source:
        p, w = s.rsplit(":", 1)
        sources.append((Path(p.strip()), float(w.strip())))
    tot_w = sum(w for _, w in sources)
    weights = {str(p): w / tot_w for p, w in sources}

    tok = Tokenizer.from_file(args.tokenizer_path)
    tok.encode_special_tokens = True

    train_target = allocate(args.target_train_tokens, weights)
    default_val_total = int(round(args.target_train_tokens * args.val_ratio))
    val_target = allocate(default_val_total, weights)
    for name in val_target:
        val_target[name] = max(val_target[name], args.min_val_tokens_per_source)

    queues = {str(p): SourceQueue(p, w, tok) for p, w in sources}
    train_remaining = dict(train_target)
    val_remaining = dict(val_target)

    out_dir = Path(args.out_dir)
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    buf_train: list[int] = []
    buf_val: list[int] = []
    shard_idx_train = 0
    shard_idx_val = 0
    total_train = 0
    total_val = 0
    seen_docs = 0
    kept_train_docs = 0
    kept_val_docs = 0

    def flush(buf, out_dir_, idx):
        arr = np.asarray(buf, dtype=np.uint16)
        path = out_dir_ / f"shard_{idx:05d}.bin"
        arr.tofile(path)
        return len(buf)

    names = list(weights.keys())
    last_report = 0
    while True:
        outstanding = {n: max(0, train_remaining[n]) + max(0, val_remaining[n]) for n in names}
        live = [n for n in names if outstanding[n] > 0]
        if not live:
            break
        probs = np.array([outstanding[n] for n in live], dtype=np.float64)
        probs /= probs.sum()
        name = live[rng.choice(len(live), p=probs)]

        item = queues[name].next()
        if item is None:
            print(f"WARNING: source exhausted before quota: {name}", flush=True)
            train_remaining[name] = 0
            val_remaining[name] = 0
            continue

        seen_docs += 1
        text, ids = item
        n_tok = len(ids)

        holdout = is_validation_holdout(
            text, train_target=train_target[name], val_target=val_target[name], seed=args.seed
        )
        if holdout:
            if val_remaining[name] <= 0:
                continue
            buf_val.extend(ids)
            val_remaining[name] -= n_tok
            total_val += n_tok
            kept_val_docs += 1
            while len(buf_val) >= args.val_shard_tokens:
                chunk, buf_val = buf_val[: args.val_shard_tokens], buf_val[args.val_shard_tokens :]
                flush(chunk, out_val, shard_idx_val)
                shard_idx_val += 1
        else:
            if train_remaining[name] <= 0:
                continue
            buf_train.extend(ids)
            train_remaining[name] -= n_tok
            total_train += n_tok
            kept_train_docs += 1
            while len(buf_train) >= args.shard_tokens:
                chunk, buf_train = buf_train[: args.shard_tokens], buf_train[args.shard_tokens :]
                flush(chunk, out_train, shard_idx_train)
                shard_idx_train += 1

        if total_train - last_report >= 50_000_000:
            last_report = total_train
            print(f"progress: train_tokens={total_train}/{args.target_train_tokens} "
                  f"val_tokens={total_val} seen_docs={seen_docs}", flush=True)

    if buf_train:
        flush(buf_train, out_train, shard_idx_train)
        shard_idx_train += 1
    if buf_val:
        flush(buf_val, out_val, shard_idx_val)
        shard_idx_val += 1

    meta = {
        "train_tokens": total_train,
        "val_tokens": total_val,
        "train_shards": shard_idx_train,
        "val_shards": shard_idx_val,
        "seen_docs": seen_docs,
        "kept_train_docs": kept_train_docs,
        "kept_val_docs": kept_val_docs,
        "train_target_per_source": train_target,
        "val_target_per_source": val_target,
        "sources": weights,
        "tokenizer_path": args.tokenizer_path,
        "seed": args.seed,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "dtype": "uint16",
        "shard_tokens": args.shard_tokens,
        "val_shard_tokens": args.val_shard_tokens,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
