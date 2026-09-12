#!/usr/bin/env python3
"""Local format preparation only; no acquisition, private template bank or teacher API."""

from __future__ import annotations

import argparse
from pathlib import Path

from adapter_common import TOKENIZER_SHA, canonical, cli, fresh, rows, sha, write
from reader_contracts import source, tokenizer, verify_code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("format", choices=["messages", "packed", "p2-plan", "p3-plan"])
    p.add_argument(
        "--input", type=Path, required=True, help="Local JSONL: audit_id/messages or id/text"
    )
    p.add_argument("--replay", type=Path, help="Required for p3-plan")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write", action="store_true", help="Create prepared files; does not train")
    a = p.parse_args(argv)
    fresh(a.out_dir)
    verify_code()
    tokenizer(a.tokenizer)
    source()
    from src.chat_template import encode_chat, load_chat_tokenizer

    tok = load_chat_tokenizer(str(a.tokenizer))
    if a.format in ("p2-plan", "p3-plan"):
        from reader_contracts import disjoint, encoded

        train = encoded(a.input, tok, limit=2048 if a.format == "p2-plan" else 512)
        if a.format == "p2-plan":
            from sft.p2_plan import derive_plan

            plan = derive_plan(train[0], train_path=a.input)
        else:
            if not a.replay:
                raise ValueError("p3-plan requires --replay")
            replay = encoded(a.replay, tok, limit=512)
            disjoint(train, replay)
            import random

            from adapter_common import SUPPORT, definitions

            ns = {
                "random": random,
                "SEED": 20260907,
                "FAMILIES": ("COPY", "FIELD", "MEMBERSHIP", "JSON"),
            }
            definitions(SUPPORT / "helpers/p3_plan.py", ["build_plan"], ns)
            plan = ns["build_plan"](train[0], replay[0])
            from p3 import validate_plan

            validate_plan(train[0], replay[0], plan)
        if a.write:
            a.out_dir.mkdir(parents=True)
            if a.format == "p2-plan":
                write(a.out_dir / "PLAN.json", plan)
            else:
                (a.out_dir / "PLAN.jsonl").write_text("".join(canonical(u) + "\n" for u in plan))
        print(canonical({"format": a.format, "written": a.write, "training": False}))
        return 0
    data = rows(a.input)
    seen = set()
    result = []
    if not data:
        raise ValueError("Empty input")
    for r in data:
        key = r.get("audit_id" if a.format == "messages" else "id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Unique nonempty string record IDs required")
        seen.add(key)
        if a.format == "messages":
            ids, labels = encode_chat(tok, r["messages"], default_system=None)
            targets = sum(v != -100 for v in labels[1:])
            if not targets or len(ids) > 2048:
                raise ValueError("Invalid assistant targets/context")
            result.append({**r, "shifted_supervised_tokens": targets})
        else:
            if not isinstance(r.get("text"), str) or not r["text"]:
                raise ValueError("Nonempty text required")
    if a.write:
        a.out_dir.mkdir(parents=True)
        if a.format == "messages":
            (a.out_dir / "messages.jsonl").write_text("".join(canonical(r) + "\n" for r in result))
        else:
            import array
            import sys

            d = a.out_dir / a.split
            d.mkdir()
            path = d / "shard_00000.bin"
            count = 0
            with path.open("xb") as f:
                for r in data:
                    ids = [2] + tok.encode(r["text"]).ids + [3]
                    buf = array.array("H", ids)
                    if sys.byteorder != "little":
                        buf.byteswap()
                    f.write(buf.tobytes())
                    count += len(ids)
            write(
                a.out_dir / "meta.json",
                {
                    "schema": "petitgpt-packed-new-v3",
                    "dtype": "uint16",
                    "tokenizer_sha256": TOKENIZER_SHA,
                    "split": a.split,
                    "shards": [{"name": path.name, "tokens": count, "sha256": sha(path)}],
                },
            )
        write(
            a.out_dir / "PREPARATION.json",
            {
                "input_sha256": sha(a.input),
                "records": len(data),
                "method": "supplied records only; no cleaning/dedup/acquisition",
                "original_corpus": False,
            },
        )
    print(
        canonical({"records": len(data), "format": a.format, "written": a.write, "training": False})
    )
    return 0


if __name__ == "__main__":
    cli(main)
