#!/usr/bin/env python3
"""Model-free recipe checks. Never imports trainers or opens checkpoints/datasets.

Optional synthetic input validates messages only; it cannot certify historical data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIGS = REPO / "configs/research-v1"
TOKENIZER = REPO / "tokenizer/releases/tokenizer_v1/tokenizer.json"
TOKENIZER_SHA = "d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser_flags(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        arg.value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        for arg in call.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }


def validate_launch(argv, source):
    supported = parser_flags(source)
    requested = {value.split("=")[0] for value in argv if value.startswith("--")}
    missing = requested - supported
    if missing:
        raise ValueError(f"{source}: unrecognized recorded flags: {sorted(missing)}")
    return len(requested)


def validate_repository():
    integration = json.loads((HERE / "provenance/INTEGRATION.json").read_text())
    for record in integration["files"]:
        p = REPO / record["repository_path"]
        if sha(p) != record["sha256"]:
            raise ValueError(f"Integrated source/config bytes changed: {p}")
    for line in (TOKENIZER.parent / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(TOKENIZER.parent / name.lstrip("*")) != expected:
            raise ValueError(f"Tokenizer manifest mismatch: {name}")
    if sha(TOKENIZER) != TOKENIZER_SHA:
        raise ValueError("Canonical tokenizer SHA256 mismatch")
    payload = json.loads(TOKENIZER.read_text())
    expected = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<|system|>", "<|user|>", "<|assistant|>"]
    specials = {x["content"]: x["id"] for x in payload["added_tokens"] if x["special"]}
    if (
        specials != dict(zip(expected, range(7), strict=True))
        or len(payload["model"]["vocab"]) != 32000
    ):
        raise ValueError("Invalid release vocabulary/special-token contract")
    if payload["normalizer"] is not None or payload["post_processor"] is not None:
        raise ValueError("Release must not normalize or insert special tokens")
    if payload["pre_tokenizer"]["add_prefix_space"]:
        raise ValueError("Release must not add prefix space")
    sources = list((HERE / "sources").rglob("*.py"))
    for p in sources:
        ast.parse(p.read_text(), filename=str(p))
    for p in CONFIGS.rglob("*.json"):
        json.loads(p.read_text())
    checked = {}
    for name, stage in [
        ("pretrain_stage_a_initial_effective", "pretrain_stage_a"),
        ("pretrain_stage_a_effective", "pretrain_stage_a"),
        ("pretrain_stage_b_effective", "pretrain_stage_b"),
    ]:
        record = json.loads((CONFIGS / (name + ".json")).read_text())
        checked[name] = validate_launch(
            shlex.split(record["cmd"]),
            HERE / "sources" / stage / "pretrain/train_pretrain_with_bench.py",
        )
    record = json.loads((CONFIGS / "p2_effective_launch.json").read_text())
    checked["p2"] = validate_launch(
        record["argv"][4:], HERE / "sources/posttraining/sft/train_sft.py"
    )
    record = json.loads((CONFIGS / "tokenizer_environment_argv.json").read_text())
    checked["tokenizer"] = validate_launch(
        record["argv"][1:],
        HERE / "sources/tokenizer/tokenizer/tokenizer_training/train_tokenizer.py",
    )
    return {
        "validation": "source hashes, Python syntax, JSON config and recorded flag correspondence checked",
        "mapped_files": len(integration["files"]),
        "python_sources": len(sources),
        "recorded_launch_flag_counts": checked,
        "tokenizer_sha256": TOKENIZER_SHA,
        "historical_data_paths_checked": False,
        "training_smoke_run": False,
        "full_historical_reproduction": False,
        "model_operations": 0,
    }


def validate_synthetic(path):
    # Use the recovered P2 encoder, whose source-preserving behavior differs from shared src.
    sys.path.insert(0, str(HERE / "sources/posttraining"))
    from src.chat_template import encode_chat, load_chat_tokenizer

    tok = load_chat_tokenizer(str(TOKENIZER))
    results = []
    with path.open() as stream:
        for n, line in enumerate(stream, 1):
            record = json.loads(line)
            if set(record) != {"messages"}:
                raise ValueError(
                    "Illustrative fixture must contain only messages; not historical audit rows"
                )
            ids, labels = encode_chat(tok, record["messages"], default_system=None)
            if len(ids) > 2048:
                raise ValueError(f"Row {n}: overlength; source-preserving mode never truncates")
            count = sum(x != -100 for x in labels[1:])
            if count == 0:
                raise ValueError(f"Row {n}: no assistant targets")
            results.append({"row": n, "tokens": len(ids), "assistant_targets_including_eos": count})
            if n >= 32:
                if stream.read(1):
                    raise ValueError("Synthetic validation is limited to 32 rows")
                break
    if not results:
        raise ValueError("Empty synthetic input")
    if "torch" in sys.modules:
        raise RuntimeError("Model-free synthetic checker unexpectedly imported torch")
    return {"illustrative_only": True, "historical_data_validation": False, "rows": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-jsonl",
        type=Path,
        help="At most 32 synthetic message rows; no benchmark or historical data",
    )
    args = parser.parse_args(argv)
    report = validate_repository()
    if args.synthetic_jsonl:
        report["synthetic_schema"] = validate_synthetic(args.synthetic_jsonl)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc
