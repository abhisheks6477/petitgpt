#!/usr/bin/env python3
"""P3 frozen supplied-data training interface; no construction of procedural data."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from adapter_common import (
    PROVENANCE,
    SOURCE,
    SUPPORT,
    TOKENIZER_SHA,
    activate_source,
    cli,
    definitions,
    fresh,
    identity,
    modes,
    read,
    rows,
    sha,
    verify_sources,
)

P3_SOURCE = SOURCE / "runs/p3_basic_instruction_generalization_20260907/runtime/execute.py"
DIFFERENCES = {
    "training-only": [
        "Baseline gate and baseline/development/val500 evaluations NOT_RUN.",
        "Evaluation timing and possible RNG effects differ; no historical execution equivalence.",
        "Checkpoint run_config.source_checkpoint is relocated; serialization/file hashes may differ.",
        "Only original freeze identity and training subset checked; private freeze dependencies omitted.",
        "Final/Part A/dev100/likelihood evaluations NOT_RUN.",
    ],
    "historical-schedule": [
        "Separate new baseline process gates training; development and val500 run at 320/640.",
        "All original freeze dependencies are checked through explicit local bindings.",
        "Filesystem paths, process orchestration/timing and serialization may differ.",
        "Final/Part A/dev100/likelihood evaluations NOT_RUN; not full historical replay.",
    ],
}


def validate_records(data, schema, tok, encode_chat):
    if len(data) != schema["rows"]:
        raise ValueError(f"Expected {schema['rows']} rows, got {len(data)}")
    types = {"str": str, "int": int, "list": list, "dict": dict}
    seen = set()
    for row in data:
        for key, spec in schema["observed_top_level_schema"].items():
            if key not in row and spec["present_rows"] < schema["rows"]:
                continue
            if key not in row or type(row[key]) not in tuple(types[t] for t in spec["types"]):
                raise ValueError(f"Row field schema mismatch: {key}")
        if not row["audit_id"] or row["audit_id"] in seen:
            raise ValueError("Empty/duplicate audit_id")
        seen.add(row["audit_id"])
        if not row["messages"] or any(
            not isinstance(m, dict)
            or m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str)
            for m in row["messages"]
        ):
            raise ValueError("messages must contain role/content objects")
        ids, labels = encode_chat(tok, row["messages"], default_system=None)
        expected = [
            t
            for m in row["messages"]
            if m["role"] == "assistant"
            for t in tok.encode(m["content"]).ids + [3]
        ]
        if (
            len(ids) > 512
            or not expected
            or [v for v in labels[1:] if v != -100] != expected
            or any(label != -100 and label != tid for tid, label in zip(ids, labels, strict=True))
        ):
            raise ValueError("Assistant/EOS masking or context mismatch")
        for key, value in [
            ("formatted_tokens", len(ids)),
            ("shifted_targets", len(expected)),
            ("input_unsupervised_tokens", len(ids) - len(expected)),
            ("p3_formatted_tokens", len(ids)),
            ("p3_shifted_targets", len(expected)),
        ]:
            if key in row and row[key] != value:
                raise ValueError(f"Encoding differs from stored {key}")
    return seen


def validate_plan(proc, replay, plan):
    # Pure supplied support subset; no curriculum/data preparation import.
    ns = {"__name__": "p3_public_plan"}
    import hashlib
    import json
    import random

    ns.update(
        hashlib=hashlib,
        json=json,
        random=random,
        SEED=20260907,
        FAMILIES=("COPY", "FIELD", "MEMBERSHIP", "JSON"),
    )
    definitions(SUPPORT / "helpers/p3_plan.py", ["build_plan"], ns)
    if len(proc) != 7168 or len(replay) != 3072:
        raise ValueError("Expected 7168 procedural and 3072 replay rows")
    ids = [r["audit_id"] for r in proc + replay]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate/overlapping train/replay audit_id")
    if collections.Counter(r["family"] for r in proc) != dict.fromkeys(ns["FAMILIES"], 1792):
        raise ValueError("Procedural family population mismatch")
    for u in plan:
        if (
            set(u) != {"update", "epoch", "block", "stream", "row_ids"}
            or any(type(u[k]) is not int for k in ("update", "epoch", "block"))
            or not isinstance(u["row_ids"], list)
            or len(u["row_ids"]) != 32
            or any(type(i) is not str for i in u["row_ids"])
        ):
            raise ValueError("Update plan schema mismatch")
    if plan != ns["build_plan"](proc, replay):
        raise ValueError("Ordered update plan differs from frozen seeded 7:3 plan")
    if collections.Counter(i for u in plan for i in u["row_ids"]) != dict.fromkeys(ids, 2):
        raise ValueError("Each row must have exactly two exposures")


def historical_inputs(args, freeze):
    if args.historical_bindings is None:
        raise ValueError(
            "historical-schedule requires --historical-bindings: all original freeze files, sources and P2 validation; old baseline markers cannot substitute"
        )
    binding = read(args.historical_bindings)
    if set(binding) != {"files", "accepted_source_files", "p2_validation"}:
        raise ValueError(
            "Historical binding schema: files, accepted_source_files, p2_validation only"
        )
    paths = [binding["p2_validation"]]
    for group in ("files", "accepted_source_files"):
        if not isinstance(binding[group], dict):
            raise ValueError(f"Historical {group} must map original logical names to local paths")
        paths.extend(binding[group].values())
    if any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths):
        raise ValueError("Historical bindings require absolute local paths")
    for group in ("files", "accepted_source_files"):
        if set(binding[group]) != set(freeze[group]):
            raise ValueError(f"Historical bindings must cover exactly FROZEN.json {group}")
        for key, expected in freeze[group].items():
            identity(Path(binding[group][key]), expected)
    f = binding["files"]
    for key, path in [
        ("train.jsonl", args.train_jsonl),
        ("replay.jsonl", args.replay_jsonl),
        ("UPDATE_PLAN.jsonl", args.update_plan),
    ]:
        if Path(f["preparation/" + key]).resolve() != path.resolve():
            raise ValueError(f"Conflicting explicit and historical binding: {key}")
    original_config = read(f["preparation/RUN_CONFIG.json"])
    expected_config = read(SUPPORT / "contracts/P3_RUN_CONFIG.adapter.json")["config"]
    relocated = {**original_config, "source_checkpoint": "${P2_CHECKPOINT}"}
    if relocated != expected_config:
        raise ValueError("Historical configuration semantic mismatch")
    val_hash = read(PROVENANCE / "SUPPLIED_INPUT_SCHEMAS_AND_CHECKS.json")["p2_inputs"][
        "val_jsonl"
    ]["sha256"]
    identity(binding["p2_validation"], val_hash)
    return binding


def validate(args):
    fresh(args.out_dir)
    verify_sources()
    contract = read(SUPPORT / "contracts/P3_FREEZE.projection.json")
    identity(args.freeze_json, contract["original_sha256"])
    freeze = read(args.freeze_json)
    for group in ("files", "accepted_source_files"):
        if any(freeze[group].get(k) != v for k, v in contract[group].items()):
            raise ValueError("Original freeze differs from published projection")
    identity(
        args.parent_checkpoint,
        read(SUPPORT / "contracts/P3_RUN_CONFIG.adapter.json")["config"]["source_sha256"],
    )
    identity(args.tokenizer, TOKENIZER_SHA)
    for key, path in [
        ("train.jsonl", args.train_jsonl),
        ("replay.jsonl", args.replay_jsonl),
        ("UPDATE_PLAN.jsonl", args.update_plan),
    ]:
        identity(path, contract["files"]["preparation/" + key])
    binding = historical_inputs(args, freeze) if args.mode == "historical-schedule" else None
    if args.mode == "training-only" and args.historical_bindings:
        raise ValueError("--historical-bindings belongs only to historical-schedule")
    activate_source()
    from src.chat_template import encode_chat, load_chat_tokenizer

    tok = load_chat_tokenizer(str(args.tokenizer))
    schemas = read(PROVENANCE / "SUPPLIED_INPUT_SCHEMAS_AND_CHECKS.json")["data"]
    proc, replay, plan = rows(args.train_jsonl), rows(args.replay_jsonl), rows(args.update_plan)
    validate_records(proc, schemas["train.jsonl"], tok, encode_chat)
    replay_ids = validate_records(replay, schemas["replay.jsonl"], tok, encode_chat)
    validate_plan(proc, replay, plan)
    if binding:
        for key in ("development.jsonl", "TRAIN_SAMPLE128.jsonl"):
            validate_records(
                rows(binding["files"]["preparation/" + key]), schemas[key], tok, encode_chat
            )
        val = rows(binding["p2_validation"])
        if len(val) != 500 or replay_ids & {r["audit_id"] for r in val}:
            raise ValueError("P2 val500 size or replay/validation disjointness mismatch")
    evidence = {
        "schema": "P3_PUBLIC_BINDINGS_V2",
        "mode": args.mode,
        "validation": "frozen identities, encoding and complete ordered plan checked",
        "model_operations": 0,
        "full_historical_replay": False,
        "differences": DIFFERENCES[args.mode],
        "bindings": {
            key: {"path": str(getattr(args, key).resolve()), "sha256": sha(getattr(args, key))}
            for key in (
                "parent_checkpoint",
                "tokenizer",
                "train_jsonl",
                "replay_jsonl",
                "update_plan",
                "freeze_json",
            )
        },
        "updates": 640,
        "exposures": 20480,
        "checkpoint_steps": [320, 640],
        "baseline": "NOT_RUN",
        "evaluations": "NOT_RUN",
    }
    return binding, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(DIFFERENCES))
    for name in (
        "parent-checkpoint",
        "tokenizer",
        "train-jsonl",
        "replay-jsonl",
        "update-plan",
        "freeze-json",
        "out-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--historical-bindings",
        type=Path,
        help="JSON local resolver for original freeze files/sources and p2_validation",
    )
    modes(parser, "Future 640-update BF16 CUDA P3 training with checkpoints at 320/640")
    args = parser.parse_args(argv)
    binding, evidence = validate(args)
    if args.execute:
        from p3_execution import execute

        execute(args, binding, evidence)
    else:
        import json

        print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    cli(main)
