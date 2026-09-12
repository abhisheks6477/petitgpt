#!/usr/bin/env python3
"""Public P2 path adapter. No model work unless --execute is explicitly supplied.

The trainer and deterministic plan implementation remain recovered candidate bytes.
Only the verified consumption plan's train_path and CLI input/output paths change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "sources/posttraining"
RECORD = REPO / "configs/research-v1/p2_effective_launch.json"
TOKENIZER = REPO / "tokenizer/releases/tokenizer_v1/tokenizer.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_identity(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Identity mismatch: {path}: expected {expected}, got {actual}")


def relocate_plan(original, train_path):
    """Change only the location field; all rows, shuffles and target counts survive."""
    result = dict(original)
    result["train_path"] = str(Path(train_path).resolve())
    return result


def make_argv(record, replacements):
    argv = record["argv"][4:]  # recorded Python, -u, -m, sft.train_sft
    if record["argv"][1:4] != ["-u", "-m", "sft.train_sft"]:
        raise ValueError("Unexpected recorded launch shape")
    argv = list(argv)
    for flag, value in replacements.items():
        if argv.count(flag) != 1:
            raise ValueError(f"Expected exactly one {flag}")
        argv[argv.index(flag) + 1] = str(value)
    return argv


def activate_source():
    # A regular package at the repository root must never shadow this closure.
    for name in ("src", "sft"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            location = getattr(loaded, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(SOURCE):
                raise RuntimeError(f"{name} already imported from a different source root")
    sys.path.insert(0, str(SOURCE))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--original-consumption-plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Hash, tokenizer, row and plan checks; no checkpoint deserialization or model construction",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Future full P2 training: 750 updates on one BF16 CUDA GPU",
    )
    args = parser.parse_args(argv)
    record = json.loads(RECORD.read_text())
    frozen = record["input_and_code_sha256"]
    # Read each identity from the actual recorded argv, including its flag switches.
    for flag, path in [
        ("--train_jsonl", args.train_jsonl),
        ("--val_jsonl", args.validation_jsonl),
        ("--init_from_pretrain", args.base_checkpoint),
        ("--p2_consumption_plan", args.original_consumption_plan),
        ("--tokenizer_path", TOKENIZER),
    ]:
        original = record["argv"][record["argv"].index(flag) + 1]
        check_identity(path, frozen[original])
    if args.out_dir.exists():
        raise ValueError("P2 requires a fresh output directory; no overwrite or automatic resume")
    integration = json.loads((HERE / "provenance/INTEGRATION.json").read_text())
    for item in integration["files"]:
        if item["repository_path"].startswith("recipes/research-v1/sources/posttraining/"):
            check_identity(REPO / item["repository_path"], item["sha256"])
    activate_source()
    from sft.p2_plan import load_checked_plan

    from sft.train_sft import build_arg_parser, preflight_sft_record, validate_sft_args
    from src.chat_template import load_chat_tokenizer

    tok = load_chat_tokenizer(str(TOKENIZER))
    seen = set()
    for split, path, expected in [
        ("train", args.train_jsonl, 12000),
        ("val", args.validation_jsonl, 500),
    ]:
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                result = preflight_sft_record(
                    split, row, tok=tok, seq_len=2048, default_system=None
                )
                if row["audit_id"] in seen:
                    raise ValueError("Duplicate or overlapping audit_id")
                seen.add(row["audit_id"])
                if row["shifted_supervised_tokens"] != result["supervised_tokens"]:
                    raise ValueError("Recorded supervised target count differs from encoding")
                count += 1
        if count != expected:
            raise ValueError(f"{split}: expected {expected} rows, found {count}")
    original_plan = json.loads(args.original_consumption_plan.read_text())
    plan = relocate_plan(original_plan, args.train_jsonl)
    # Temporary file is a public location adaptation, never a replacement historical contract.
    with tempfile.TemporaryDirectory(prefix="petitgpt-p2-plan-") as tmp:
        plan_path = Path(tmp) / "CONSUMPTION_PLAN.public.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        replacements = {
            "--train_jsonl": args.train_jsonl.resolve(),
            "--val_jsonl": args.validation_jsonl.resolve(),
            "--init_from_pretrain": args.base_checkpoint.resolve(),
            "--out_dir": args.out_dir.resolve(),
            "--p2_consumption_plan": plan_path,
            "--tokenizer_path": TOKENIZER,
        }
        launch = make_argv(record, replacements)
        parsed = build_arg_parser().parse_args(launch)
        validate_sft_args(parsed)
        load_checked_plan(
            plan_path,
            train_path=args.train_jsonl,
            dataset_rows=12000,
            seed=parsed.seed,
            micro_bsz=parsed.micro_bsz,
            grad_accum=parsed.grad_accum,
            max_steps=parsed.max_steps,
        )
        evidence = {
            "validation": "data-path checked; no model work",
            "original_plan_sha256": digest(args.original_consumption_plan),
            "public_plan_sha256": digest(plan_path),
            "semantic_changes": [],
            "location_changes": ["consumption_plan.train_path", *replacements],
            "train_rows": 12000,
            "validation_rows": 500,
            "source_root": str(SOURCE),
        }
        if not args.execute:
            print(json.dumps(evidence, indent=2))
            return 0
        args.out_dir.mkdir(parents=True)
        persistent_plan = args.out_dir.resolve() / plan_path.name
        persistent_plan.write_bytes(plan_path.read_bytes())
        replacements["--p2_consumption_plan"] = persistent_plan
        launch = make_argv(record, replacements)
        evidence["argv"] = launch
        (args.out_dir / "PUBLIC_PATH_BINDINGS.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
        from sft.train_sft import main as train

        train(launch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"P2 validation failed: {exc}") from exc
