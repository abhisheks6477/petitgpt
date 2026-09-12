#!/usr/bin/env python3
"""Prepared inputs to a new PetitGPT run. Historical adapters keep their own contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from adapter_common import cli, fresh


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "stage",
        choices=["pretrain", "p2", "p3", "interpolate", "export", "evaluate", "bind-checkpoint"],
    )
    parser.add_argument("--policy", choices=["new-run", "historical"], required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["--help"]:
        parser.print_help()
        return 0
    selection, rest = parser.parse_known_args(argv)
    if selection.policy == "historical":
        names = {"p2": "p2", "p3": "p3", "interpolate": "interpolate", "export": "export_native"}
        if selection.stage not in names:
            raise ValueError(
                "Historical pretrain/evaluation use original frozen sources; see POSTTRAINING.md"
            )
        import importlib

        return importlib.import_module(names[selection.stage]).main(rest)
    p = argparse.ArgumentParser(prog=f"reader.py {selection.stage} --policy new-run")
    if selection.stage == "bind-checkpoint":
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--kind", choices=["pretrain", "p2", "p3", "blend"], required=True)
        p.add_argument("--step", type=int, required=True)
        p.add_argument("--tokenizer", type=Path, required=True)
        a = p.parse_args(rest)
        from reader_contracts import receipt, tokenizer, verify_code

        verify_code()
        tokenizer(a.tokenizer)
        if a.step < 0:
            raise ValueError("Negative checkpoint step")
        fresh(Path(str(a.checkpoint) + ".reader.json"))
        receipt(a.checkpoint, a.kind, a.step, checked=False)
        return 0
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--validate-only",
        action="store_true",
        help="Metadata/schema/encoding checks; no model load",
    )
    group.add_argument(
        "--execute", action="store_true", help="Run the declared stage and write new artifacts"
    )
    if selection.stage == "pretrain":
        from reader_pretrain import arguments, run
    elif selection.stage == "evaluate":
        from reader_evaluate import arguments, run
    else:
        from reader_posttrain import arguments, run
    arguments(p, selection.stage)
    a = p.parse_args(rest)
    a.stage = selection.stage
    from reader_contracts import tokenizer, verify_code

    fresh(a.out_dir)
    verify_code()
    tokenizer(a.tokenizer)
    evidence = run(a)
    if not a.execute:
        import json

        print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    cli(main)
