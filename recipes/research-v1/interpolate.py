#!/usr/bin/env python3
"""Exactly two fixed P2/P3 blends. Validation hashes opaque files only, never tensors."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from adapter_common import (
    SOURCE,
    SUPPORT,
    TOKENIZER_SHA,
    activate_source,
    canonical,
    cli,
    definitions,
    fresh,
    identity,
    modes,
    read,
    sha,
    verify_sources,
    write,
)

BLEND = SOURCE / "runs/p3_retention_two_point_interpolation_20260907/runtime/blend.py"


def validate(args):
    fresh(args.out_dir)
    verify_sources()
    contract = read(SUPPORT / "contracts/INTERPOLATION_EXPORT_IDENTITIES.json")
    for path, item in zip((args.p2_step750, args.p3_step320), contract["parents"], strict=True):
        identity(path, item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Parent size mismatch: {path}")
    identity(args.tokenizer, TOKENIZER_SHA)
    return {
        "schema": "FIXED_INTERPOLATION_BINDINGS_V2",
        "validation": "opaque file identities and local bindings only",
        "tensor_equality_established": False,
        "model_operations": 0,
        "parents": [
            {"path": str(p.resolve()), "sha256": sha(p)} for p in (args.p2_step750, args.p3_step320)
        ],
        "tokenizer": {"path": str(args.tokenizer.resolve()), "sha256": TOKENIZER_SHA},
        "fixed_candidates": [0.5, 0.75],
        "new_outputs": "NOT_GENERATED",
        "full_historical_replay": False,
    }


def execute(args, evidence):
    activate_source()
    import torch

    from sft.train_sft import load_ckpt, save_checkpoint_atomic
    from src.chat_template import load_chat_tokenizer
    from src.model import GPT, gpt_config_from_checkpoint_dict

    contract = read(SUPPORT / "contracts/INTERPOLATION_EXPORT_IDENTITIES.json")
    # Keep all frozen arithmetic, shape/alias/buffer/finite/save/reload/nonmutation
    # checks. Add complete recorded configuration guards to the scoped loader.

    def checked_load(path):
        ck = load_ckpt(path)
        cfg = ck.get("config") or ck.get("cfg")
        if cfg != contract["recorded_config"]:
            raise ValueError(
                "Complete checkpoint configuration differs from recorded interpolation config"
            )
        return ck

    fresh(args.out_dir)
    args.out_dir.mkdir(parents=True)
    for directory in ("preparation", "weights"):
        (args.out_dir / directory).mkdir()
    write(args.out_dir / "PUBLIC_BINDINGS.json", evidence)
    ns = dict(
        torch=torch,
        GPT=GPT,
        gpt_config_from_checkpoint_dict=gpt_config_from_checkpoint_dict,
        load_chat_tokenizer=load_chat_tokenizer,
        load_ckpt=checked_load,
        save_checkpoint_atomic=save_checkpoint_atomic,
        collections=collections,
        datetime=datetime,
        hashlib=hashlib,
        sys=sys,
        time=time,
        json=json,
        canonical=canonical,
        sha=sha,
        write=write,
        jsonl=lambda p, xs: Path(p).write_text("".join(canonical(x) + "\n" for x in xs)),
        RUN=args.out_dir.resolve(),
        TOK=args.tokenizer.resolve(),
        TOK_SHA=TOKENIZER_SHA,
        PARENTS=[
            (p.resolve(), c["sha256"])
            for p, c in zip((args.p2_step750, args.p3_step320), contract["parents"], strict=True)
        ],
        CANDIDATES={"alpha050": 0.5, "alpha075": 0.75},
        FLAGS={
            "adapter_schema": "FIXED_INTERPOLATION_V2",
            "evaluation": "NOT_RUN",
            "full_historical_replay": False,
        },
    )
    definitions(BLEND, ["interpolate", "state_digest", "main"], ns)
    ns["main"]()
    actual = read(args.out_dir / "preparation/MATERIALIZATION_CHECKS.json")
    for new, old in zip(actual["artifacts"], contract["fixed_candidates"], strict=True):
        if new["model_state_digest"] != old["model_state_digest"]:
            raise ValueError("New blend tensor digest differs from historical candidate")
    if (
        actual["fixed_state_keys"] != contract["fixed_state_keys"]
        or len(actual["nonpersistent_architecture_buffers_equal"])
        != contract["nonpersistent_buffer_count"]
    ):
        raise ValueError("Recorded fixed/nonpersistent buffer contract differs")
    write(
        args.out_dir / "INTERPOLATION_RECEIPT.json",
        {
            "schema": "FIXED_INTERPOLATION_OUTPUT_V2",
            "parents": evidence["parents"],
            "tokenizer_sha256": TOKENIZER_SHA,
            "artifacts": actual["artifacts"],
            "historical_artifacts": contract["fixed_candidates"],
            "evaluation": "NOT_RUN",
            "full_historical_replay": False,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p2-step750", "p3-step320", "tokenizer", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    modes(parser, "Future CPU model/tensor construction, blending and checkpoint serialization")
    args = parser.parse_args(argv)
    evidence = validate(args)
    if args.execute:
        execute(args, evidence)
    else:
        print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    cli(main)
