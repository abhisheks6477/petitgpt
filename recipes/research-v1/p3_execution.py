"""Execution-only bindings to frozen P3 functions. Never imported by validation."""

from __future__ import annotations

import ast
import collections
import datetime
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time

from adapter_common import (
    SUPPORT,
    activate_source,
    canonical,
    definitions,
    fresh,
    read,
    rows,
    sha,
    write,
)
from p3 import P3_SOURCE


def training_only_gate(node):
    """Exactly one explicit difference in the recovered trainer: no baseline gate.

    Evaluation globals below are also explicit NOT_RUN callbacks in this mode.
    The entire optimizer/update/checkpoint loop is otherwise the frozen function.
    """
    if node.name == "train":
        expected = "assert (EVAL / 'BASELINE_COMPLETE.json').exists()"
        if not isinstance(node.body[0], ast.Assert) or ast.unparse(node.body[0]) != expected:
            raise ValueError("Unexpected historical baseline gate shape")
        node.body = node.body[1:]
    return node


def phase(args, binding, name):
    started = time.monotonic()
    activate_source()
    from sft.p2_evaluation import evaluate_position_nll
    from sft.p2_loss import assistant_nll_sum_fp32
    import torch
    from torch.utils.data import DataLoader

    from sft.train_sft import (
        checked_optimizer_step,
        collate_fn_builder,
        effective_batch_backward,
        load_ckpt,
        save_checkpoint_atomic,
    )
    from src.canonical_schedule import lr_schedule
    from src.chat_template import encode_chat, encode_prompt, load_chat_tokenizer
    from src.model import GPT, gpt_config_from_checkpoint_dict

    run = args.out_dir.resolve()
    data = run / "input-bindings"  # Logical namespace only; no input copies or writes.
    p2 = run / "p2-input-bindings"
    mapping = {
        data / "train.jsonl": args.train_jsonl,
        data / "replay.jsonl": args.replay_jsonl,
        data / "UPDATE_PLAN.jsonl": args.update_plan,
        data / "FROZEN.json": args.freeze_json,
    }
    if binding:
        for key in ("development.jsonl", "TRAIN_SAMPLE128.jsonl"):
            mapping[data / key] = Path(binding["files"]["preparation/" + key])
        mapping[p2 / "data/P2_VALIDATION.jsonl"] = Path(binding["p2_validation"])
    config = dict(read(SUPPORT / "contracts/P3_RUN_CONFIG.adapter.json")["config"])
    config["source_checkpoint"] = str(args.parent_checkpoint.resolve())
    ns = dict(
        torch=torch,
        DataLoader=DataLoader,
        GPT=GPT,
        gpt_config_from_checkpoint_dict=gpt_config_from_checkpoint_dict,
        encode_prompt=encode_prompt,
        encode_chat=encode_chat,
        lr_schedule=lr_schedule,
        load_ckpt=load_ckpt,
        effective_batch_backward=effective_batch_backward,
        checked_optimizer_step=checked_optimizer_step,
        collate_fn_builder=collate_fn_builder,
        save_checkpoint_atomic=save_checkpoint_atomic,
        assistant_nll_sum_fp32=assistant_nll_sum_fp32,
        evaluate_position_nll=evaluate_position_nll,
        collections=collections,
        datetime=datetime,
        json=json,
        os=os,
        random=random,
        signal=signal,
        sys=sys,
        time=time,
        Path=Path,
        canonical=canonical,
        write=write,
        sha=lambda p: sha(mapping.get(Path(p), p)),
        rows=lambda p: rows(mapping.get(Path(p), p)),
        DATA=data,
        P2=p2,
        EVAL=run / "evaluation",
        TRAIN=run / "train",
        RUN=run,
        CKPT=args.parent_checkpoint.resolve(),
        EXPECTED=config["source_sha256"],
        CONFIG=config,
        SEED=20260907,
    )
    names = ["load_model", "encoded_rows", "checkpoint", "train"]
    if args.mode == "historical-schedule":
        # The original private checker is hash-checked by the full freeze resolver;
        # the supplied pure function subset avoids its unrelated startup imports.
        import re

        ns["re"] = re
        definitions(SUPPORT / "helpers/p3_checkers.py", ["check"], ns)
        names += ["generate", "agg", "evaluate", "val500", "baseline"]
    else:

        def omitted(*unused):
            return {"status": "NOT_RUN", "reason": "explicit training-only mode"}

        ns.update(evaluate=omitted, val500=omitted)
    definitions(
        P3_SOURCE, names, ns, transform=training_only_gate if args.mode == "training-only" else None
    )
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    random.seed(20260907)
    if not torch.cuda.is_available():
        raise RuntimeError("P3 execution requires the separately validated BF16 CUDA environment")
    tok = load_chat_tokenizer(str(args.tokenizer))
    if name == "baseline":
        ns["baseline"](tok)
    else:
        ns["train"](tok, started)


def execute(args, binding, evidence):
    # Baseline and train are separate spawned processes, matching the historical
    # phase boundary. An old marker is never copied into a new run.
    fresh(args.out_dir)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "train").mkdir()
    (args.out_dir / "evaluation").mkdir()
    write(args.out_dir / "PUBLIC_BINDINGS.json", evidence)
    if binding:
        write(args.out_dir / "HISTORICAL_LOCAL_BINDINGS.json", binding)
    context = multiprocessing.get_context("spawn")
    phases = ["baseline", "train"] if binding else ["train"]
    for name in phases:
        child = context.Process(target=phase, args=(args, binding, name))
        child.start()
        child.join()
        if child.exitcode != 0:
            write(
                args.out_dir / "ADAPTER_STATUS.json",
                {
                    "status": "FAILED",
                    "phase": name,
                    "exitcode": child.exitcode,
                    "automatic_restart": False,
                },
            )
            raise RuntimeError(
                f"P3 {name} failed with exit code {child.exitcode}; no automatic restart"
            )
        if name == "baseline":
            marker = read(args.out_dir / "evaluation/BASELINE_COMPLETE.json")
            if marker != {
                "status": "COMPLETE",
                "optimizer_updates": 0,
                "source_sha256": evidence["bindings"]["parent_checkpoint"]["sha256"],
            }:
                raise RuntimeError("New baseline process did not produce the required gate")
    write(
        args.out_dir / "ADAPTER_STATUS.json",
        {
            "status": "COMPLETED",
            "mode": args.mode,
            "full_historical_replay": False,
            "final_evaluations": "NOT_RUN",
        },
    )
