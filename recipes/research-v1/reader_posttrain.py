"""New-run posttraining bindings; historical adapters and frozen snapshots stay unchanged."""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

from adapter_common import (
    HERE,
    SOURCE,
    SUPPORT,
    TOKENIZER_SHA,
    canonical,
    definitions,
    read,
    rows,
    sha,
    write,
)
from reader_contracts import checkpoint, disjoint, encoded, receipt, source, state


def arguments(p, stage):
    if stage in ("p2", "p3"):
        p.add_argument("--parent", type=Path, required=True)
        p.add_argument("--train-jsonl", type=Path, required=True)
        p.add_argument(
            "--plan",
            type=Path,
            help="Optional supplied plan; must exactly equal deterministic derivation",
        )
        p.add_argument(
            "--validation-jsonl" if stage == "p2" else "--replay-jsonl", type=Path, required=True
        )
    elif stage == "interpolate":
        p.add_argument("--p2-parent", type=Path, required=True)
        p.add_argument("--p3-parent", type=Path, required=True)
        p.add_argument("--include-alpha050", action="store_true")
    elif stage == "export":
        p.add_argument("--parent", type=Path, required=True)
        p.add_argument("--frozen-inference-root", type=Path, default=SUPPORT / "frozen_native")


def run(a):
    evidence = {
        "schema": "petitgpt-reader-v3",
        "stage": a.stage,
        "policy": "new-run",
        "original_release": False,
        "execution": "NOT_RUN",
        "tensor_checks": "NOT_RUN",
        "tokenizer_sha256": TOKENIZER_SHA,
        "source_root": str(SOURCE),
    }
    if a.stage in ("p2", "p3"):
        parent = checkpoint(
            a.parent, ["pretrain"] if a.stage == "p2" else ["p2"], 49590 if a.stage == "p2" else 750
        )
        source()
        from src.chat_template import load_chat_tokenizer

        tok = load_chat_tokenizer(str(a.tokenizer))
        train = encoded(a.train_jsonl, tok, limit=2048 if a.stage == "p2" else 512)
        other = encoded(
            a.validation_jsonl if a.stage == "p2" else a.replay_jsonl,
            tok,
            limit=2048 if a.stage == "p2" else 512,
        )
        disjoint(train, other)
        if a.stage == "p2":
            from sft.p2_plan import derive_plan

            plan = derive_plan(train[0], train_path=a.train_jsonl)
            if plan["max_steps"] < 1:
                raise ValueError("P2 needs at least one complete 32-record update")
        else:
            ns = {
                "random": random,
                "SEED": 20260907,
                "FAMILIES": ("COPY", "FIELD", "MEMBERSHIP", "JSON"),
            }
            definitions(SUPPORT / "helpers/p3_plan.py", ["build_plan"], ns)
            plan = ns["build_plan"](train[0], other[0])
            from p3 import validate_plan

            validate_plan(train[0], other[0], plan)
        if a.plan:
            supplied = read(a.plan) if a.stage == "p2" else rows(a.plan)
            if supplied != plan:
                raise ValueError("Supplied plan differs from deterministic order/targets")
        evidence.update(
            parent=parent,
            train_sha256=sha(a.train_jsonl),
            secondary_sha256=sha(a.validation_jsonl if a.stage == "p2" else a.replay_jsonl),
            train_rows=len(train[0]),
            secondary_rows=len(other[0]),
            plan_sha256=hashlib.sha256(canonical(plan).encode()).hexdigest(),
            updates=plan["max_steps"] if a.stage == "p2" else 640,
            selection="P2 final / P3 step320 from 640-update horizon",
            omitted_hooks="P3 private baseline/development/val500/final; no historical trajectory equivalence",
            resume="No P2/P3 resume in materialized recipe",
        )
        if a.execute:
            a.out_dir.mkdir(parents=True)
            write(a.out_dir / "INPUTS.json", evidence)
            if a.stage == "p2":
                write(a.out_dir / "PLAN.json", plan)
                p2_execute(a, plan)
            else:
                (a.out_dir / "PLAN.jsonl").write_text("".join(canonical(u) + "\n" for u in plan))
                p3_execute(a)
            evidence["execution"] = "COMPLETED"
            write(a.out_dir / "READER_STATUS.json", evidence)
    elif a.stage == "interpolate":
        p2 = checkpoint(a.p2_parent, ["p2"], 750)
        p3 = checkpoint(a.p3_parent, ["p3"], 320)
        if p2["config"] != p3["config"]:
            raise ValueError("Parent configurations differ")
        evidence.update(parents=[p2, p3], alphas=[0.5, 0.75] if a.include_alpha050 else [0.75])
        if a.execute:
            interpolate_execute(a, evidence)
    else:
        evidence["parent"] = checkpoint(a.parent, ["blend"], 0)
        from export_native import asset_bindings

        assets = asset_bindings(
            a.frozen_inference_root,
            a.tokenizer,
            read(SUPPORT / "contracts/INTERPOLATION_EXPORT_IDENTITIES.json"),
        )
        evidence["assets"] = {k: sha(v) for k, v in assets.items()}
        if a.execute:
            export_execute(a, evidence, assets)
    return evidence


def p2_execute(a, plan):
    from p2 import make_argv
    from reader_contracts import load

    import sft.train_sft as trainer

    trainer.load_ckpt = load
    replacements = {
        "--train_jsonl": a.train_jsonl.resolve(),
        "--val_jsonl": a.validation_jsonl.resolve(),
        "--out_dir": a.out_dir.resolve(),
        "--tokenizer_path": a.tokenizer.resolve(),
        "--init_from_pretrain": a.parent.resolve(),
        "--p2_consumption_plan": (a.out_dir / "PLAN.json").resolve(),
        "--max_steps": plan["max_steps"],
        "--warmup_steps": plan["warmup_steps"],
    }
    launch = make_argv(
        read(HERE.parents[1] / "configs/research-v1/p2_effective_launch.json"), replacements
    )
    trainer.main(launch)
    for path in a.out_dir.glob("step_*.pt"):
        receipt(path, "p2", int(path.stem.split("_")[1]), checked=True)


def p3_execute(a):
    from reader_contracts import load
    from reader_p3 import bind_new_run
    from sft.p2_loss import assistant_nll_sum_fp32
    import torch

    from sft.train_sft import (
        checked_optimizer_step,
        effective_batch_backward,
        save_checkpoint_atomic,
    )
    from src.canonical_schedule import lr_schedule
    from src.chat_template import encode_chat, load_chat_tokenizer
    from src.model import GPT, gpt_config_from_checkpoint_dict

    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    random.seed(20260907)
    if not torch.cuda.is_available():
        raise RuntimeError("P3 requires BF16 CUDA")
    data = a.out_dir / "logical-inputs"
    mapping = {
        data / "train.jsonl": a.train_jsonl,
        data / "replay.jsonl": a.replay_jsonl,
        data / "UPDATE_PLAN.jsonl": a.out_dir / "PLAN.jsonl",
        data / "FROZEN.json": a.out_dir / "INPUTS.json",
    }
    cfg = dict(read(SUPPORT / "contracts/P3_RUN_CONFIG.adapter.json")["config"])
    cfg.update(
        source_checkpoint=str(a.parent.resolve()),
        source_sha256=sha(a.parent),
        data_identity="new-run INPUTS.json; not original FROZEN",
        private_evaluation_hooks="NOT_RUN",
    )
    for name in ("train", "evaluation"):
        (a.out_dir / name).mkdir()
    ns = dict(
        torch=torch,
        GPT=GPT,
        gpt_config_from_checkpoint_dict=gpt_config_from_checkpoint_dict,
        load_ckpt=load,
        encode_chat=encode_chat,
        lr_schedule=lr_schedule,
        effective_batch_backward=effective_batch_backward,
        checked_optimizer_step=checked_optimizer_step,
        save_checkpoint_atomic=save_checkpoint_atomic,
        assistant_nll_sum_fp32=assistant_nll_sum_fp32,
        collections=collections,
        datetime=datetime,
        json=json,
        os=os,
        random=random,
        signal=signal,
        sys=sys,
        time=time,
        sha=lambda p: sha(mapping.get(p, p)),
        rows=lambda p: rows(mapping.get(p, p)),
        write=write,
        canonical=canonical,
        CKPT=a.parent,
        EXPECTED=sha(a.parent),
        CONFIG=cfg,
        SEED=20260907,
        DATA=data,
        TRAIN=a.out_dir / "train",
        EVAL=a.out_dir / "evaluation",
    )
    bind_new_run(ns)
    ns["train"](load_chat_tokenizer(str(a.tokenizer)), start)


def interpolate_execute(a, evidence):
    source()
    from interpolate import BLEND
    from reader_contracts import load
    import torch

    from sft.train_sft import save_checkpoint_atomic
    from src.chat_template import load_chat_tokenizer
    from src.model import GPT, gpt_config_from_checkpoint_dict

    a.out_dir.mkdir(parents=True)
    for name in ("preparation", "weights"):
        (a.out_dir / name).mkdir()
    ns = dict(
        torch=torch,
        GPT=GPT,
        gpt_config_from_checkpoint_dict=gpt_config_from_checkpoint_dict,
        load_chat_tokenizer=load_chat_tokenizer,
        load_ckpt=load,
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
        RUN=a.out_dir,
        TOK=a.tokenizer,
        TOK_SHA=TOKENIZER_SHA,
        PARENTS=[(p, sha(p)) for p in (a.p2_parent, a.p3_parent)],
        CANDIDATES={**({"new_alpha050": 0.5} if a.include_alpha050 else {}), "new_alpha075": 0.75},
        FLAGS={"policy": "new-run", "original_release": False, "evaluation": "NOT_RUN"},
    )
    from reader_contracts import raw_load

    ns["load_ckpt"] = lambda p: load(p) if Path(p) in (a.p2_parent, a.p3_parent) else raw_load(p)
    definitions(BLEND, ["interpolate", "state_digest", "main"], ns)
    ns["main"]()
    for path in (a.out_dir / "weights").glob("*.pt"):
        receipt(
            path,
            "blend",
            0,
            checked=True,
            formula="P2 + alpha*(P3step320-P2)",
            parents=evidence["parents"],
        )
    write(
        a.out_dir / "READER_STATUS.json",
        {**evidence, "execution": "COMPLETED", "tensor_checks": "PASSED"},
    )


def export_execute(a, evidence, assets):
    source()
    from export_native import package
    from reader_contracts import CONFIG, load
    from reader_tensors import save_native

    ck = load(a.parent)
    sd = ck["model"]
    a.out_dir.mkdir(parents=True)
    bundle = a.out_dir / "bundle"
    (bundle / "src").mkdir(parents=True)
    for name, path in assets.items():
        shutil.copyfile(path, bundle / name)
    restored = save_native(sd, bundle / "model.safetensors", {"tok_emb.weight": "lm_head.weight"})
    state(restored)
    write(
        bundle / "MODEL_PROVENANCE.json",
        {
            **evidence,
            "execution": "COMPLETED",
            "tensor_checks": "source/export equal",
            "new_export_sha256": sha(bundle / "model.safetensors"),
            "config": CONFIG,
            "parity": "NOT_RUN",
            "nonpersistent_buffers": "Derived by unchanged native code/config; no full model instantiation in converter",
        },
    )
    write(bundle / "EVALUATION_INDEX.json", {"evaluation": "NOT_RUN", "original_release": False})
    for name in ("LICENSE", "SOURCE_NOTICE.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(HERE / "provenance/notices" / name, bundle / name)
    (bundle / "README.md").write_text(
        "# New PetitGPT native model\n\nUser-trained artifact; not the published alpha075.\nSee MODEL_PROVENANCE.json. No benchmark scores or generation parity are asserted.\n"
    )
    package(a.out_dir, bundle)
    write(
        a.out_dir / "READER_STATUS.json",
        {**evidence, "execution": "COMPLETED", "tensor_checks": "source/export equal"},
    )
