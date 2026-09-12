"""Prepared uint16 streams bound to recovered A/B trainers, without historical approvals."""

from __future__ import annotations

from pathlib import Path
import sys

from adapter_common import HERE, REPO, TOKENIZER_SHA, read, sha, write
from reader_contracts import checkpoint, module, receipt, source

ENDS = {"stage_a": 38146, "stage_b": 49590}
STARTS = {"stage_a": 0, "stage_b": 38146}
SEEDS = {"stage_a": 20260832, "stage_b": 20260833}


def arguments(p, stage):
    p.add_argument("--stage", dest="pretrain_stage", choices=list(ENDS), required=True)
    p.add_argument("--stage-a-dir", type=Path, required=True)
    p.add_argument("--stage-b-dir", type=Path, required=True)
    p.add_argument("--validation-dir", type=Path, required=True)
    p.add_argument(
        "--resume",
        type=Path,
        help="Full-state checkpoint with adjacent reader receipt; required for B",
    )


def packed(directory, split):
    import numpy as np

    directory = Path(directory).resolve()
    meta = read(directory.parent / "meta.json")
    if (
        meta.get("schema") != "petitgpt-packed-new-v3"
        or meta.get("dtype") != "uint16"
        or meta.get("tokenizer_sha256") != TOKENIZER_SHA
        or meta.get("split") != split
    ):
        raise ValueError("Packed meta schema/dtype/tokenizer/split mismatch")
    records = meta.get("shards")
    if not isinstance(records, list) or not records:
        raise ValueError("Nonempty shard inventory required")
    names = [r["name"] for r in records]
    if names != sorted(set(names)) or any(
        Path(n).name != n or not n.endswith(".bin") for n in names
    ):
        raise ValueError("Shards must have unique sorted simple .bin names")
    if set(names) != {p.name for p in directory.glob("*.bin")}:
        raise ValueError("Packed directory differs from exact shard inventory")
    total = 0
    for record in records:
        path = directory / record["name"]
        if (
            type(record["tokens"]) is not int
            or record["tokens"] <= 0
            or path.stat().st_size != record["tokens"] * 2
        ):
            raise ValueError("Packed token count/byte size mismatch")
        if sha(path) != record["sha256"]:
            raise ValueError("Packed shard hash mismatch")
        mm = np.memmap(path, dtype="<u2", mode="r")
        for start in range(0, len(mm), 1048576):
            if bool((mm[start : start + 1048576] >= 32000).any()):
                raise ValueError("Token ID outside canonical vocabulary")
        total += record["tokens"]
    return {
        "meta_sha256": sha(directory.parent / "meta.json"),
        "tokenizer_sha256": TOKENIZER_SHA,
        "shards": records,
        "tokens": total,
        "blocks": (total - 1) // 2048,
        "split": split,
    }


def plan_inputs(a):
    stages = {s: packed(getattr(a, s + "_dir"), "train") for s in ENDS}
    val = packed(a.validation_dir, "val")
    if val["blocks"] < 1:
        raise ValueError("Validation requires a complete 2048-transition block")
    train_hashes = {r["sha256"] for v in stages.values() for r in v["shards"]}
    if train_hashes & {r["sha256"] for r in val["shards"]}:
        raise ValueError("Train/validation shard overlap")
    for name, item in stages.items():
        needed = (ENDS[name] - STARTS[name]) * 128
        if item["blocks"] < needed:
            raise ValueError(
                f"{name} needs {needed} unique blocks for the fixed no-replacement schedule"
            )
    return {
        "schema": "petitgpt-pretrain-plan-new-v3",
        "stages": stages,
        "validation": val,
        "ends": ENDS,
        "starts": STARTS,
        "sampler_seeds": SEEDS,
        "model_seed": 20260831,
        "batch": 8,
        "accumulation": 16,
        "schedule": {
            "kind": "wsd",
            "horizon": 49590,
            "warmup": 500,
            "decay_start": 44631,
            "decay_end": 49590,
            "peak": 0.0006,
            "floor_ratio": 0.1,
        },
    }


def validate_transition(saved, current, step):
    if saved == current:
        return
    if (
        saved.get("policy") != "new-run"
        or current.get("policy") != "new-run"
        or saved.get("plan") != current.get("plan")
        or saved.get("stage") != "stage_a"
        or current.get("stage") != "stage_b"
        or step != 38146
    ):
        raise RuntimeError("Invalid new-run A/B plan handover")


def run(a):
    plan = plan_inputs(a)
    previous = None
    if a.resume:
        previous = checkpoint(a.resume, ["pretrain"])
        if previous.get("pretrain_plan") != plan:
            raise ValueError("Resume prepared data/schedule changed")
        if not STARTS[a.pretrain_stage] <= previous["step"] < ENDS[a.pretrain_stage]:
            raise ValueError("Resume step outside current stage")
    elif a.pretrain_stage == "stage_b":
        raise ValueError("Stage B requires full-state A step38146 via --resume")
    evidence = {
        "policy": "new-run",
        "stage": a.pretrain_stage,
        "plan": plan,
        "execution": "NOT_RUN",
        "resume": previous,
        "private_governance": "not applicable; new-run data contract",
        "optional_generation_and_validation_hooks": "omitted; trajectory/timing differs from historical execution",
    }
    if a.execute:
        execute(a, plan, evidence)
    return evidence


def bind_trainer(a, plan):
    # Import the actual recovered stage closure, never the repository shared trainer.
    root = HERE / "sources" / ("pretrain_" + a.pretrain_stage)
    source(root)
    trainer = module(root / "pretrain/train_pretrain_with_bench.py", "reader_bound_pretrain")
    sys.argv = [
        str(root / "pretrain/train_pretrain_with_bench.py"),
        "--train_dir",
        str(getattr(a, a.pretrain_stage + "_dir").resolve()),
        "--val_dir",
        str(a.validation_dir.resolve()),
        "--out_dir",
        str(a.out_dir.resolve()),
        "--samples_dir",
        str((a.out_dir / "samples").resolve()),
        "--tokenizer_path",
        str(a.tokenizer.resolve()),
    ]
    parsed = trainer.parse_args()
    record = read(REPO / f"configs/research-v1/pretrain_{a.pretrain_stage}_effective.json")["args"]
    # Use the same effective settings in A/B. Public hooks are disabled explicitly.
    for key in (
        "vocab_size",
        "seq_len",
        "layers",
        "d_model",
        "n_heads",
        "n_kv_heads",
        "d_ff",
        "dropout",
        "precision",
        "micro_bsz",
        "grad_accum",
        "lr",
        "weight_decay",
        "optimizer",
        "muon_lr",
        "muon_momentum",
        "warmup_steps",
        "lr_schedule",
        "schedule_total_steps",
        "decay_start_step",
        "decay_end_step",
        "min_lr_ratio",
        "grad_clip",
        "num_workers",
        "seed",
        "val_seed",
        "save_every",
        "save_steps",
        "compile",
        "eos_weight",
        "eos_weight_warmup_steps",
        "bos_id",
        "eos_id",
    ):
        setattr(parsed, key, record[key])
    parsed.max_steps = ENDS[a.pretrain_stage]
    parsed.data_stage_start_step = STARTS[a.pretrain_stage]
    parsed.run_plan_stage = a.pretrain_stage
    parsed.run_plan_json = ""
    parsed.stage_a_sampler_seed = SEEDS["stage_a"]
    parsed.stage_b_sampler_seed = SEEDS["stage_b"]
    parsed.sampler_seed = SEEDS[a.pretrain_stage]
    parsed.eval_steps = []
    parsed.eval_every = 0
    parsed.bench_eval_every = 0
    parsed.val_samples = 0
    parsed.val_samples_per_source = 0
    parsed.resume_path = str(a.resume.resolve()) if a.resume else ""
    parsed.resume_full = bool(a.resume)
    trainer.validate_training_args(parsed)
    trainer.parse_args = lambda: parsed
    binding = {
        "policy": "new-run",
        "plan": plan,
        "stage": a.pretrain_stage,
        "expected_stage_samples": (parsed.max_steps - parsed.data_stage_start_step) * 128,
    }

    def bind(args, **kw):
        if plan_inputs(a) != plan:
            raise ValueError("Packed inputs changed before execution")
        return dict(binding)

    trainer.load_run_plan_binding = bind
    base_dataset = trainer.PackedBinDataset

    class PreparedDataset(base_dataset):
        def __init__(self, path, *args, **kwargs):
            split = "val" if Path(path).resolve() == a.validation_dir.resolve() else "train"
            actual = packed(path, split)
            kwargs["require_release_manifest"] = False
            super().__init__(path, *args, **kwargs)
            self.reader_identity = actual

    trainer.PackedBinDataset = PreparedDataset

    def check_dataset(b, d):
        if d.reader_identity != plan["stages"][a.pretrain_stage]:
            raise ValueError("Training dataset binding differs")

    def check_val(b, d):
        if d.reader_identity != plan["validation"]:
            raise ValueError("Validation dataset binding differs")

    trainer.validate_run_plan_dataset = check_dataset
    trainer.validate_run_plan_validation_dataset = check_val
    trainer.validate_run_plan_resume_transition = lambda old, new, checkpoint_step, **kw: (
        validate_transition(old, new, checkpoint_step)
    )
    original_resume = trainer.validate_resume_contract

    def resume_contract(ck, current, **kw):
        saved = ck.get("run_contract", {}).get("run_plan", {})
        new = current["run_plan"]
        step = kw["checkpoint_step"]
        validate_transition(saved, new, step)
        if saved.get("stage") == "stage_a" and new["stage"] == "stage_b":
            kw["governed_stage_transition"] = "A_TO_B"
        original_resume(ck, current, **kw)

    trainer.validate_resume_contract = resume_contract
    # Restricted CPU unpickling with only the existing NumPy RNG metadata allowlist;
    # preserve every subsequent optimizer/RNG/sampler restore check in the source.
    import ast

    from adapter_common import definitions

    def restricted(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.load":
                n.func = ast.Name(id="reader_safe_load", ctx=ast.Load())
                n.keywords = [k for k in n.keywords if k.arg != "weights_only"]
        return node

    def safe_load(path, **kwargs):
        import numpy as np
        import torch

        core = getattr(np, "_core", np.core)
        with torch.serialization.safe_globals([
            core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype("uint32")),
        ]):
            ck = torch.load(path, weights_only=True, **kwargs)
        from reader_contracts import config, state

        if ck.get("global_step") != checkpoint(path, ["pretrain"])["step"]:
            raise ValueError("Actual resume step differs from declared step")
        config(ck["config"])
        state(ck["model"])
        return ck

    trainer.__dict__["reader_safe_load"] = safe_load
    definitions(
        root / "pretrain/train_pretrain_with_bench.py",
        ["load_ckpt"],
        trainer.__dict__,
        transform=restricted,
    )
    atomic_save = trainer._atomic_torch_save

    def save_with_receipt(obj, path):
        atomic_save(obj, path)
        if "model" in obj and "global_step" in obj:
            receipt(Path(path), "pretrain", obj["global_step"], checked=True, pretrain_plan=plan)

    trainer._atomic_torch_save = save_with_receipt
    return trainer, parsed


def execute(a, plan, evidence):
    trainer, parsed = bind_trainer(a, plan)
    a.out_dir.mkdir(parents=True)
    write(a.out_dir / "READER_INPUTS.json", evidence)
    trainer.main()
    write(a.out_dir / "READER_STATUS.json", {**evidence, "execution": "COMPLETED"})
