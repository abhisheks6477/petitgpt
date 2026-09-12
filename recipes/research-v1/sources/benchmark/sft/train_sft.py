#!/usr/bin/env python3

"""Supervised fine-tuning on chat data (also the engine behind distill/train_distill.py).

Chat encoding, loss masking, and refusal weighting all live in
src/chat_template.py — a single source of truth shared with DPO/GRPO and with
the sampling code below, so training-time and inference-time token sequences
are identical by construction.

Contract: [BOS] <|system|> ... <|user|> ... <|assistant|> {answer} [EOS] ...,
supervised span = every assistant answer + its trailing EOS.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal

# Make imports work no matter where you run from
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chat_template import (  # noqa: E402
    IGNORE_INDEX,
    build_example,
    decode_completion,
    encode_chat,
    encode_prompt,
    extract_last_user_and_ref,
    load_chat_tokenizer,
    prepare_prompt_messages,
    truncate_chat_sequence,
)
from src.model import GPT, GPTConfig, gpt_config_from_checkpoint_dict  # noqa: E402
from src.optim import build_optimizer  # noqa: E402
from src.posttrain_preflight import (  # noqa: E402
    require_preflight_passed,
    run_jsonl_preflight,
)
from src.posttrain_resume import (  # noqa: E402
    DeterministicEpochBatchSampler,
    build_resume_contract_base,
    capture_rng_state,
    make_loader_generator,
    require_resume_step,
    restore_rng_state,
    restore_training_state,
    resume_contract_for_step,
    validate_resume_contract,
    validate_training_controls,
)
from src.special_tokens import EOS_ID, PAD_ID  # noqa: E402
from src.tracking import Tracker  # noqa: E402


# -------------------------
# Dataset: jsonl offsets
# -------------------------
class JsonlOffsetsDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        off = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        return json.loads(line)


# -------------------------
# Read train/val jsonl
# -------------------------
def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preflight_sft_record(
    split: str,
    example: dict[str, Any],
    *,
    tok,
    seq_len: int,
    default_system: str | None,
) -> dict[str, int]:
    """Validate one full SFT row, or one prompt-only sampling row."""
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages list")

    if split == "sample_eval":
        prompt_messages = prepare_prompt_messages(messages, default_system)
        ids = encode_prompt(tok, prompt_messages, default_system=default_system, mode="full_context")
        kept_ids, _ = truncate_chat_sequence(ids, labels=None, max_len=seq_len - 1)
        return {
            "encoded_prompt_tokens": len(ids),
            "retained_prompt_tokens": len(kept_ids),
            "truncated_examples": int(len(kept_ids) != len(ids)),
        }

    ids, labels = encode_chat(tok, messages, default_system)
    if default_system is None and len(ids) > seq_len:
        raise ValueError("source-preserving SFT rejects overlength conversations")
    kept_ids, kept_labels = truncate_chat_sequence(ids, labels, max_len=seq_len)
    assert kept_labels is not None
    supervised = sum(label != IGNORE_INDEX for label in kept_labels[1:])
    if supervised == 0:
        raise ValueError("SFT example retained no supervised assistant target")
    return {
        "encoded_tokens": len(ids),
        "retained_tokens": len(kept_ids),
        "supervised_tokens": supervised,
        "truncated_examples": int(len(kept_ids) != len(ids)),
    }


# -------------------------
# Fixed prompts
# -------------------------
FIXED_PROMPTS = [
    "[Code] Write a Python function running_sum(nums) that returns cumulative sums.",
    "[Code] Write a Python function lowercase_keys(d) that returns a new dictionary with lowercase string keys.",
    "[General] Write a short polite email asking for an update on a job application after an interview.",
    "[General] Write a short professional email asking to reschedule a meeting to next week.",
    "[General] Summarize this in 3 bullet points: 'Regular exercise can improve mood, support heart health, and help maintain energy levels.'",
    "[General] Rewrite this to be more formal: 'Thanks for the quick reply. I’ll send the file tomorrow.'",
    "[General] Rewrite this to be more concise: 'I am writing this email in order to ask whether it would be possible to move our meeting to Friday afternoon.'",
    "[General] Explain in at most 4 sentences what a budget is, in simple everyday language.",
]


def collate_fn_builder(
    tok,
    seq_len: int,
    default_system: str | None,
    debug_first_batch: bool,
    refusal_downweight: float,
    refusal_patterns: list[str],
    refusal_mode: str,
):
    printed = {"done": False}

    def collate(batch: list[dict[str, Any]]):
        xs, ys, ws = [], [], []
        for ex in batch:
            x, y, w = build_example(
                ex,
                tok,
                seq_len,
                default_system,
                refusal_downweight,
                refusal_patterns,
                refusal_mode,
                pad_id=PAD_ID,
            )
            xs.append(x)
            ys.append(y)
            ws.append(w)

        input_ids = torch.stack(xs, dim=0)
        labels = torch.stack(ys, dim=0)
        weights = torch.tensor(ws, dtype=torch.float32)

        if debug_first_batch and not printed["done"]:
            printed["done"] = True
            sup = int((labels[0] != -100).sum().item())
            tot = labels[0].numel()
            print(f"[dbg] supervised tokens(sample0): {sup}/{tot} ({sup / tot:.3f})")
            print(f"[dbg] example_weight(sample0): {float(weights[0].item()):.3f}")

            idx = (labels[0] != -100).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                dec = tok.decode(input_ids[0, idx].tolist())
                print("[dbg] decoded supervised span(first 300 chars):")
                print(dec[:300])
            else:
                print("[dbg] WARNING: no supervised tokens in sample0")

        row_ids = [str(ex.get("audit_id") or hashlib.sha256(
            json.dumps(ex, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()) for ex in batch]
        return {"input_ids": input_ids, "labels": labels, "weights": weights, "row_ids": row_ids}

    return collate


def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    reduction: str = "token_mean",
) -> torch.Tensor:
    """
    Next-token CE, ignoring -100 labels.

    reduction:
      - token_mean: average over all supervised tokens in batch (mainstream
        default: every supervised token gets equal weight)
      - example_mean: mean CE per-example (over supervised tokens), then average
        across batch (upweights tokens in short answers)

    If weights provided (shape [B]), they act as per-example scalars (e.g. refusal downweight).
    """
    B, T, V = logits.size()
    # next-token
    logits2 = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    labels2 = labels[:, 1:].contiguous()  # [B, T-1]

    # per-token loss
    loss_tok = F.cross_entropy(
        logits2.view(-1, V),
        labels2.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T - 1)  # [B, T-1]

    mask = (labels2 != -100).float()  # [B, T-1]
    if weights is None:
        weights = torch.ones((B,), device=logits.device, dtype=torch.float32)
    else:
        weights = weights.to(device=logits.device, dtype=torch.float32)

    if reduction == "token_mean":
        w = weights.view(B, 1)  # [B,1]
        loss_tok = loss_tok * mask * w
        denom = (mask * w).sum().clamp_min(1.0)
        return loss_tok.sum() / denom

    if reduction == "example_mean":
        tok_cnt = mask.sum(dim=1).clamp_min(1.0)  # [B]
        loss_ex = (loss_tok * mask).sum(dim=1) / tok_cnt  # [B]
        denom = weights.sum().clamp_min(1.0)
        return (loss_ex * weights).sum() / denom

    raise ValueError(f"unknown reduction: {reduction}")


def supervised_target_count(labels: torch.Tensor) -> int:
    """Count shifted assistant/EOS targets and reject any all-masked row."""
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("expected [batch, sequence] labels with a next-token target")
    counts = (labels[:, 1:] != IGNORE_INDEX).sum(dim=1)
    if counts.numel() == 0 or bool((counts == 0).any().item()):
        raise ValueError("zero supervised targets in an example")
    return int(counts.sum().item())


def assistant_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Unweighted next-token CE numerator; includes every supervised EOS."""
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def effective_batch_backward(model, batches, *, device, autocast_dtype=None, nll_sum_fn=None) -> dict:
    """One CPU-buffered effective batch; backward CE_sum / shared target count.

    This function never steps an optimizer or retains a microbatch graph.
    Production permits bf16 single-GPU only; deterministic CPU tests use fp32.
    """
    if nll_sum_fn is None:
        nll_sum_fn = assistant_nll_sum
    if not batches:
        raise ValueError("empty effective batch")
    counts = []
    row_ids = []
    for batch in batches:
        weights = batch.get("weights")
        if weights is not None and not bool((weights == 1).all().item()):
            raise ValueError("effective_batch_tokens does not support sample weighting")
        counts.append(supervised_target_count(batch["labels"]))
        row_ids.extend(batch.get("row_ids", []))
    total_targets = sum(counts)
    nll_total = 0.0
    for batch in batches:
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits = model(inputs)
            nll = nll_sum_fn(logits, labels)
            loss = nll / total_targets
        if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(nll).item()):
            raise FloatingPointError("nonfinite effective-batch loss before backward/optimizer.step")
        nll_total += float(nll.detach().item())
        loss.backward()
        del loss, nll, logits, inputs, labels
    return {"loss": nll_total / total_targets, "nll_sum": nll_total,
            "supervised_targets": total_targets, "microbatches": len(batches),
            "rows": sum(int(b["labels"].shape[0]) for b in batches),
            "row_ids": row_ids,
            "row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest()}


def checked_optimizer_step(model, optimizer, *, loss: float, grad_clip: float) -> float:
    """The actual pilot step gate: fail before changing any optimizer state."""
    if not math.isfinite(loss):
        raise FloatingPointError("nonfinite loss before optimizer.step")
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("no gradients before optimizer.step")
    finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
    if not bool(finite.item()):
        raise FloatingPointError("nonfinite gradients before optimizer.step")
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, grad_clip if grad_clip > 0 else float("inf"), error_if_nonfinite=True
    )
    value = float(norm.item())
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite gradient norm before optimizer.step")
    optimizer.step()
    return value


@torch.no_grad()
def evaluate_token_mean(model, loader, *, device, autocast_dtype=None, expected_rows=None, should_stop=None) -> dict:
    """Full validation corpus CE sum / assistant+EOS targets, including singleton."""
    was_training = model.training
    model.eval()
    total_nll, total_targets, rows, batches = 0.0, 0, 0, 0
    row_ids = []
    try:
        for batch in loader:
            if should_stop is not None and should_stop():
                raise InterruptedError("pilot stop requested during validation")
            count = supervised_target_count(batch["labels"])
            inputs = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                logits = model(inputs)
                nll = assistant_nll_sum(logits, labels)
            if not bool(torch.isfinite(nll).item()):
                raise FloatingPointError("nonfinite validation NLL")
            total_nll += float(nll.item())
            total_targets += count
            rows += int(labels.shape[0])
            batches += 1
            row_ids.extend(batch.get("row_ids", []))
            del logits, nll, inputs, labels
    finally:
        model.train(was_training)
    if total_targets == 0:
        raise ValueError("zero supervised targets in validation")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"incomplete validation coverage: {rows} != {expected_rows}")
    if row_ids and (len(row_ids) != rows or len(set(row_ids)) != rows):
        raise ValueError("validation row identities are missing or duplicated")
    return {"val_loss": total_nll / total_targets, "nll_sum": total_nll,
            "supervised_targets": total_targets, "rows": rows, "batches": batches,
            "row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest()}


def write_pilot_status(out_dir: str, status: dict) -> None:
    path = Path(out_dir) / "TRAINING_STATUS.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint_atomic(path: str, obj: dict[str, Any]):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str) -> dict[str, Any]:
    # Accepted local checkpoints record NumPy's uint32 RNG state. Keep the
    # restricted unpickler and scope only those metadata types to this load.
    numpy_rng_types = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                       type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(numpy_rng_types):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                    help="muon: Muon on hidden matrices + AdamW on embeddings/norms (default). adamw: AdamW everywhere.")
    ap.add_argument("--muon_lr", type=float, default=0.0,
                    help="LR for Muon matrix groups (<=0: reuse --lr; Muon update RMS is matched to AdamW's).")
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--max_steps", type=int, default=15000)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="0 disables grad clipping")

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--init_from_pretrain", default="")
    ap.add_argument(
        "--resume",
        default="",
        help="Exact continuation from this run's own stage checkpoint. Requires "
        "matching arguments, inputs, runtime, optimizer/scaler, RNG, loop state, "
        "and deterministic data cursor; use --init_from_pretrain for weights-only.",
    )
    ap.add_argument("--default_system", default="You are a helpful assistant.")
    ap.add_argument("--preserve_messages", action="store_true",
                    help="Keep source content verbatim, add no system turn, and reject overlength training rows.")

    ap.add_argument("--n_layers", type=int, default=30)
    ap.add_argument("--d_model", type=int, default=576)
    ap.add_argument("--n_heads", type=int, default=9)
    ap.add_argument("--n_kv_heads", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie_embeddings", action="store_true")

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")

    ap.add_argument(
        "--loss_reduction",
        choices=["token_mean", "example_mean"],
        default="token_mean",
        help="token_mean (mainstream default): every supervised token weighs the same. "
        "example_mean: every example weighs the same (upweights short answers).",
    )

    ap.add_argument(
        "--loss_normalization", choices=["legacy", "effective_batch_tokens"], default="legacy",
        help="Explicit single-GPU bf16 pilot objective: every microbatch CE sum / shared effective-batch targets.",
    )

    ap.add_argument("--ce_precision", choices=["legacy", "selected_fp32"], default="legacy",
                    help="P2 opt-in: cast selected supervised logits to FP32 before disabled-autocast CE.")
    ap.add_argument("--p2_consumption_plan", default="",
                    help="Frozen finite two-pass P2 row plan; requires fresh effective-token SFT.")

    # ---- refusal downweight (TRAINING, not sampling) ----
    ap.add_argument(
        "--refusal_downweight",
        type=float,
        default=0.25,
        help="If an example contains refusal-like assistant text, multiply its loss by this factor (e.g. 0.1~0.5). 1.0=off",
    )
    ap.add_argument(
        "--refusal_mode",
        choices=["contains_any"],
        default="contains_any",
        help="How to detect refusal examples.",
    )
    ap.add_argument(
        "--refusal_patterns",
        type=str,
        default="i am not capable,i am not able,i do not have access,i can't,i cannot,as an ai,i'm unable,unable to,not allowed to,can't help with,do not have the capability",
        help="Comma-separated substrings used to detect refusals in assistant content.",
    )

    # sampling
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--samples_dir", type=str, default="")
    ap.add_argument("--sample_max_new_tokens", type=int, default=192)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=50)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument(
        "--sample_repetition_penalty",
        type=float,
        default=1.12,
        help=">1.0 discourages repeating tokens during sampling (e.g. 1.05~1.25)",
    )
    ap.add_argument(
        "--sample_repetition_window",
        type=int,
        default=256,
        help="how many recent tokens to apply repetition penalty over",
    )
    ap.add_argument(
        "--sample_no_repeat_ngram",
        type=int,
        default=3,
        help="disallow repeating n-grams of this size during sampling (0=off). e.g. 3",
    )

    # in-domain sampling (val_jsonl)
    ap.add_argument("--sample_in_domain_n", type=int, default=10)
    ap.add_argument("--sample_in_domain_seed", type=int, default=1234)
    ap.add_argument("--sample_in_domain_show_ref", action="store_true")
    ap.add_argument(
        "--sample_in_domain_mode",
        choices=["last_user", "full_context"],
        default="full_context",
    )
    ap.add_argument(
        "--sample_in_domain_dump_prompt",
        action="store_true",
        help="also dump the rendered prompt (decoded) for debugging inputs",
    )
    ap.add_argument(
        "--sample_eval_jsonl",
        default="",
        help="Optional curated eval jsonl for sampling. If set, this is used instead of random val_jsonl in-domain sampling.",
    )
    ap.add_argument(
        "--sample_only_ckpt",
        default="",
        help="Optional ckpt path. If set, load this checkpoint, write samples once, then exit without training.",
    )
    return ap


def validate_sft_args(args: argparse.Namespace) -> None:
    if getattr(args, "preserve_messages", False):
        args.default_system = None
    validate_training_controls(
        args,
        positive_fields=(
            "micro_bsz",
            "grad_accum",
            "eval_batches",
            "sample_max_new_tokens",
        ),
        nonnegative_fields=(
            "num_workers",
            "eval_every",
            "save_every",
            "sample_every",
            "sample_in_domain_n",
        ),
    )
    if getattr(args, "loss_normalization", "legacy") == "effective_batch_tokens":
        if args.loss_reduction != "token_mean" or args.refusal_downweight != 1.0 or args.refusal_patterns.strip():
            raise ValueError("effective_batch_tokens requires token_mean, refusal_downweight=1 and empty refusal_patterns")
        if args.precision != "bf16":
            raise ValueError("effective_batch_tokens pilot supports bf16 only")
        if args.resume:
            raise ValueError("effective_batch_tokens pilot requires fresh weights-only initialization, no resume")
    if args.ce_precision == "selected_fp32" and args.loss_normalization != "effective_batch_tokens":
        raise ValueError("selected_fp32 requires effective_batch_tokens")
    if args.p2_consumption_plan:
        if (args.loss_normalization != "effective_batch_tokens" or args.ce_precision != "selected_fp32"
                or not args.init_from_pretrain or not args.preserve_messages or args.resume):
            raise ValueError("P2 requires fresh Base, preserve_messages, effective tokens and selected_fp32")
        if args.micro_bsz != 2 or args.grad_accum != 16 or args.seq_len != 2048 or args.optimizer != "adamw":
            raise ValueError("P2 fixed geometry is micro2, accum16, context2048, AdamW")
        if args.max_steps > 1000:
            raise ValueError("P2 may not exceed 1000 updates")
    if args.seq_len <= 1:
        raise ValueError("--seq_len must be greater than 1")
    if args.resume and args.sample_only_ckpt:
        raise ValueError("--resume and --sample_only_ckpt are mutually exclusive")


def main(
    argv: list[str] | None = None, *, stage_kind: str = "sft"
) -> None:
    if stage_kind not in {"sft", "distill"}:
        raise ValueError("stage_kind must be 'sft' or 'distill'")
    args = build_arg_parser().parse_args(argv)
    validate_sft_args(args)
    run_args = {**vars(args), "stage_kind": stage_kind}
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    effective_tokens = args.loss_normalization == "effective_batch_tokens"
    p2_plan = None
    ce_sum_fn = assistant_nll_sum
    validation_fn = evaluate_token_mean
    if args.ce_precision == "selected_fp32":
        from sft.p2_loss import assistant_nll_sum_fp32
        from sft.p2_evaluation import evaluate_position_nll
        ce_sum_fn = assistant_nll_sum_fp32
        validation_fn = evaluate_position_nll
    if effective_tokens:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if (stage_kind != "sft" or device != "cuda" or torch.cuda.device_count() != 1
                or int(os.environ.get("WORLD_SIZE", "1")) != 1
                or (distributed and torch.distributed.get_world_size() != 1)):
            raise ValueError("effective_batch_tokens pilot requires one CUDA GPU and a single process")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("effective_batch_tokens pilot requires native bf16 support")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok = load_chat_tokenizer(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    tracker = Tracker(args.out_dir)
    tracker.log_run_start(run_args, args.tokenizer_path)

    preflight_datasets = {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
    }
    if args.sample_eval_jsonl:
        preflight_datasets["sample_eval"] = args.sample_eval_jsonl
    report = run_jsonl_preflight(
        stage=stage_kind,
        datasets=preflight_datasets,
        validate_record=lambda split, record: preflight_sft_record(
            split,
            record,
            tok=tok,
            seq_len=args.seq_len,
            default_system=args.default_system,
        ),
        report_path=os.path.join(args.out_dir, "posttrain_preflight.json"),
        metadata={"seq_len": args.seq_len, "default_system": args.default_system},
    )
    tracker.log(
        "preflight",
        0,
        stage=stage_kind,
        status=report["status"],
        records=report["total_records"],
        valid=report["total_valid"],
        rejected=report["total_rejected"],
    )
    require_preflight_passed(report)
    if effective_tokens:
        train_rows = int(report["splits"]["train"]["records"])
        val_rows = int(report["splits"]["val"]["records"])
        if args.p2_consumption_plan:
            from sft.p2_plan import load_checked_plan
            if not 8000 <= train_rows <= 16000 or not 300 <= val_rows <= 800:
                raise ValueError("P2 selected supply is outside the authorized train/validation range")
            p2_plan = load_checked_plan(args.p2_consumption_plan, train_path=args.train_jsonl,
                dataset_rows=train_rows, seed=args.seed, micro_bsz=args.micro_bsz,
                grad_accum=args.grad_accum, max_steps=args.max_steps)
            if args.warmup_steps != p2_plan["warmup_steps"]:
                raise ValueError("P2 warmup differs from the complete frozen plan")
        elif args.max_steps * args.grad_accum > train_rows // args.micro_bsz:
            raise ValueError("effective_batch_tokens pilot may not enter a second epoch")
        if args.eval_batches < math.ceil(val_rows / args.micro_bsz):
            raise ValueError("effective_batch_tokens requires complete validation coverage")

    resume_checkpoint: Mapping[str, Any] | None = None
    start_step = 0
    if args.resume:
        loaded_resume = load_ckpt(args.resume)
        if not isinstance(loaded_resume, Mapping):
            raise RuntimeError("--resume checkpoint must be a mapping")
        resume_checkpoint = loaded_resume
        start_step = require_resume_step(
            resume_checkpoint,
            stage=stage_kind,
            weights_only_hint="--init_from_pretrain",
        )

    resume_inputs: dict[str, str] = {
        "tokenizer": args.tokenizer_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
    }
    if args.p2_consumption_plan:
        resume_inputs["p2_consumption_plan"] = args.p2_consumption_plan
    if args.sample_eval_jsonl:
        resume_inputs["sample_eval_jsonl"] = args.sample_eval_jsonl
    if args.init_from_pretrain:
        resume_inputs["init_from_pretrain"] = args.init_from_pretrain
    resume_contract_base = build_resume_contract_base(
        stage=stage_kind,
        args=run_args,
        input_paths=resume_inputs,
        dataset_size=int(report["splits"]["train"]["records"]),
        batch_size=args.micro_bsz,
        batches_per_step=args.grad_accum,
        seed=args.seed,
    )
    if p2_plan is not None:
        resume_contract_base["data_order"].update({
            "sampler": p2_plan["schema"], "drop_last": "terminal complete-update groups only",
            "max_passes": 2, "planned_updates": p2_plan["max_steps"],
            "consumed_row_ids_sha256": p2_plan["consumed_row_ids_sha256"],
            "unused_tail_rows": p2_plan["unused_tail_rows"], "resume_authorized": False})
    if resume_checkpoint is not None:
        validate_resume_contract(
            resume_checkpoint,
            resume_contract_for_step(resume_contract_base, start_step),
            weights_only_hint="--init_from_pretrain",
        )

    refusal_patterns = [
        p.strip() for p in args.refusal_patterns.split(",") if p.strip()
    ]

    cfg: GPTConfig | None = None

    if args.init_from_pretrain:
        ck = load_ckpt(args.init_from_pretrain)
        cfg_dict = ck.get("config") or ck.get("cfg")
        if not isinstance(cfg_dict, dict):
            raise RuntimeError("pretrain ckpt missing 'config' dict")
        cfg_dict = dict(cfg_dict)
        if p2_plan is not None and (cfg_dict["vocab_size"] != vocab_size or cfg_dict["max_seq_len"] != args.seq_len):
            raise ValueError("P2 must retain the exact Base model configuration")
        cfg_dict["vocab_size"] = vocab_size
        cfg_dict["max_seq_len"] = args.seq_len
        cfg = gpt_config_from_checkpoint_dict(cfg_dict)
        model = GPT(cfg).to(device)

        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("pretrain ckpt missing 'model'")
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}

        model.load_state_dict(sd, strict=True)
        print(f"[*] initialized from pretrain: {args.init_from_pretrain}")
    else:
        cfg = GPTConfig(
            vocab_size=vocab_size,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
            tie_embeddings=args.tie_embeddings,
        )
        model = GPT(cfg).to(device)

    optimizer = build_optimizer(
        model,
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
    )

    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = (
        torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if resume_checkpoint is not None:
        restore_training_state(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            use_fp16=use_fp16,
        )
        print(f"[*] resumed: {args.resume} at step={start_step}")

    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    val_ds = JsonlOffsetsDataset(args.val_jsonl)
    sample_eval_ds: list[dict[str, Any]] = []
    if args.sample_eval_jsonl:
        sample_eval_ds = read_jsonl(args.sample_eval_jsonl)
        print(
            f"[*] sample_eval_jsonl: {args.sample_eval_jsonl} lines={len(sample_eval_ds)}"
        )

    print(f"[*] dataset: train_lines={len(train_ds)} val_lines={len(val_ds)}")
    print(
        f"[*] effective_tokens/step = micro_bsz({args.micro_bsz}) * grad_accum({args.grad_accum}) * seq_len({args.seq_len})"
        f" = {args.micro_bsz * args.grad_accum * args.seq_len}"
    )
    print(
        f"[*] refusal downweight: mode={args.refusal_mode} downweight={args.refusal_downweight} patterns={len(refusal_patterns)}"
    )

    if p2_plan is not None:
        from sft.p2_plan import MaterializedP2BatchSampler
        train_batch_sampler = MaterializedP2BatchSampler(p2_plan)
    else:
        train_batch_sampler = DeterministicEpochBatchSampler(
            len(train_ds), args.micro_bsz, seed=args.seed,
            start_batch=start_step * args.grad_accum, drop_last=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(
            tok,
            args.seq_len,
            args.default_system,
            args.debug_first_batch,
            args.refusal_downweight,
            refusal_patterns,
            args.refusal_mode,
        ),
        generator=make_loader_generator(args.seed, 1),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(
            tok,
            args.seq_len,
            args.default_system,
            False,
            args.refusal_downweight,
            refusal_patterns,
            args.refusal_mode,
        ),
        drop_last=False,
        generator=make_loader_generator(args.seed, 2),
    )

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    @torch.no_grad()
    def sample_from_ids(prompt_ids: list[int]) -> str:
        """Generate a completion for an already-encoded prompt (which ends with
        the <|assistant|> cue) and decode ONLY the generated tokens."""
        was_training = model.training
        model.eval()

        prompt_ids, _ = truncate_chat_sequence(
            prompt_ids, labels=None, max_len=args.seq_len - 1
        )

        ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
        prompt_len = ids.size(1)
        generation_steps = min(
            args.sample_max_new_tokens, args.seq_len - prompt_len
        )
        g = torch.Generator(device=device)
        g.manual_seed(args.sample_seed)

        def top_k_top_p(
            logits_1d: torch.Tensor, top_k: int, top_p: float
        ) -> torch.Tensor:
            if top_k and top_k > 0:
                k = min(top_k, logits_1d.size(-1))
                v, _ = torch.topk(logits_1d, k)
                thresh = v[-1]
                logits_1d = torch.where(
                    logits_1d < thresh,
                    torch.full_like(logits_1d, -float("inf")),
                    logits_1d,
                )
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask_sorted = cum > top_p
                mask_sorted[0] = False
                mask = torch.zeros_like(mask_sorted)
                mask.scatter_(0, sorted_idx, mask_sorted)
                logits_1d = torch.where(
                    mask, torch.full_like(logits_1d, -float("inf")), logits_1d
                )
            return logits_1d

        def apply_repetition_penalty(
            logits_1d: torch.Tensor, prev_ids: list[int], penalty: float
        ) -> torch.Tensor:
            if penalty is None or penalty <= 1.0:
                return logits_1d
            if not prev_ids:
                return logits_1d
            uniq = set(prev_ids)
            for tid in uniq:
                if tid < 0 or tid >= logits_1d.numel():
                    continue
                v = logits_1d[tid]
                logits_1d[tid] = v / penalty if v > 0 else v * penalty
            return logits_1d

        def ban_repeat_ngrams(
            logits_1d: torch.Tensor, generated: list[int], n: int
        ) -> torch.Tensor:
            if n is None or n <= 0:
                return logits_1d
            if len(generated) < n - 1:
                return logits_1d
            prefix = tuple(generated[-(n - 1) :])
            banned = set()
            for i in range(len(generated) - n + 1):
                if tuple(generated[i : i + n - 1]) == prefix:
                    banned.add(generated[i + n - 1])
            if banned:
                for t in banned:
                    if 0 <= t < logits_1d.numel():
                        logits_1d[t] = -1e10
            return logits_1d

        for _ in range(generation_steps):
            logits = model(ids)
            next_logits = logits[0, -1, :].float()

            if args.sample_temperature <= 0:
                nxt = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / args.sample_temperature
                recent = ids[0, -args.sample_repetition_window :].tolist()
                next_logits = apply_repetition_penalty(
                    next_logits, recent, args.sample_repetition_penalty
                )
                gen_so_far = ids[0].tolist()
                next_logits = ban_repeat_ngrams(
                    next_logits, gen_so_far, args.sample_no_repeat_ngram
                )
                next_logits = top_k_top_p(
                    next_logits, top_k=args.sample_top_k, top_p=args.sample_top_p
                )
                probs = torch.softmax(next_logits, dim=-1)
                if torch.isnan(probs).any() or float(probs.sum().item()) == 0.0:
                    nxt = int(torch.argmax(next_logits).item())
                else:
                    nxt = int(
                        torch.multinomial(probs, num_samples=1, generator=g).item()
                    )

            ids = torch.cat(
                [ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1
            )
            if nxt == EOS_ID:
                break

        completion = decode_completion(tok, ids[0, prompt_len:].tolist())
        if was_training:
            model.train()
        return completion

    def build_fixed_prompt_ids(user_q: str) -> list[int]:
        return encode_prompt(
            tok,
            [{"role": "user", "content": user_q}],
            default_system=args.default_system,
            mode="last_user",
        )

    def build_sampling_examples() -> tuple[str, list[tuple[str, dict[str, Any]]]]:
        """
        Returns:
          eval_name: str
          rows: list[(tag, example_dict)]
        """
        rows: list[tuple[str, dict[str, Any]]] = []
        if sample_eval_ds:
            for i, ex in enumerate(sample_eval_ds, start=1):
                meta = ex.get("meta") or {}
                tag = str(meta.get("name", f"eval_{i:02d}"))
                rows.append((tag, ex))
            return "sample_eval_jsonl", rows

        if args.sample_in_domain_n > 0 and len(val_ds) > 0:
            n = min(args.sample_in_domain_n, len(val_ds))
            idxs = [in_rng.randrange(len(val_ds)) for _ in range(n)]
            for idx in idxs:
                ex = val_ds[idx]
                rows.append((f"idx={idx}", ex))
            return "val_jsonl", rows

        return "", rows

    in_rng = random.Random(args.sample_in_domain_seed)

    @torch.no_grad()
    def emit_samples(step_tag: str) -> None:
        sdir = args.samples_dir or os.path.join(args.out_dir, "samples")
        Path(sdir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(sdir, f"{step_tag}.txt")

        lines: list[str] = []
        lines.append(f"step={step_tag}\n")
        lines.append(
            f"sampling: temp={args.sample_temperature} top_p={args.sample_top_p} top_k={args.sample_top_k} max_new={args.sample_max_new_tokens}\n"
        )
        lines.append(
            f"in_domain: n={args.sample_in_domain_n} mode={args.sample_in_domain_mode} "
            f"show_ref={bool(args.sample_in_domain_show_ref)} dump_prompt={bool(args.sample_in_domain_dump_prompt)} "
            f"sample_eval_jsonl={bool(args.sample_eval_jsonl)}\n"
        )
        lines.append("=" * 80 + "\n")

        # A) fixed prompts
        lines.append("[Fixed prompts]\n")
        lines.append("-" * 80 + "\n")
        for i, q in enumerate(FIXED_PROMPTS):
            ans = sample_from_ids(build_fixed_prompt_ids(q))
            lines.append(f"[Q{i + 1}] {q}\n")
            lines.append(f"[A{i + 1}] {ans}\n")
            lines.append("-" * 80 + "\n")

        # B) curated eval prompts or fallback val_jsonl
        eval_name, eval_rows = build_sampling_examples()
        if eval_rows:
            lines.append(f"\n[In-domain prompts from {eval_name}]\n")
            lines.append("-" * 80 + "\n")
            tags = [tag for tag, _ in eval_rows]
            lines.append(f"[eval tags] {tags}\n")
            lines.append("-" * 80 + "\n")

            for k, (tag, ex) in enumerate(eval_rows, start=1):
                msgs = ex.get("messages") or []
                user_q, ref_a = extract_last_user_and_ref(msgs)
                if not user_q:
                    continue

                prompt_messages = prepare_prompt_messages(msgs, args.default_system)
                prompt_ids = encode_prompt(
                    tok, prompt_messages, args.default_system, args.sample_in_domain_mode
                )
                ans = sample_from_ids(prompt_ids)

                meta = ex.get("meta") or {}
                bucket = str(meta.get("bucket", ""))
                lines.append(f"[V{k}] {tag} bucket={bucket}\n")
                lines.append("[User]\n")
                lines.append(f"{user_q}\n")

                if args.sample_in_domain_dump_prompt:
                    lines.append("[Prompt]\n")
                    p = tok.decode(prompt_ids)
                    if len(p) > 4000:
                        p = p[:4000] + "\n...[truncated]\n"
                    lines.append(p + "\n")

                if args.sample_in_domain_show_ref and ref_a:
                    lines.append("[Ref assistant]\n")
                    lines.append(ref_a + "\n")

                lines.append("[Model]\n")
                lines.append(ans + "\n")
                lines.append("-" * 80 + "\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[sample] wrote {out_path}")

    if args.sample_only_ckpt:
        ck = load_ckpt(args.sample_only_ckpt)
        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("sample_only_ckpt missing 'model'")
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        print(f"[*] sample-only loaded: {args.sample_only_ckpt}")
        model.eval()
        emit_samples(Path(args.sample_only_ckpt).stem)
        print("[done]")
        return

    running_loss = 0.0
    if resume_checkpoint is not None:
        loop_state = resume_checkpoint.get("loop_state")
        if not isinstance(loop_state, Mapping):
            raise RuntimeError("SFT exact resume checkpoint lacks loop_state")
        try:
            running_loss = float(loop_state["running_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "SFT exact resume has invalid running_loss state"
            ) from exc
        if not math.isfinite(running_loss):
            raise RuntimeError("SFT exact resume running_loss must be finite")

        aux_rng_state = resume_checkpoint.get("aux_rng_state")
        if not isinstance(aux_rng_state, Mapping):
            raise RuntimeError("SFT exact resume checkpoint lacks auxiliary RNG state")
        try:
            in_rng.setstate(aux_rng_state["sample_in_domain"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "SFT exact resume has invalid in-domain sampling RNG state"
            ) from exc
        restore_rng_state(resume_checkpoint["rng_state"])

    model.train()
    t0 = time.time()
    step = start_step
    last_saved_step: int | None = None

    def save_training_checkpoint(checkpoint_step: int) -> None:
        nonlocal last_saved_step
        if last_saved_step == checkpoint_step:
            return
        ckpt_path = os.path.join(args.out_dir, f"step_{checkpoint_step:06d}.pt")
        ckpt = {
            "step": checkpoint_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_fp16 else None,
            "cfg": asdict(cfg) if cfg is not None else None,
            "args": run_args,
            "kind": stage_kind,
            "resume_contract": resume_contract_for_step(
                resume_contract_base, checkpoint_step
            ),
            "rng_state": capture_rng_state(),
            "aux_rng_state": {"sample_in_domain": in_rng.getstate()},
            "loop_state": {"running_loss": running_loss},
        }
        save_checkpoint_atomic(ckpt_path, ckpt)
        save_checkpoint_atomic(os.path.join(args.out_dir, "latest.pt"), ckpt)
        last_saved_step = checkpoint_step
        print(f"[ckpt] saved {ckpt_path}")

    train_iter = iter(train_loader)
    pilot_started = time.perf_counter()
    stop_requested = {"signal": None}
    previous_handlers = {}
    buffered_batches = []
    update_rng_state = None
    phase = "before_first_update"
    if effective_tokens:
        def request_pilot_stop(signum, _frame):
            stop_requested["signal"] = signum
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_pilot_stop)
        torch.cuda.reset_peak_memory_stats()
        write_pilot_status(args.out_dir, {"status": "RUNNING", "last_completed_step": step,
                           "pid": os.getpid(), "loss_normalization": args.loss_normalization})

    try:
        try:
            while step < args.max_steps:
                if effective_tokens and stop_requested["signal"] is not None:
                    break
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                micro_loss = 0.0

                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * pg.get("lr_ratio", 1.0)

                if effective_tokens:
                    phase = "buffering_before_optimizer"
                    buffered_batches = []
                    for _ in range(args.grad_accum):
                        try:
                            buffered_batches.append(next(train_iter))
                        except StopIteration as error:
                            raise RuntimeError("finite pilot stream exhausted; refusing unplanned exposure") from error
                    if p2_plan is not None:
                        planned = p2_plan["updates"][step]
                        actual_ids = [rid for batch in buffered_batches for rid in batch["row_ids"]]
                        actual_targets = sum(supervised_target_count(batch["labels"]) for batch in buffered_batches)
                        if actual_ids != planned["row_ids"] or actual_targets != planned["supervised_targets"]:
                            raise RuntimeError("P2 buffered rows/targets differ from frozen update plan")
                    update_rng_state = capture_rng_state()
                    phase = "forward_backward_before_optimizer"
                    update_metrics = effective_batch_backward(
                        model, buffered_batches, device=device, autocast_dtype=autocast_dtype,
                        nll_sum_fn=ce_sum_fn
                    )
                    micro_loss = update_metrics["loss"]
                    phase = "finite_gate_before_optimizer"
                    preclip_norm = checked_optimizer_step(
                        model, optimizer, loss=micro_loss, grad_clip=args.grad_clip
                    )
                else:
                    for _ in range(args.grad_accum):
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            batch = next(train_iter)

                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)
                        weights = batch["weights"].to(device, non_blocking=True)

                        with torch.autocast(
                            device_type="cuda",
                            dtype=autocast_dtype,
                            enabled=(autocast_dtype is not None),
                        ):
                            logits = model(input_ids)
                            loss = (
                                masked_ce_loss(
                                    logits, labels, weights=weights, reduction=args.loss_reduction
                                )
                                / args.grad_accum
                            )

                        if use_fp16:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        micro_loss += float(loss.item())

                    if args.grad_clip > 0:
                        if use_fp16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    if use_fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                step += 1
                running_loss += micro_loss
                if effective_tokens:
                    phase = "completed_update"
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - update_started
                    total_elapsed = time.perf_counter() - pilot_started
                    event = {"event": "optimizer_update", "step": step, **update_metrics,
                             "lr": lr, "preclip_gradient_norm": preclip_norm,
                             "elapsed_seconds": elapsed, "training_elapsed_seconds": total_elapsed,
                             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                             "loss_normalization": args.loss_normalization}
                    with open(os.path.join(args.out_dir, "UPDATE_METRICS.jsonl"), "a") as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    tracker.log("train", step, loss=micro_loss, lr=lr,
                                supervised_targets=update_metrics["supervised_targets"],
                                preclip_gradient_norm=preclip_norm, elapsed_seconds=elapsed,
                                peak_cuda_allocated_bytes=event["peak_cuda_allocated_bytes"])
                    write_pilot_status(args.out_dir, {"status": "RUNNING", "last_completed_step": step,
                                       "last_update": event, "pid": os.getpid()})
                    print(f"[train] step={step} loss={micro_loss:.6f} targets={update_metrics['supervised_targets']} "
                          f"lr={lr:.8g} grad_norm={preclip_norm:.6f} elapsed={elapsed:.3f}s", flush=True)
                    buffered_batches = []
                    if stop_requested["signal"] is not None:
                        break

                if not effective_tokens and step % 50 == 0:
                    dt = time.time() - t0
                    tokens_per_step = args.micro_bsz * args.grad_accum * args.seq_len
                    train_loss_avg = running_loss / 50
                    tok_s = tokens_per_step * 50.0 / max(dt, 1e-9)
                    print(
                        f"[train] step={step} loss={train_loss_avg:.4f} lr={get_lr(step):.2e} tok/s≈{tok_s:.0f} dt={dt:.1f}s"
                    )
                    tracker.log("train", step, loss=train_loss_avg, lr=get_lr(step), tok_s=tok_s)
                    running_loss = 0.0
                    t0 = time.time()

                evaluate_due = (step in p2_plan["evaluation_steps"] if p2_plan is not None
                                else args.eval_every > 0 and step % args.eval_every == 0)
                if evaluate_due:
                    if effective_tokens:
                        phase = "validation"
                        validation = validation_fn(
                            model, val_loader, device=device, autocast_dtype=autocast_dtype,
                            expected_rows=len(val_ds), should_stop=lambda: stop_requested["signal"] is not None,
                        )
                        with open(os.path.join(args.out_dir, "VALIDATION_METRICS.jsonl"), "a") as stream:
                            stream.write(json.dumps({"step": step, **validation}, allow_nan=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        tracker.log("val", step, **validation, loss_normalization="corpus_token_mean")
                        print(f"[eval] step={step} val_loss={validation['val_loss']:.6f} "
                              f"rows={validation['rows']} targets={validation['supervised_targets']}", flush=True)
                    else:
                        model.eval()
                        losses = []
                        with torch.no_grad():
                            for j, vb in enumerate(val_loader):
                                if j >= args.eval_batches:
                                    break
                                vi = vb["input_ids"].to(device)
                                vl = vb["labels"].to(device)
                                with torch.autocast(
                                    device_type="cuda",
                                    dtype=autocast_dtype,
                                    enabled=(autocast_dtype is not None),
                                ):
                                    v_logits = model(vi)
                                    # Unweighted token-mean: val loss stays a clean metric,
                                    # comparable across runs and refusal/weight settings.
                                    v_loss = masked_ce_loss(
                                        v_logits, vl, weights=None, reduction="token_mean"
                                    )
                                losses.append(float(v_loss.item()))
                        val_loss_avg = sum(losses) / max(1, len(losses))
                        print(
                            f"[eval] step={step} val_loss={val_loss_avg:.4f}"
                        )
                        tracker.log("val", step, val_loss=val_loss_avg)
                        tracker.render()
                        model.train()

                # ---- Sampling (fixed + in-domain) ----
                if (
                    args.sample_every
                    and args.sample_every > 0
                    and step % args.sample_every == 0
                ):
                    emit_samples(f"step_{step:06d}")

                save_due = (step in p2_plan["checkpoint_steps"] if p2_plan is not None
                            else args.save_every > 0 and step % args.save_every == 0)
                if save_due:
                    save_training_checkpoint(step)

        except InterruptedError:
            if not effective_tokens or stop_requested["signal"] is None:
                raise
        save_training_checkpoint(step)
    except Exception as error:
        if effective_tokens:
            failure = {"status": "FAILED", "last_completed_step": step,
                       "attempted_step": step + 1 if phase.endswith("optimizer") else step,
                       "phase": phase, "exception_type": type(error).__name__, "error": str(error),
                       "pid": os.getpid(), "optimizer_step_retried": False, "exit_reason": "exception",
                       "buffered_row_ids": [rid for batch in buffered_batches for rid in batch.get("row_ids", [])]}
            try:
                evidence_path = os.path.join(args.out_dir, f"FAILED_STEP_{step:06d}.pt")
                save_checkpoint_atomic(evidence_path, {
                    "last_completed_step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "buffered_batches": buffered_batches,
                    "update_start_rng_state": update_rng_state, "failure": failure,
                    "gradients": {name: param.grad for name, param in model.named_parameters() if param.grad is not None},
                })
                failure["evidence_path"] = evidence_path
            except Exception as evidence_error:
                failure["evidence_write_error"] = type(evidence_error).__name__ + ": " + str(evidence_error)
            write_pilot_status(args.out_dir, failure)
            print(json.dumps(failure), flush=True)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if effective_tokens:
        interrupted = stop_requested["signal"] is not None and step < args.max_steps
        write_pilot_status(args.out_dir, {"status": "INTERRUPTED" if interrupted else "COMPLETED",
                           "last_completed_step": step, "signal": stop_requested["signal"],
                           "exit_reason": "operator_signal" if interrupted else "max_steps_reached",
                           "pid": os.getpid(), "training_elapsed_seconds": time.perf_counter() - pilot_started,
                           "loss_normalization": args.loss_normalization})
        if interrupted:
            raise SystemExit(128 + stop_requested["signal"])
    tracker.render()
    print("[done]")


if __name__ == "__main__":
    main()
