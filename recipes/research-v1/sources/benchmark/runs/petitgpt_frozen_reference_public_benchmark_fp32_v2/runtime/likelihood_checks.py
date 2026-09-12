"""Small validation helpers; not a model loader or a complete benchmark runner.

No files are written and no CUDA work is launched when this module is imported.
All tolerances describe engineering checks, not task-accuracy guarantees.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch

TOKEN_ATOL = 5e-4
TOKEN_RTOL = 1e-5
SAME_LOGITS_ATOL = 1e-5
SAME_LOGITS_RTOL = 1e-6


def continuation_logprobs(
    logits: torch.Tensor, context_length: int, continuation_ids: Sequence[int]
) -> torch.Tensor:
    """Score all K continuation tokens at positions C-1,...,C+K-2.

    `logits` comes from a causal full forward of (context + continuation)[:-1],
    possibly with a test-only FUTURE suffix. No prompt or suffix is scored.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected logits [1, time, vocabulary].")
    if context_length < 1 or not continuation_ids:
        raise ValueError("Nonempty context and continuation required.")
    if logits.dtype != torch.float32:
        raise ValueError("Production V2 logits must be FP32, not BF16 cast afterward.")
    k = len(continuation_ids)
    stop = context_length + k - 1
    if logits.shape[1] < stop:
        raise ValueError("Insufficient logits for the complete continuation.")
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=logits.device)
    if bool(((targets < 0) | (targets >= logits.shape[2])).any()):
        raise ValueError("Token id outside model vocabulary.")
    selected = logits[0, context_length - 1 : stop, :]
    values = selected.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Non-finite continuation log-probability.")
    return values


def token_parity(left: Sequence[float], right: Sequence[float]) -> dict:
    """Pointwise test plus a sum bound derived from pointwise allowances.

    Uses symmetric relative scaling. Not a BF16-to-FP32 equality gate.
    """
    if len(left) != len(right) or not left:
        raise ValueError("Equal nonempty token vectors required.")
    if not all(math.isfinite(float(v)) for v in list(left) + list(right)):
        raise ValueError("Non-finite values.")
    errors = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    allowances = [TOKEN_ATOL + TOKEN_RTOL * max(abs(a), abs(b))
                  for a, b in zip(left, right)]
    return {
        "token_count": len(left),
        "token_atol": TOKEN_ATOL,
        "token_rtol": TOKEN_RTOL,
        "max_token_absolute_error": max(errors),
        "sum_absolute_error": abs(math.fsum(left) - math.fsum(right)),
        "sum_allowance": math.fsum(allowances),
        "per_token_errors": errors,
        "per_token_allowances": allowances,
        "pass": all(e <= a for e, a in zip(errors, allowances))
                and abs(math.fsum(left) - math.fsum(right)) <= math.fsum(allowances),
    }


def sum_parity(left: float, right: float, token_count: int) -> dict:
    """Cross-forward scalar helper check when only a sequence sum is exposed."""
    if token_count < 1 or not all(math.isfinite(v) for v in (left, right)):
        raise ValueError("Finite scores and a positive target count required.")
    allowed = token_count * TOKEN_ATOL + TOKEN_RTOL * max(abs(left), abs(right))
    return {"absolute_error": abs(left-right), "allowed_error": allowed,
            "pass": abs(left-right) <= allowed}


def same_logits_sum_parity(left: float, right: float) -> dict:
    allowed = SAME_LOGITS_ATOL + SAME_LOGITS_RTOL * max(abs(left), abs(right))
    return {"absolute_error": abs(left-right), "allowed_error": allowed,
            "pass": all(math.isfinite(v) for v in (left, right))
                    and abs(left-right) <= allowed}


def record_before_check(path: Path, record: dict, passed: bool) -> None:
    """Persist a completed measurement even if the next statement will raise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
    if not passed:
        raise RuntimeError("Numerical gate failed; see the already-persisted record.")


def metrics(scores: Sequence[float], candidate_texts: Sequence[str], gold: int) -> dict:
    if len(scores) != len(candidate_texts) or not scores or not 0 <= gold < len(scores):
        raise ValueError("Invalid choice set.")
    if not all(math.isfinite(s) for s in scores) or any(len(t) == 0 for t in candidate_texts):
        raise ValueError("Nonfinite likelihood or empty candidate text.")
    # Python max returns the first maximum, as does the frozen numpy.argmax rule.
    raw = max(range(len(scores)), key=lambda j: scores[j])
    norm = max(range(len(scores)), key=lambda j: scores[j] / len(candidate_texts[j]))
    return {"prediction_acc": raw, "prediction_acc_norm": norm,
            "acc": int(raw == gold), "acc_norm": int(norm == gold),
            "normalization_denominators": [len(t) for t in candidate_texts]}
