"""P2 assistant-only CE: BF16 forward values, explicit FP32 loss arithmetic.

The caller still owns effective-update normalization, finite gates, and the
optimizer. This module only computes the next-token supervised NLL numerator.
"""
from typing import Literal

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _selected_ce_fp32(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: Literal["none", "sum"],
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected aligned [batch, sequence, vocabulary] logits and [batch, sequence] labels")
    if logits.shape[1] < 2:
        raise ValueError("expected at least one shifted next-token position")
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != IGNORE_INDEX
    selected_labels = shifted_labels[mask]
    if selected_labels.numel() == 0:
        raise ValueError("zero supervised shifted targets")
    # Select first, so ignored role/user/system/padding logits are neither cast
    # nor passed through log-softmax. Every nonignored EOS is retained.
    selected_logits = logits[:, :-1, :][mask]
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.cross_entropy(selected_logits.float(), selected_labels, reduction=reduction)


def assistant_nll_sum_fp32(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """FP32 summed CE over shifted nonignored targets, including assistant EOS.

    Gradients flow through the FP32 cast to the original forward graph. Divide
    this numerator once by all supervised targets in the effective update;
    there is no per-example averaging or grad_accum division in this helper.
    """
    return _selected_ce_fp32(logits, labels, "sum")


def assistant_token_nll_fp32(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """FP32 per-target NLL in row-major ``labels[:, 1:] != -100`` order.

    This vector supports disjoint validation position buckets. Its length is
    the existing shifted supervised-target count; no ignored slots are emitted.
    """
    return _selected_ce_fp32(logits, labels, "none")
