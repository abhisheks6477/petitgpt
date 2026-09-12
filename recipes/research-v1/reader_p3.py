"""New-run-only P3 hook omission and receipts for finalized checkpoints.

The frozen trainer and V2 training-only/historical bindings remain unchanged.
"""

from __future__ import annotations

import ast
from functools import wraps
from pathlib import Path

from adapter_common import definitions
from p3 import P3_SOURCE
from reader_contracts import config, receipt


def new_run_train(node):
    """Remove precisely the baseline gate and two private hook expressions.

    Check the entire boundary before mutation, including checkpoint and CUDA
    bookkeeping. Dropping call expressions prevents eager argument evaluation.
    Any changed/moved/duplicated hook requires explicit adapter review.
    """
    if node.name != "train":
        return node
    gate = ast.parse("assert (EVAL / 'BASELINE_COMPLETE.json').exists()").body[0]
    expected = ast.parse(
        """
if step in (320, 640):
    checkpoint(model, opt, cfg, step, status)
    evaluate(model, tok, rows(DATA / 'development.jsonl'), f'p3_step{step}_development')
    val500(model, tok, f'p3_step{step}')
    torch.cuda.reset_peak_memory_stats()
"""
    ).body[0]
    if not node.body or ast.dump(node.body[0]) != ast.dump(gate):
        raise ValueError("Unexpected new-run P3 baseline gate shape")
    boundaries = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(expected.test)
    ]
    if len(boundaries) != 1 or ast.dump(boundaries[0]) != ast.dump(expected):
        raise ValueError("Unexpected new-run P3 checkpoint boundary shape/count")
    for name in ("checkpoint", "evaluate", "val500"):
        calls = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        ]
        if len(calls) != 1:
            raise ValueError(f"Unexpected new-run P3 {name} call count")
    node.body = node.body[1:]
    boundary = boundaries[0]
    boundary.body = [boundary.body[0], boundary.body[3]]
    return node


def checkpoint_with_receipt(save, train_dir):
    """Publish metadata only after the frozen checkpoint function finalizes a save.

    Its atomic weight save and identity record must both succeed. Receipt errors
    propagate through the frozen trainer's STOPPED handler; an earlier checkpoint
    remains usable independently of the later run status. No resume is added.
    """

    @wraps(save)
    def finalized(model, opt, cfg, step, status):
        if type(step) is not int or step not in (320, 640):
            raise ValueError("New-run P3 receipts require exact step320 or step640")
        actual_config = dict(config(cfg))
        expected = Path(train_dir) / f"step_{step:06d}.pt"
        sidecar = Path(str(expected) + ".reader.json")
        if expected.exists() or expected.is_symlink() or sidecar.exists() or sidecar.is_symlink():
            raise ValueError("P3 checkpoint/receipt must be new")
        path = Path(save(model, opt, cfg, step, status))
        if path != expected or path.is_symlink() or not path.is_file():
            raise ValueError("P3 checkpoint did not finalize the expected weight file")
        receipt(path, "p3", step, checked=True, config=actual_config)
        return path

    return finalized


def bind_new_run(ns):
    """Production binding, also usable with explicit model-free test callbacks."""
    definitions(
        P3_SOURCE,
        ["load_model", "encoded_rows", "checkpoint", "train"],
        ns,
        transform=new_run_train,
    )
    ns["checkpoint"] = checkpoint_with_receipt(ns["checkpoint"], ns["TRAIN"])
    return ns
