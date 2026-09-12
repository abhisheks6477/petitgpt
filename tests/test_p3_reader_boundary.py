"""P3 production binding with opaque files and inert callbacks; no torch/model work.

The real extracted train/checkpoint functions run against stubs. The deliberately
sparse synthetic plan only visits boundaries; it is not a valid training corpus,
a 640-update run, or evidence about tensors/numerical training behavior.
"""

import ast
import collections
import copy
import datetime
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/research-v1"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))

    class BlockModels:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "safetensors", "src", "sft"}:
                raise AssertionError("PROHIBITED_IMPORT:" + fullname)

    monkeypatch.setattr(sys, "meta_path", [BlockModels(), *sys.meta_path])
    return SimpleNamespace(
        helper=importlib.import_module("reader_p3"),
        common=importlib.import_module("adapter_common"),
        contracts=importlib.import_module("reader_contracts"),
    )


def harness(tmp_path, api, steps=(320, 640), save_fault=None, later_fault=False):
    """Bind the production helper and actual frozen train/checkpoint functions."""
    events = []
    data = tmp_path / "logical-inputs"
    data.mkdir()
    train = tmp_path / "train"
    train.mkdir()
    supplied = tmp_path / "supplied.jsonl"
    supplied.write_text(json.dumps({"audit_id": "synthetic", "messages": []}) + "\n")
    replay = tmp_path / "replay.jsonl"
    replay.write_text("")
    plan = tmp_path / "PLAN.jsonl"
    plan.write_text(
        "".join(
            json.dumps({"update": s, "row_ids": ["synthetic"] * 32, "stream": "synthetic"}) + "\n"
            for s in steps
        )
    )
    inputs = tmp_path / "INPUTS.json"
    api.common.write(inputs, {"execution": "NOT_RUN", "scope": "synthetic stub control flow"})
    mapping = {
        data / "train.jsonl": supplied,
        data / "replay.jsonl": replay,
        data / "UPDATE_PLAN.jsonl": plan,
        data / "FROZEN.json": inputs,
    }

    def rows(path):
        events.append(("read", path.name))
        if path not in mapping:
            raise AssertionError("UNEXPECTED_PRIVATE_INPUT:" + str(path))
        return api.common.rows(mapping[path])

    def forbidden(*args, **kwargs):
        raise AssertionError("PRIVATE_HOOK_EXECUTED")

    model = SimpleNamespace(
        train=lambda: None, parameters=lambda: (), state_dict=lambda: {"stub": True}
    )
    opt = SimpleNamespace(
        state={}, param_groups=[{}], zero_grad=lambda **k: None, state_dict=lambda: {"stub": True}
    )

    def save(path, obj):
        events.append(("save", obj["step"]))
        assert obj["config"] == api.contracts.CONFIG
        assert obj["scheduler"] == {
            "canonical": True,
            "completed_updates": obj["step"],
            "planned_updates": 640,
            "warmup_steps": 32,
            "base_lr": 5e-5,
            "min_lr_ratio": 0.1,
        }
        if save_fault == "before":
            raise OSError("synthetic save failed")
        if save_fault == "missing":
            return
        temp = Path(path + ".tmp")
        temp.write_bytes(b"Synthetic opaque checkpoint, not model tensors")
        if save_fault == "temporary":
            raise OSError("synthetic temporary save failed")
        temp.replace(path)
        events.append(("finalized", obj["step"]))
        if save_fault == "after":
            raise OSError("synthetic failure after writing weights")

    def reset():
        existing = sorted(train.glob("*.pt"))
        events.append(("cuda_reset_stub", len(existing)))
        # Reset at a boundary must observe its receipt already published.
        for p in existing:
            api.contracts.checkpoint(p, ["p3"], int(p.stem.split("_")[1]))

    def backward_stub(_model, batches, **kwargs):
        if later_fault and (train / "step_000320.pt.reader.json").exists():
            raise RuntimeError("synthetic later failure")
        ids = [k for b in batches for k in b["row_ids"]]
        return {"row_ids": ids, "supervised_targets": len(ids) * 2, "loss": 0.0}

    def schedule(step, warmup, lr, **kwargs):
        assert warmup == 32 and lr == 5e-5
        assert kwargs == {"schedule": "cosine", "schedule_total_steps": 640}
        events.append(("lr_stub", step))
        return lr

    cuda = SimpleNamespace(
        manual_seed_all=lambda s: None,
        reset_peak_memory_stats=reset,
        synchronize=lambda: None,
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
        get_rng_state_all=lambda: [],
    )
    torch = SimpleNamespace(
        cuda=cuda,
        manual_seed=lambda s: None,
        optim=SimpleNamespace(AdamW=lambda *a, **k: opt),
        stack=lambda x: x,
        tensor=lambda x, **k: x,
        long="stub",
        bfloat16="stub",
        get_rng_state=lambda: [],
    )
    ns = dict(
        torch=torch,
        collections=collections,
        datetime=datetime,
        json=json,
        os=os,
        random=random.Random(),
        signal=SimpleNamespace(
            SIGALRM=1, ITIMER_REAL=1, signal=lambda *a: None, setitimer=lambda *a: None
        ),
        sys=sys,
        time=time,
        sha=lambda p: api.common.sha(mapping.get(p, p)),
        rows=rows,
        write=api.common.write,
        canonical=api.common.canonical,
        CKPT=tmp_path / "unused-parent",
        CONFIG={"scope": "synthetic stub control flow"},
        SEED=20260907,
        DATA=data,
        TRAIN=train,
        EVAL=tmp_path / "absent-evaluation",
        save_checkpoint_atomic=save,
        encode_chat=lambda *a, **k: ([5, 6, 7, 3], [-100, -100, 7, 3]),
        lr_schedule=schedule,
        effective_batch_backward=backward_stub,
        checked_optimizer_step=lambda *a, **k: 0.0,
        assistant_nll_sum_fp32=forbidden,
        evaluate=forbidden,
        val500=forbidden,
    )
    api.helper.bind_new_run(ns)
    # Model loading is outside this control-flow test. All boundary validators,
    # the extracted checkpoint function and receipt writer remain real.
    ns["load_model"] = lambda *a: (model, dict(api.contracts.CONFIG))
    return ns, events, mapping


@pytest.mark.parametrize("step", [320, 640])
def test_actual_bound_train_boundary_finalizes_receipt_before_cuda_reset(tmp_path, api, step):
    ns, events, _ = harness(tmp_path, api, steps=(step,))
    ns["train"](None, time.monotonic())
    path = ns["TRAIN"] / f"step_{step:06d}.pt"
    meta = api.contracts.checkpoint(path, ["p3"], step)
    assert meta["sha256"] == api.common.sha(path)
    assert meta["config"] == api.contracts.CONFIG
    assert meta["tokenizer_sha256"] == api.common.TOKENIZER_SHA
    assert meta["original_release"] is False
    assert events.count(("finalized", step)) == 1
    assert events.index(("finalized", step)) < events.index(("cuda_reset_stub", 1))
    assert [x for x in events if x[0] == "read"] == [
        ("read", "train.jsonl"),
        ("read", "replay.jsonl"),
        ("read", "UPDATE_PLAN.jsonl"),
    ]
    assert api.common.read(ns["TRAIN"] / "STATUS.json")["status"] == "COMPLETED"
    assert not ns["EVAL"].exists()


def test_step320_receipt_survives_later_failure_and_links_interpolation(tmp_path, api):
    ns, _, _ = harness(tmp_path, api, later_fault=True)
    with pytest.raises(RuntimeError, match="synthetic later failure"):
        ns["train"](None, time.monotonic())
    assert api.common.read(ns["TRAIN"] / "STATUS.json")["status"] == "STOPPED"
    p3 = ns["TRAIN"] / "step_000320.pt"
    p2 = tmp_path / "p2.pt"
    p2.write_bytes(b"synthetic opaque p2 parent")
    api.contracts.receipt(p2, "p2", 750)
    run = importlib.import_module("reader_posttrain").run
    report = run(
        SimpleNamespace(
            stage="interpolate", p2_parent=p2, p3_parent=p3, include_alpha050=False, execute=False
        )
    )
    assert report["execution"] == "NOT_RUN" and report["alphas"] == [0.75]
    assert report["parents"][1]["sha256"] == api.common.sha(p3)
    assert not (ns["TRAIN"] / "step_000640.pt.reader.json").exists()
    assert not (tmp_path / "READER_STATUS.json").exists()


@pytest.mark.parametrize("fault", ["before", "after", "missing", "temporary"])
def test_failed_frozen_save_has_no_receipt_or_completion(tmp_path, api, fault):
    ns, _, _ = harness(tmp_path, api, steps=(320,), save_fault=fault)
    with pytest.raises(OSError):
        ns["train"](None, time.monotonic())
    assert not list(ns["TRAIN"].glob("*.reader.json"))
    assert api.common.read(ns["TRAIN"] / "STATUS.json")["status"] == "STOPPED"
    assert not (tmp_path / "READER_STATUS.json").exists()


def test_receipt_write_failure_propagates_and_stops_run(tmp_path, api, monkeypatch):
    ns, _, _ = harness(tmp_path, api, steps=(320,))

    def failed_write(path, value):
        raise OSError("synthetic receipt write failure")

    monkeypatch.setattr(api.contracts, "write", failed_write)
    with pytest.raises(OSError, match="synthetic receipt write failure"):
        ns["train"](None, time.monotonic())
    assert (ns["TRAIN"] / "step_000320.pt").is_file()
    assert not list(ns["TRAIN"].glob("*.reader.json"))
    status = api.common.read(ns["TRAIN"] / "STATUS.json")
    assert status["status"] == "STOPPED" and "receipt write failure" in status["error"]
    assert not (tmp_path / "READER_STATUS.json").exists()


@pytest.mark.parametrize("input_name", ["train.jsonl", "replay.jsonl", "UPDATE_PLAN.jsonl"])
def test_nonomitted_missing_input_still_fails(tmp_path, api, input_name):
    ns, _, mapping = harness(tmp_path, api)
    mapping[ns["DATA"] / input_name].unlink()
    with pytest.raises(FileNotFoundError):
        ns["train"](None, time.monotonic())
    assert api.common.read(ns["TRAIN"] / "STATUS.json")["status"] == "STOPPED"
    assert not list(ns["TRAIN"].glob("*.reader.json"))


def frozen_train(api):
    return next(
        n
        for n in ast.parse(api.helper.P3_SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "train"
    )


def boundary(node):
    return next(
        n
        for n in ast.walk(node)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "step in (320, 640)"
    )


def test_transform_changes_only_three_omitted_statements(api):
    original = frozen_train(api)
    expected = copy.deepcopy(original)
    expected.body.pop(0)
    del boundary(expected).body[1:3]
    transformed = api.helper.new_run_train(original)
    assert ast.dump(transformed) == ast.dump(expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "gate",
        "argument",
        "duplicate_hook",
        "duplicate_boundary",
        "checkpoint",
        "reset",
        "extra_statement",
    ],
)
def test_transform_rejects_unexpected_hook_shapes_and_counts(api, mutation):
    node = frozen_train(api)
    b = boundary(node)
    if mutation == "gate":
        node.body[0] = ast.parse("assert True").body[0]
    elif mutation == "argument":
        b.body[1].value.args[2] = ast.Constant(None)
    elif mutation == "duplicate_hook":
        node.body.append(copy.deepcopy(b.body[1]))
    elif mutation == "duplicate_boundary":
        node.body.append(copy.deepcopy(b))
    elif mutation == "checkpoint":
        b.body[0].value.args[3] = ast.Constant(320)
    elif mutation == "reset":
        b.body.pop()
    else:
        b.body.append(ast.Pass())
    with pytest.raises(ValueError, match="Unexpected new-run P3"):
        api.helper.new_run_train(node)


@pytest.mark.parametrize("fault", ["missing", "temporary", "wrong_step", "bad_config"])
def test_receipt_wrapper_rejects_unfinalized_or_incompatible_save(tmp_path, api, fault):
    path = tmp_path / "step_000320.pt"
    calls = []

    def save(*args):
        calls.append("save")
        if fault == "temporary":
            temp = Path(str(path) + ".tmp")
            temp.write_bytes(b"synthetic temporary file")
            return temp
        return path

    cfg = dict(api.contracts.CONFIG)
    if fault == "bad_config":
        cfg["n_layers"] = 1
    wrapped = api.helper.checkpoint_with_receipt(save, tmp_path)
    with pytest.raises(ValueError):
        wrapped(None, None, cfg, 321 if fault == "wrong_step" else 320, {})
    assert not list(tmp_path.glob("*.reader.json"))
    if fault in ("wrong_step", "bad_config"):
        assert not calls


def test_historical_boundary_and_recorded_bytes_remain_unchanged(tmp_path, api):
    assert (
        api.common.sha(api.helper.P3_SOURCE)
        == "50e1a842d41569d90f6cbe4d0ac4ba428ab3e59afc21ba2cff5b4b4b3f3c4f5b"
    )
    assert (
        api.common.sha(RECIPE / "p3_execution.py")
        == "13b5c5d70fe2eea68efe5fb2ca882876391f4ff7881f3448d843a0e1ec320bd4"
    )
    original = frozen_train(api)
    assert ast.unparse(original.body[0]) == "assert (EVAL / 'BASELINE_COMPLETE.json').exists()"
    events = []
    ns = dict(
        step=320,
        model=None,
        opt=None,
        cfg={},
        status={},
        tok=None,
        DATA=tmp_path,
        checkpoint=lambda *a: events.append("checkpoint"),
        rows=lambda p: events.append(p.name) or ["synthetic historical hook row"],
        evaluate=lambda *a: events.append("evaluate"),
        val500=lambda *a: events.append("val500"),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: events.append("reset"))
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[boundary(original)], type_ignores=[])),
            str(api.helper.P3_SOURCE),
            "exec",
        ),
        ns,
    )
    assert events == ["checkpoint", "development.jsonl", "evaluate", "val500", "reset"]
    # Historical-schedule still uses no transform. V2 training-only retains its
    # old baseline-only gate and is deliberately not claimed repaired here.
    old = importlib.import_module("p3_execution")
    assert ast.dump(boundary(old.training_only_gate(frozen_train(api)))) == ast.dump(
        boundary(original)
    )


def test_production_p3_execute_uses_new_binding_without_delayed_receipt_loop(api):
    node = next(
        n
        for n in ast.parse((RECIPE / "reader_posttrain.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "p3_execute"
    )
    assert ast.unparse(node.body[-2]) == "bind_new_run(ns)"
    assert ast.unparse(node.body[-1]) == "ns['train'](load_chat_tokenizer(str(a.tokenizer)), start)"
