"""New-run input contracts. Declarations are checked separately from loaded tensors."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from adapter_common import HERE, REPO, SOURCE, TOKENIZER_SHA, identity, read, sha, write

CONFIG = read(HERE / "runtime-support-v2/contracts/INTERPOLATION_EXPORT_IDENTITIES.json")[
    "recorded_config"
]


def source(root=SOURCE):
    for name, module in tuple(sys.modules.items()):
        if name in ("src", "sft", "pretrain", "dataset_pretrain", "sample") or name.startswith((
            "src.",
            "sft.",
            "pretrain.",
        )):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(root):
                raise RuntimeError(f"Wrong source import: {name}: {location}; use a fresh process")
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(root / "pretrain"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_code():
    # The V2 map remains immutable; V3 adds a separate current-reader identity map.
    from adapter_common import verify_sources

    verify_sources()
    for item in read(HERE / "provenance/INTEGRATION.json")["files"]:
        if "/sources/" in item["repository_path"]:
            identity(REPO / item["repository_path"], item["sha256"])
    for item in read(HERE / "provenance/READER_V3.json")["files"]:
        identity(REPO / item["path"], item["sha256"])


def config(value):
    if value != CONFIG or any(type(value[k]) is not type(v) for k, v in CONFIG.items()):
        raise ValueError("Incompatible complete research-v1 architecture/config")
    return value


def tokenizer(path):
    identity(path, TOKENIZER_SHA)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    names = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<|system|>", "<|user|>", "<|assistant|>"]
    if tok.get_vocab_size() != 32000 or [tok.token_to_id(n) for n in names] != list(range(7)):
        raise ValueError("Incompatible canonical tokenizer token IDs/vocabulary")
    return tok


def checkpoint(path, kinds=None, step=None):
    meta = read(str(path) + ".reader.json")
    if meta["schema"] != "petitgpt-new-checkpoint-v3" or meta["tokenizer_sha256"] != TOKENIZER_SHA:
        raise ValueError("Incompatible checkpoint/tokenizer declaration")
    config(meta["config"])
    identity(path, meta["sha256"])
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if kinds and meta["kind"] not in kinds:
        raise ValueError(f"Expected parent kind {kinds}, got {meta['kind']}")
    if step is not None and meta["step"] != step:
        raise ValueError(f"Expected checkpoint step {step}, got {meta['step']}")
    return meta


def receipt(path, kind, step, *, checked=False, **extra):
    value = {
        "schema": "petitgpt-new-checkpoint-v3",
        "kind": kind,
        "step": step,
        "config": CONFIG,
        "tokenizer_sha256": TOKENIZER_SHA,
        "sha256": sha(path),
        "tensor_contract_checked": checked,
        "original_release": False,
        **extra,
    }
    write(str(path) + ".reader.json", value)
    return value


def encoded(path, tok, *, limit, count=None):
    from src.chat_template import encode_chat

    rows = []
    ids = set()
    messages = set()
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        aid = r.get("audit_id")
        ms = r.get("messages")
        if not isinstance(aid, str) or not aid or aid in ids:
            raise ValueError("audit_id must be unique nonempty strings")
        if (
            not isinstance(ms, list)
            or not ms
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str)
                for m in ms
            )
        ):
            raise ValueError("messages requires role/content objects")
        tokens, labels = encode_chat(tok, ms, default_system=None)
        targets = [
            t for m in ms if m["role"] == "assistant" for t in tok.encode(m["content"]).ids + [3]
        ]
        if not targets or len(tokens) > limit or [v for v in labels[1:] if v != -100] != targets:
            raise ValueError("Context or shifted assistant/EOS masking mismatch")
        if any(v != -100 and v != t for t, v in zip(tokens, labels, strict=True)):
            raise ValueError("Shifted labels mismatch")
        if type(r.get("shifted_supervised_tokens")) is not int or r[
            "shifted_supervised_tokens"
        ] != len(targets):
            raise ValueError("shifted_supervised_tokens must equal encoded assistant/EOS targets")
        key = json.dumps(ms, sort_keys=True, ensure_ascii=False)
        ids.add(aid)
        messages.add(key)
        rows.append(r)
    if not rows or (count is not None and len(rows) != count):
        raise ValueError(f"Expected {count or 'nonempty'} records, got {len(rows)}")
    return rows, ids, messages


def disjoint(a, b):
    if a[1] & b[1] or a[2] & b[2]:
        raise ValueError("Input splits overlap by audit_id or exact messages")


def shapes():
    d = 576
    ff = 1536
    result = {"tok_emb.weight": (32000, d), "lm_head.weight": (32000, d), "norm_f.weight": (d,)}
    for i in range(30):
        for key, shape in {
            "norm1.weight": (d,),
            "norm2.weight": (d,),
            "attn.qkv.weight": (960, d),
            "attn.proj.weight": (d, d),
            "mlp.w1.weight": (ff, d),
            "mlp.w3.weight": (ff, d),
            "mlp.w2.weight": (d, ff),
        }.items():
            result[f"blocks.{i}.{key}"] = shape
    return result


def state(sd):
    import torch

    original_count = len(sd)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    if original_count != len(sd):
        raise ValueError("Compiled prefix collision")
    if len(sd) != 213 or set(sd) != set(shapes()):
        raise ValueError("Checkpoint state keys differ from native architecture")
    for k, s in shapes().items():
        if (
            tuple(sd[k].shape) != s
            or sd[k].dtype != torch.float32
            or not bool(torch.isfinite(sd[k]).all())
        ):
            raise ValueError(f"Checkpoint shape/dtype/finite contract failed: {k}")
    if not torch.equal(sd["tok_emb.weight"], sd["lm_head.weight"]):
        raise ValueError("Tied aliases have different values")
    return sd


def raw_load(path):
    """Same restricted NumPy RNG allowlist as recovered loader, for NumPy 1.x/2.x."""
    import numpy as np
    import torch

    core = np._core if hasattr(np, "_core") else np.core
    with torch.serialization.safe_globals([
        core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("uint32")),
    ]):
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
    config(ck.get("config") or ck.get("cfg"))
    ck["model"] = state(ck["model"])
    return ck


def load(path):
    meta = checkpoint(path)
    ck = raw_load(path)
    actual_step = ck.get("global_step", ck.get("step", 0))
    if actual_step != meta["step"]:
        raise ValueError("Loaded checkpoint step differs from declaration")
    return ck
