#!/usr/bin/env python3
"""Build a fresh native inference bundle from pinned frozen assets and alpha075.

The historical nine-line build_bundle.py excerpt is evidence, never executed here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from adapter_common import (
    HERE,
    SOURCE,
    SUPPORT,
    TOKENIZER_SHA,
    activate_source,
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


def asset_bindings(root, tokenizer, contract):
    result = {}
    for item in contract["frozen_portability_inputs"]:
        name = item["path"]
        if name == "model.safetensors":
            continue  # Historical weight identity is evidence, never a copied input.
        if name == "tokenizer.json":
            path = tokenizer
        elif name in (
            "src/__init__.py",
            "src/model.py",
            "src/chat_template.py",
            "src/special_tokens.py",
        ):
            # The supplied frozen-native root is explicitly a subset. Reuse the
            # already recovered, hash-identical source closure for these four files.
            path = root / name if (root / name).exists() else SOURCE / name
        else:
            path = root / name
        identity(path, item["sha256"])
        result[name] = path.resolve()
    return result


def checkpoint_binding(args, contract):
    if args.interpolation_receipt is None:
        return identity(args.source_checkpoint, contract["export_source_checkpoint_sha256"])
    receipt = read(args.interpolation_receipt)
    if (
        receipt["schema"] != "FIXED_INTERPOLATION_OUTPUT_V2"
        or receipt["tokenizer_sha256"] != TOKENIZER_SHA
        or [p["sha256"] for p in receipt["parents"]] != [p["sha256"] for p in contract["parents"]]
    ):
        raise ValueError("Invalid fixed interpolation receipt bindings")
    candidates = receipt["artifacts"]
    if len(candidates) != 2 or [(c["tag"], c["alpha"]) for c in candidates] != [
        ("alpha050", 0.5),
        ("alpha075", 0.75),
    ]:
        raise ValueError("Receipt must describe only the two fixed candidates")
    for new, old in zip(candidates, contract["fixed_candidates"], strict=True):
        if new["model_state_digest"] != old["model_state_digest"]:
            raise ValueError("Receipt tensor identity differs from historical record")
    # This is a declared tensor identity until --execute independently recomputes
    # it. Receipt paths are informational; the explicit source path is authoritative.
    return identity(args.source_checkpoint, candidates[1]["sha256"])


def validate(args):
    fresh(args.out_dir)
    verify_sources()
    contract = read(SUPPORT / "contracts/INTERPOLATION_EXPORT_IDENTITIES.json")
    actual_sha = checkpoint_binding(args, contract)
    assets = asset_bindings(args.frozen_inference_root, args.tokenizer, contract)
    return assets, {
        "schema": "NATIVE_EXPORT_BINDINGS_V2",
        "validation": "opaque identities and frozen asset bindings only",
        "source_checkpoint": {"path": str(args.source_checkpoint.resolve()), "sha256": actual_sha},
        "identity_mode": "original-alpha075"
        if args.interpolation_receipt is None
        else "adapter-interpolation-receipt",
        "assets": {name: {"path": str(p), "sha256": sha(p)} for name, p in assets.items()},
        "historical_export_sha256": next(
            x["sha256"]
            for x in contract["frozen_portability_inputs"]
            if x["path"] == "model.safetensors"
        ),
        "new_export_sha256": None,
        "new_outputs": "NOT_GENERATED",
        "parity": "NOT_RUN",
        "tensor_equality_established": False,
        "model_operations": 0,
    }


def package(run, bundle):
    files = [
        {"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
    ]
    write(
        bundle / "MANIFEST.json",
        {"schema": "NATIVE_BUNDLE_V2", "files": files, "self_excluded": "MANIFEST.json"},
    )
    archive = run / "petitgpt-alpha075-native.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    expected = {p.relative_to(bundle).as_posix(): p for p in bundle.rglob("*") if p.is_file()}
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Native bundle archive member set mismatch")
        for member in members:
            if not member.isfile() or member.size != expected[member.name].stat().st_size:
                raise ValueError("Native bundle archive member type/size mismatch")
            with tar.extractfile(member) as f:
                digest = (
                    hashlib.file_digest(f, "sha256").hexdigest()
                    if hasattr(hashlib, "file_digest")
                    else _stream_sha(f)
                )
            if digest != sha(expected[member.name]):
                raise ValueError("Native bundle archive hash mismatch")
    (run / (archive.name + ".sha256")).write_text(sha(archive) + "  " + archive.name + "\n")


def _stream_sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def execute(args, assets, evidence):
    activate_source()
    from safetensors import safe_open
    from safetensors.torch import load_model, save_model
    import torch

    from sft.train_sft import load_ckpt
    from src.model import GPT, audit_gpt_parameter_count, gpt_config_from_checkpoint_dict
    from src.special_tokens import assert_tokenizer_contract

    contract = read(SUPPORT / "contracts/INTERPOLATION_EXPORT_IDENTITIES.json")
    ck = load_ckpt(str(args.source_checkpoint))
    cfg = gpt_config_from_checkpoint_dict(ck.get("config") or ck.get("cfg"))
    if dataclasses.asdict(cfg) != contract["recorded_config"]:
        raise ValueError("Complete native checkpoint configuration mismatch")
    sd = ck["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        raise ValueError("Native export forbids compiled state prefixes")
    ns = {"hashlib": hashlib}
    definitions(
        SOURCE / "runs/p3_retention_two_point_interpolation_20260907/runtime/blend.py",
        ["state_digest"],
        ns,
    )
    if ns["state_digest"](sd) != contract["fixed_candidates"][1]["model_state_digest"]:
        raise ValueError("Alpha075 tensor digest differs from historical candidate")
    model = GPT(cfg).eval()
    model.load_state_dict(sd, strict=True)
    audit = audit_gpt_parameter_count(model, cfg)
    assert audit["actual_total"] == contract["unique_parameters"]
    assert model.tok_emb.weight is model.lm_head.weight
    assert torch.equal(sd["tok_emb.weight"], sd["lm_head.weight"])
    assert len(sd) == 213
    assert all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in sd.values())
    for name, t in model.state_dict().items():
        assert (
            t.dtype == sd[name].dtype and t.shape == sd[name].shape and torch.equal(t, sd[name])
        ), name
    fresh(args.out_dir)
    args.out_dir.mkdir(parents=True)
    bundle = args.out_dir / "bundle"
    (bundle / "src").mkdir(parents=True)
    for name, path in assets.items():
        shutil.copyfile(path, bundle / name)
    # Preserve the complete config formatting from actual export_weights.py.
    (bundle / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
    identity(bundle / "config.json", evidence["assets"]["config.json"]["sha256"])
    save_model(model, str(bundle / "model.safetensors"))
    with safe_open(str(bundle / "model.safetensors"), framework="pt", device="cpu") as f:
        aliases, stored = f.metadata(), list(f.keys())
    assert len(stored) == 212 and aliases == {"tok_emb.weight": "lm_head.weight"}
    reloaded = GPT(cfg).eval()
    load_model(reloaded, str(bundle / "model.safetensors"), strict=True, device="cpu")
    assert reloaded.tok_emb.weight is reloaded.lm_head.weight
    assert reloaded.tok_emb.weight.data_ptr() == reloaded.lm_head.weight.data_ptr()
    assert audit_gpt_parameter_count(reloaded, cfg)["actual_total"] == 124635456
    assert set(model.state_dict()) == set(reloaded.state_dict()) == set(sd)
    comparisons = []
    for name, t in sd.items():
        other = reloaded.state_dict()[name]
        assert t.dtype == other.dtype and t.shape == other.shape and torch.equal(t, other), name
        comparisons.append({
            "name": name,
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "equal": True,
            "value_sha256": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(),
        })
    mb, rb = dict(model.named_buffers()), dict(reloaded.named_buffers())
    assert mb.keys() == rb.keys() and len(mb) == 60
    assert all(torch.equal(t, rb[name]) for name, t in mb.items())
    assert_tokenizer_contract(str(bundle / "tokenizer.json"))
    identity(args.source_checkpoint, evidence["source_checkpoint"]["sha256"])
    tensor_evidence = {
        "source_strict_load": True,
        "source_to_reloaded_comparisons": comparisons,
        "nonpersistent_buffers_equal": list(mb),
        "aliases": aliases,
        "named_state_entries": len(sd),
        "stored_tensor_count": len(stored),
        "unique_parameters": audit,
        "new_export_sha256": sha(bundle / "model.safetensors"),
        "parity": "NOT_RUN",
    }
    write(args.out_dir / "TENSOR_EQUALITY.json", tensor_evidence)
    write(
        bundle / "MODEL_PROVENANCE.json",
        {
            **evidence,
            "new_outputs": "GENERATED",
            "new_export_sha256": tensor_evidence["new_export_sha256"],
            "tensor_equality_established": True,
            "model_operations": "conversion only; see external TENSOR_EQUALITY.json",
        },
    )
    write(
        bundle / "EVALUATION_INDEX.json",
        {
            "parity": "NOT_RUN",
            "evaluation": "NOT_RUN",
            "historical_evidence": "Original release evidence is historical; not reproduced by this export.",
        },
    )
    (bundle / "README.md").write_text(
        "# PetitGPT native export\n\nNative GPT with tied FP32 safetensors; no HF AutoModel claim.\n"
        "See MODEL_PROVENANCE.json and EVALUATION_INDEX.json. Parity has not been run.\n"
        "Install the recorded requirements only in a separately authorized environment.\n"
        "Run `python inference.py --help` for the frozen inference interface.\n"
    )
    for name in ("LICENSE", "SOURCE_NOTICE.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(HERE / "provenance/notices" / name, bundle / name)
    package(args.out_dir, bundle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-checkpoint", "frozen-inference-root", "tokenizer", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--interpolation-receipt",
        type=Path,
        help="Optional new fixed-interpolation receipt; execute also recomputes the historical tensor digest",
    )
    modes(
        parser,
        "Future CPU checkpoint conversion, strict safetensors reload, bundle and archive creation",
    )
    args = parser.parse_args(argv)
    assets, evidence = validate(args)
    if args.execute:
        execute(args, assets, evidence)
    else:
        print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    cli(main)
