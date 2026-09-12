"""Small CPU tensor operations shared with the reader export and tested on toy state."""

from __future__ import annotations


def save_native(sd, path, aliases):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import torch

    for alias, target in aliases.items():
        if not torch.equal(sd[alias], sd[target]):
            raise ValueError("Alias values differ")
    save_file(
        {k: v.contiguous() for k, v in sd.items() if k not in aliases}, str(path), metadata=aliases
    )
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if f.metadata() != aliases:
            raise ValueError("Serialized alias metadata differs")
    restored = load_file(str(path), device="cpu")
    for alias, target in aliases.items():
        restored[alias] = restored[target]
    if set(restored) != set(sd):
        raise ValueError("Serialized state keys differ")
    for k, v in sd.items():
        if (
            v.dtype != restored[k].dtype
            or v.shape != restored[k].shape
            or not torch.equal(v, restored[k])
        ):
            raise ValueError(f"Source/export mismatch: {k}")
    return restored
