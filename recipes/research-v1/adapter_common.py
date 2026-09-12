"""Small shared identity and source-binding helpers; importing performs no model work."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "sources/posttraining"
SUPPORT = HERE / "runtime-support-v2"
PROVENANCE = HERE / "provenance/runtime-support-v2"
TOKENIZER_SHA = "d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Identity mismatch: {path}: expected {expected}, got {actual}")
    return actual


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fresh(path):
    if path.exists() or path.is_symlink():
        raise ValueError(f"Requires a fresh nonexistent output directory: {path}")


def verify_sources():
    for item in read(HERE / "provenance/INTEGRATION.json")["files"]:
        if item["repository_path"].startswith("recipes/research-v1/sources/posttraining/"):
            identity(REPO / item["repository_path"], item["sha256"])
    for item in read(HERE / "provenance/RELOCATION_V2.json")["files"]:
        identity(REPO / item["repository_path"], item["sha256"])


def activate_source():
    # Check transitive modules too: a poisoned src.model must not survive an absent src.
    for name, module in tuple(sys.modules.items()):
        if name in ("src", "sft") or name.startswith(("src.", "sft.")):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(SOURCE):
                raise RuntimeError(f"Wrong source import: {name}: {path}")
    sys.path[:] = [str(SOURCE), *[p for p in sys.path if p != str(SOURCE)]]


def definitions(path, names, namespace, *, transform=None):
    """Bind an explicit list of frozen functions without running startup orchestration.

    Used only by execution branches (or AST-only tests). No whole-module imports of
    prepare/curriculum/blend/export scripts; function bodies stay source-derived.
    """
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f"Frozen function set changed: {path}")
    if transform:
        selected = [transform(n) for n in selected]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec", optimize=0), namespace)
    return namespace


def modes(parser, execute_help):
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only", action="store_true", help="Read-only identities/schema; no model work"
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help=execute_help + "; requires separate bounded owner authorization",
    )


def cli(main):
    if sys.flags.optimize:
        raise SystemExit("Do not use -O: frozen execution assertions are required")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
        raise SystemExit(f"Adapter failed: {exc}") from exc
