"""Model-free regression checks for release source isolation and public path binding."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/research-v1"


def _adapter():
    spec = importlib.util.spec_from_file_location("public_p2_adapter", RECIPE / "p2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_relocation_changes_only_path(tmp_path):
    original = {
        "train_path": "/old/location.jsonl",
        "train_sha256": "abc",
        "updates": [{"row_ids": ["a", "b"], "supervised_targets": 7}],
    }
    relocated = _adapter().relocate_plan(original, tmp_path / "train.jsonl")
    assert relocated["train_path"] == str(tmp_path / "train.jsonl")
    assert {k: v for k, v in relocated.items() if k != "train_path"} == {
        k: v for k, v in original.items() if k != "train_path"
    }
    assert original["train_path"] == "/old/location.jsonl"


def test_public_argv_retains_every_nonpath_recorded_argument(tmp_path):
    record = json.loads((ROOT / "configs/research-v1/p2_effective_launch.json").read_text())
    flags = [
        "--train_jsonl",
        "--val_jsonl",
        "--out_dir",
        "--tokenizer_path",
        "--init_from_pretrain",
        "--p2_consumption_plan",
    ]
    result = _adapter().make_argv(record, {flag: tmp_path / flag[2:] for flag in flags})
    original = record["argv"][4:]
    allowed = {original.index(flag) + 1 for flag in flags}
    assert len(result) == len(original)
    assert all(
        a == b for n, (a, b) in enumerate(zip(original, result, strict=True)) if n not in allowed
    )


def test_hash_mismatch_fails_closed(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Identity mismatch"):
        _adapter().check_identity(path, "0" * 64)


def test_validation_runs_outside_checkout_without_torch(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RECIPE / "validate.py"),
            "--synthetic-jsonl",
            str(RECIPE / "examples/messages.synthetic.jsonl"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["model_operations"] == 0
    assert report["historical_data_paths_checked"] is False
    assert report["synthetic_schema"]["rows"][0]["assistant_targets_including_eos"] == 3


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"messages": []}],
        [{"messages": [], "audit_id": "fake"}],
        [{"messages": [{"role": "user", "content": "No answer"}]}],
    ],
)
def test_synthetic_validation_rejects_bad_or_historical_shaped_rows(tmp_path, rows):
    path = tmp_path / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = subprocess.run(
        [sys.executable, str(RECIPE / "validate.py"), "--synthetic-jsonl", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Validation failed:" in result.stderr


def test_p2_missing_inputs_exits_before_importing_torch(tmp_path):
    # An import blocker makes accidental training dependencies a hard failure.
    script = """
import sys, runpy
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sft', 'src'}:
            raise AssertionError('unexpected training import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
sys.argv = [sys.argv[1], '--train-jsonl', 'missing', '--validation-jsonl', 'missing',
            '--base-checkpoint', 'missing', '--original-consumption-plan', 'missing',
            '--out-dir', 'new-output', '--validate-only']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(RECIPE / "p2.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "P2 validation failed:" in result.stderr
    assert "unexpected training import" not in result.stderr
    assert not (tmp_path / "new-output").exists()


def test_recovered_p2_plan_isolated_and_tampering_rejected(tmp_path):
    # Only parser/sampler/file operations, no model initialization or checkpoint load.
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from sft.p2_plan import derive_plan, load_checked_plan
from sft.train_sft import build_arg_parser, validate_sft_args
import src.chat_template, src.model
assert Path(src.chat_template.__file__).is_relative_to(Path(sys.argv[1]))
assert Path(src.model.__file__).is_relative_to(Path(sys.argv[1]))
record=json.loads(Path(sys.argv[2]).read_text())
args=build_arg_parser().parse_args(record['argv'][4:]);validate_sft_args(args)
assert args.preserve_messages and args.default_system is None
rows=[{'audit_id':str(i),'shifted_supervised_tokens':3} for i in range(64)]
train=Path('train.jsonl').resolve(); train.write_text(''.join(json.dumps(r)+'\\n' for r in rows))
plan=derive_plan(rows,train_path=train)
p=Path('plan.json');p.write_text(json.dumps(plan))
kw=dict(train_path=train,dataset_rows=64,seed=20260906,micro_bsz=2,grad_accum=16,max_steps=4)
assert load_checked_plan(p,**kw)==plan
plan['updates'][0]['row_ids'].reverse();p.write_text(json.dumps(plan))
try: load_checked_plan(p,**kw)
except ValueError: pass
else: raise AssertionError('tampered plan accepted')
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(RECIPE / "sources/posttraining"),
            str(ROOT / "configs/research-v1/p2_effective_launch.json"),
        ],
        cwd=tmp_path,
        check=True,
    )
