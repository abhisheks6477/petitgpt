"""Outside-checkout, model-free contract tests. Synthetic rows are never historical data."""

import ast
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/research-v1"
GUARD = r"""
import sys, runpy
from pathlib import Path
recipe = Path(sys.argv[1])
sys.path.insert(0, str(recipe))
class BlockModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'safetensors', 'p3_execution', 'prepare', 'curriculum', 'checkers'}:
            raise AssertionError('PROHIBITED_IMPORT:' + fullname)
sys.meta_path.insert(0, BlockModelImports())
"""


def run(tmp_path, body, *args, expected=0):
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": "",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", GUARD + body, str(RECIPE), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert "PROHIBITED_IMPORT" not in result.stderr
    return result


@pytest.mark.parametrize("script", ["p2.py", "p3.py", "interpolate.py", "export_native.py"])
def test_help_has_no_transitive_model_import(tmp_path, script):
    result = run(
        tmp_path,
        "sys.argv=[str(recipe/sys.argv[2]), '--help'];runpy.run_path(sys.argv[0],run_name='__main__')",
        script,
    )
    assert "--validate-only" in result.stdout and "--execute" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "script,flags",
    [
        (
            "p3.py",
            [
                "--mode",
                "training-only",
                "--parent-checkpoint",
                "missing",
                "--tokenizer",
                "missing",
                "--train-jsonl",
                "missing",
                "--replay-jsonl",
                "missing",
                "--update-plan",
                "missing",
                "--freeze-json",
                "missing",
            ],
        ),
        (
            "interpolate.py",
            ["--p2-step750", "missing", "--p3-step320", "missing", "--tokenizer", "missing"],
        ),
        (
            "export_native.py",
            [
                "--source-checkpoint",
                "missing",
                "--frozen-inference-root",
                "missing",
                "--tokenizer",
                "missing",
            ],
        ),
    ],
)
def test_validate_missing_inputs_never_imports_models_or_creates_output(tmp_path, script, flags):
    result = run(
        tmp_path,
        "sys.argv=[str(recipe/sys.argv[2]), *sys.argv[3:]];runpy.run_path(sys.argv[0],run_name='__main__')",
        script,
        *flags,
        "--out-dir",
        "fresh-output",
        "--validate-only",
        expected=1,
    )
    assert "Adapter failed:" in result.stderr
    assert not (tmp_path / "fresh-output").exists()


def test_frozen_source_and_new_relocation_map_hashes(tmp_path):
    run(tmp_path, "from adapter_common import verify_sources; verify_sources()")


def test_transitive_wrong_source_rejected_even_without_parent_package(tmp_path):
    run(
        tmp_path,
        """
from types import ModuleType
from adapter_common import activate_source
bad=ModuleType('src.model');bad.__file__='/wrong/src/model.py';sys.modules['src.model']=bad
try:activate_source()
except RuntimeError as exc: assert 'Wrong source import: src.model' in str(exc)
else:raise AssertionError('poisoned transitive import accepted')
""",
    )


def test_real_tokenizer_and_synthetic_masking_schema(tmp_path):
    run(
        tmp_path,
        """
from p3 import validate_records
from adapter_common import activate_source, SOURCE
activate_source()
from src.chat_template import load_chat_tokenizer, encode_chat
import src.chat_template, src.special_tokens
for name,m in sys.modules.items():
 if name=='src' or name.startswith('src.'):
  assert Path(m.__file__).resolve().is_relative_to(SOURCE)
tok=load_chat_tokenizer(str(recipe.parents[1]/'tokenizer/releases/tokenizer_v1/tokenizer.json'))
messages=[{'role':'user','content':'Hi [EOS]'}, {'role':'assistant','content':'Hello'},
          {'role':'user','content':'Again'}, {'role':'assistant','content':'OK'}]
ids, labels=encode_chat(tok,messages,default_system=None)
row={'audit_id':'synthetic-only','messages':messages,'formatted_tokens':len(ids),
     'shifted_targets':sum(t!=-100 for t in labels[1:]),
     'input_unsupervised_tokens':len(ids)-sum(t!=-100 for t in labels[1:])}
schema={'rows':1,'observed_top_level_schema':{k:{'present_rows':1,'types':[type(v).__name__]} for k,v in row.items()}}
assert validate_records([row],schema,tok,encode_chat)=={'synthetic-only'}
row['shifted_targets']+=1
try:validate_records([row],schema,tok,encode_chat)
except ValueError as exc:assert 'Encoding differs' in str(exc)
else:raise AssertionError('bad target count accepted')
assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
""",
    )


def test_full_synthetic_plan_order_exposures_and_tampering(tmp_path):
    # 10,240 IDs only; no procedural examples or private content are constructed.
    run(
        tmp_path,
        """
import random,copy
from adapter_common import definitions,SUPPORT
from p3 import validate_plan
families=('COPY','FIELD','MEMBERSHIP','JSON')
proc=[{'audit_id':f'synthetic-{f}-{i}','family':f} for f in families for i in range(1792)]
replay=[{'audit_id':f'synthetic-replay-{i}'} for i in range(3072)]
ns={'random':random,'SEED':20260907,'FAMILIES':families}
definitions(SUPPORT/'helpers/p3_plan.py',['build_plan'],ns)
plan=ns['build_plan'](proc,replay);validate_plan(proc,replay,plan)
assert len(plan)==640
for variant in ['reverse','stream','boolean-update','duplicate','omit']:
 bad=copy.deepcopy(plan)
 if variant=='reverse':bad[0]['row_ids'].reverse()
 elif variant=='stream':bad[0]['stream']='wrong'
 elif variant=='boolean-update':bad[0]['update']=True
 elif variant=='duplicate':bad[0]['row_ids'][0]=bad[0]['row_ids'][1]
 else:bad.pop()
 try:validate_plan(proc,replay,bad)
 except ValueError:pass
 else:raise AssertionError('tampered plan accepted:'+variant)
""",
    )


def test_historical_mode_requires_full_private_bindings(tmp_path):
    run(
        tmp_path,
        """
from argparse import Namespace
from p3 import historical_inputs
try:historical_inputs(Namespace(historical_bindings=None),{})
except ValueError as exc:assert 'old baseline markers cannot substitute' in str(exc)
else:raise AssertionError('missing private bindings accepted')
""",
    )


def test_export_real_assets_and_synthetic_opaque_receipt(tmp_path):
    # Positive metadata-only path: bytes are expressly synthetic; declared tensor
    # digests are not validated as tensors until future execution.
    run(
        tmp_path,
        """
import json,copy
from argparse import Namespace
from adapter_common import SUPPORT,read,sha
from export_native import validate
contract=read(SUPPORT/'contracts/INTERPOLATION_EXPORT_IDENTITIES.json')
checkpoint=Path('synthetic-opaque.pt');checkpoint.write_bytes(b'NOT A CHECKPOINT')
artifacts=copy.deepcopy(contract['fixed_candidates']);artifacts[1]['sha256']=sha(checkpoint)
receipt={'schema':'FIXED_INTERPOLATION_OUTPUT_V2','parents':contract['parents'],
         'tokenizer_sha256':'d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce',
         'artifacts':artifacts}
p=Path('synthetic-receipt.json');p.write_text(json.dumps(receipt))
args=Namespace(source_checkpoint=checkpoint,interpolation_receipt=p,
 frozen_inference_root=SUPPORT/'frozen_native',
 tokenizer=recipe.parents[1]/'tokenizer/releases/tokenizer_v1/tokenizer.json',out_dir=Path('output'))
assets,evidence=validate(args)
assert len(assets)==10 and evidence['tensor_equality_established'] is False
assert evidence['new_export_sha256'] is None and not args.out_dir.exists()
checkpoint.write_bytes(b'TAMPERED')
try:validate(args)
except ValueError as exc:assert 'Identity mismatch' in str(exc)
else:raise AssertionError('opaque hash mismatch accepted')
""",
    )


def test_fresh_directory_rejects_dangling_symlink(tmp_path):
    run(
        tmp_path,
        """
from adapter_common import fresh
p=Path('dangling');p.symlink_to('absent')
try:fresh(p)
except ValueError:pass
else:raise AssertionError('dangling symlink accepted')
""",
    )


def test_execution_function_bindings_are_complete_without_importing_models(tmp_path):
    # Analyze global name loads with symtable, compile the selected definitions
    # using inert decorators only, and verify all required globals are bound by
    # the explicit execution adapters. No function body/model is executed.
    run(
        tmp_path,
        """
import ast,symtable
from adapter_common import definitions,SOURCE,SUPPORT
class Decorators:
 def inference_mode(self):return lambda f:f
p3=SOURCE/'runs/p3_basic_instruction_generalization_20260907/runtime/execute.py'
names=['load_model','encoded_rows','checkpoint','train','generate','agg','evaluate','val500','baseline']
ns={'torch':Decorators()};definitions(p3,names,ns)
assert all(callable(ns[n]) for n in names)
execution=ast.parse((recipe/'p3_execution.py').read_text())
phase=next(n for n in execution.body if isinstance(n,ast.FunctionDef) and n.name=='phase')
# Collect named keyword globals from the explicit ns=dict(...) binding.
bound={kw.arg for n in ast.walk(phase) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
       and n.func.id=='dict' for kw in n.keywords}
bound.update(names);bound.add('check')
import builtins
symbols=symtable.symtable(p3.read_text(),str(p3),'exec')
for table in symbols.get_children():
 if table.get_name() in names:
  missing={s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
  assert not missing-bound-set(dir(builtins)),(table.get_name(),missing-bound-set(dir(builtins)))
blend=SOURCE/'runs/p3_retention_two_point_interpolation_20260907/runtime/blend.py'
definitions(blend,['interpolate','state_digest','main'],{})
""",
    )


def test_p3_training_loop_ast_preserved_except_explicit_training_only_gate():
    source = (
        RECIPE
        / "sources/posttraining/runs/p3_basic_instruction_generalization_20260907/runtime/execute.py"
    )
    original = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    # Execute only the AST transform function, avoiding the execution module import.
    tree = ast.parse((RECIPE / "p3_execution.py").read_text())
    transform = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "training_only_gate"
    )
    ns = {"ast": ast}
    exec(compile(ast.Module(body=[transform], type_ignores=[]), "<transform>", "exec"), ns)
    changed = ns["training_only_gate"](copy.deepcopy(original))
    assert ast.dump(changed.body[0]) == ast.dump(original.body[1])
    assert [ast.dump(n) for n in changed.body] == [ast.dump(n) for n in original.body[1:]]
    original.body[0] = ast.Pass()
    with pytest.raises(ValueError, match="baseline gate shape"):
        ns["training_only_gate"](original)


def test_p2_transitive_parser_imports_are_pinned_and_model_work_guarded(tmp_path):
    script = r"""
import sys,json
from pathlib import Path
import torch
calls=[]
def forbidden(*a,**kw):
 calls.append('prohibited');raise AssertionError('PROHIBITED_MODEL_OPERATION')
torch.nn.Module.__init__=forbidden
torch.optim.Optimizer.__init__=forbidden
torch.load=forbidden
torch.cuda._lazy_init=forbidden
recipe=Path(sys.argv[1]);sys.path.insert(0,str(recipe))
import p2
p2.activate_source()
from sft.train_sft import build_arg_parser,validate_sft_args,preflight_sft_record
from src.chat_template import load_chat_tokenizer
record=json.loads(p2.RECORD.read_text())
args=build_arg_parser().parse_args(record['argv'][4:]);validate_sft_args(args)
tok=load_chat_tokenizer(str(p2.TOKENIZER))
row={'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'}]}
result=preflight_sft_record('synthetic-only',row,tok=tok,seq_len=2048,default_system=None)
assert result['supervised_tokens']>0
for name,m in sys.modules.items():
 if name in ('src','sft') or name.startswith(('src.','sft.')):
  assert Path(m.__file__).resolve().is_relative_to(p2.SOURCE),(name,m.__file__)
assert not calls
print('P2 parser/preflight transitive source imports pinned; guarded model operations=0; synthetic only')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(RECIPE)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "guarded model operations=0" in result.stdout
    assert list(tmp_path.iterdir()) == []
