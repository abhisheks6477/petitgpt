"""Restricted reader loading on self-authored tiny CPU payloads, never real weights.

Alternate pickle GLOBAL spellings simulate serialized names, not a NumPy 2.x
runtime. Subprocesses prevent a prior submodule import from masking the lazy-stub
failure. One explicitly isolated test bypasses tensor schema only to reach step
validation; other loader/config/state checks run without validator replacements.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

RECIPE = Path(__file__).resolve().parents[1] / "recipes/research-v1"
HEADER = r"""
import io,sys,pickle,zipfile,importlib
from pathlib import Path
from unittest import mock
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch

def forbidden(*args,**kwargs):raise AssertionError('PROHIBITED_MODEL_OPERATION')
torch.nn.Module.__init__=forbidden
torch.optim.Optimizer.__init__=forbidden
torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
import reader_contracts as c

def payload(value):
 f=io.BytesIO();torch.save(value,f,pickle_protocol=2);f.seek(0);return f
"""


def run(tmp_path, body):
    result = subprocess.run(
        [sys.executable, "-B", "-c", HEADER + body, str(RECIPE)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_process_lazy_parent_and_tensor_only_cpu_weights_only(tmp_path):
    run(
        tmp_path,
        r"""
stub=importlib.import_module('numpy._core')
# On the actual 1.26.4 laptop this is the unmodified compatibility stub.
# Do not mutate modules to manufacture the missing-attribute condition.
if np.__version__=='1.26.4':
 assert 'multiarray' not in vars(stub)
 assert 'numpy._core.multiarray' not in sys.modules
 original_failure=False
 try:stub.multiarray._reconstruct
 except AttributeError:original_failure=True
 assert original_failure
before=set(torch.serialization.get_safe_globals())
f=payload({'tiny':torch.tensor([1.,2.])})
with mock.patch.object(torch,'load',wraps=torch.load) as observed:
 result=c.restricted_load(f)
 assert observed.call_count==1
 assert observed.call_args.args==(f,)
 assert observed.call_args.kwargs=={'map_location':'cpu','weights_only':True}
assert result['tiny'].device.type=='cpu'
assert torch.equal(result['tiny'],torch.tensor([1.,2.]))
assert set(torch.serialization.get_safe_globals())==before
assert c.numpy_reconstruct() is importlib.import_module('numpy.core.multiarray')._reconstruct
""",
    )


@pytest.mark.parametrize("name", ["numpy.core.multiarray", "numpy._core.multiarray"])
def test_uint32_rng_both_exact_serialized_global_names(tmp_path, name):
    run(
        tmp_path,
        "NAME="
        + repr(name)
        + "\n"
        + r"""
# Only bytes generated here are changed. No real checkpoint or module mutation.
rng=np.random.RandomState(7).get_state()
original=payload({'tiny':torch.tensor([3]),'rng':rng})
rewritten=io.BytesIO()
with zipfile.ZipFile(original) as source,zipfile.ZipFile(rewritten,'w') as target:
 found=0
 for info in source.infolist():
  data=source.read(info.filename)
  if info.filename.endswith('/data.pkl'):
   known=[b'cnumpy.core.multiarray\n_reconstruct\n',b'cnumpy._core.multiarray\n_reconstruct\n']
   assert sum(data.count(k) for k in known)==1
   for old in known:
    if old in data:
     data=data.replace(old,('c'+NAME+'\n_reconstruct\n').encode());found+=1;break
  target.writestr(info,data)
 assert found==1
rewritten.seek(0)
assert NAME+'._reconstruct' in torch.serialization.get_unsafe_globals_in_checkpoint(rewritten)
rewritten.seek(0)
class ExistingCallerRole:pass
# Overlap ndarray, but leave both reconstruction names for the helper to authorize.
with torch.serialization.safe_globals([ExistingCallerRole,np.ndarray]):
 before=set(torch.serialization.get_safe_globals())
 restored=c.restricted_load(rewritten)
 assert set(torch.serialization.get_safe_globals())==before
assert restored['rng'][0]==rng[0] and restored['rng'][2:]==rng[2:]
assert restored['rng'][1].dtype==np.uint32 and np.array_equal(restored['rng'][1],rng[1])
assert restored['tiny'].device.type=='cpu' and torch.equal(restored['tiny'],torch.tensor([3]))
""",
    )


def test_unrelated_class_rejected_and_allowlist_restored_after_failure(tmp_path):
    run(
        tmp_path,
        r"""
class Unrelated:pass
class ExistingCallerRole:pass
baseline=set(torch.serialization.get_safe_globals())
f=payload({'unrelated':Unrelated()})
with torch.serialization.safe_globals([ExistingCallerRole,np.dtype]):
 before=set(torch.serialization.get_safe_globals())
 with mock.patch.object(torch,'load',wraps=torch.load) as observed:
  try:c.restricted_load(f)
  except pickle.UnpicklingError as exc:assert 'Unrelated' in str(exc)
  else:raise AssertionError('unrelated object authorized')
  assert observed.call_count==1
  assert observed.call_args.kwargs=={'map_location':'cpu','weights_only':True}
 assert set(torch.serialization.get_safe_globals())==before
assert set(torch.serialization.get_safe_globals())==baseline
""",
    )


@pytest.mark.parametrize("missing", ["numpy._core", "numpy._core.multiarray"])
def test_resolver_falls_back_only_for_missing_compat_namespace(tmp_path, missing):
    run(
        tmp_path,
        "MISSING="
        + repr(missing)
        + "\n"
        + r"""
# Simulate only import availability; the returned reconstruction function is real.
real_import=importlib.import_module
calls=[]
def resolve(name):
 calls.append(name)
 if name=='numpy._core.multiarray':raise ModuleNotFoundError('missing compatibility namespace',name=MISSING)
 return real_import(name)
with mock.patch.object(c.importlib,'import_module',side_effect=resolve):
 found=c.numpy_reconstruct()
assert found is real_import('numpy.core.multiarray')._reconstruct
assert calls==['numpy._core.multiarray','numpy.core.multiarray']
""",
    )


@pytest.mark.parametrize("failure", ["native_missing", "native_import", "missing_attribute"])
def test_resolver_does_not_hide_other_import_or_attribute_errors(tmp_path, failure):
    run(
        tmp_path,
        "FAILURE="
        + repr(failure)
        + "\n"
        + r"""
from types import SimpleNamespace
if FAILURE=='native_missing':error=ModuleNotFoundError('native dependency missing',name='numpy._core._multiarray_umath')
elif FAILURE=='native_import':error=ImportError('native library ABI failure')
else:error=AttributeError('missing reconstruction function')
def resolve(name):
 if FAILURE=='missing_attribute':return SimpleNamespace()
 raise error
with mock.patch.object(c.importlib,'import_module',side_effect=resolve) as observed:
 try:c.numpy_reconstruct()
 except type(error):pass
 else:raise AssertionError('unrelated import error hidden')
 assert observed.call_count==1
""",
    )


@pytest.mark.parametrize("invalid", ["config", "state_keys", "state_shape"])
def test_actual_raw_loader_retains_schema_rejection_on_tiny_files(tmp_path, invalid):
    run(
        tmp_path,
        "INVALID="
        + repr(invalid)
        + "\n"
        + r"""
cfg=dict(c.CONFIG)
sd={}
if INVALID=='config':cfg['n_layers']=1
if INVALID=='state_shape':sd={k:torch.empty(0) for k in c.shapes()}
p=Path('synthetic-invalid.pt');torch.save({'model':sd,'config':cfg},p)
expected={'config':'architecture/config','state_keys':'state keys','state_shape':'shape/dtype/finite'}[INVALID]
try:c.raw_load(p)
except ValueError as exc:assert expected in str(exc)
else:raise AssertionError('schema rejection bypassed')
""",
    )


def test_actual_step_rejection_with_explicitly_isolated_tensor_schema(tmp_path):
    run(
        tmp_path,
        r"""
p=Path('synthetic-step.pt');torch.save({'config':c.CONFIG,'model':{},'step':1},p)
c.receipt(p,'blend',0)
# Only isolate tensor schema to reach loaded-step validation without a full model.
# Deserialization, metadata/file hash, config and loaded-step checks stay real.
with mock.patch.object(c,'state',return_value={}) as isolated:
 try:c.load(p)
 except ValueError as exc:assert 'Loaded checkpoint step differs' in str(exc)
 else:raise AssertionError('loaded step check bypassed')
 isolated.assert_called_once_with({})
""",
    )
