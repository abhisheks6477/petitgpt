"""Reader contracts on synthetic local bytes/rows and small CPU tensors only."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes/research-v1"
HEADER = """
import sys,json,copy
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch

def forbidden(*args,**kwargs):raise AssertionError('PROHIBITED_MODEL_OPERATION')
torch.nn.Module.__init__=forbidden
torch.optim.Optimizer.__init__=forbidden
torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
from adapter_common import HERE,SOURCE,TOKENIZER_SHA,read,write,sha
from reader_contracts import CONFIG,receipt,checkpoint,source
"""


def run(tmp_path, body):
    result = subprocess.run(
        [sys.executable, "-B", "-c", HEADER + body, str(R)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": "",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("stage", ["pretrain", "p2", "p3", "interpolate", "export", "evaluate"])
def test_stage_help_does_not_import_models(tmp_path, stage):
    code = """
import sys,runpy
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','safetensors','prepare','curriculum','transformers'):
   raise AssertionError('PROHIBITED_IMPORT:'+name)
sys.meta_path.insert(0,Block())
sys.argv=[str(Path(sys.argv[1])/'reader.py'),sys.argv[2],'--policy','new-run','--help']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(R), stage],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--validate-only" in result.stdout and "--execute" in result.stdout
    assert not list(tmp_path.iterdir())


def test_new_metadata_links_stages_and_historical_rejects_it(tmp_path):
    run(
        tmp_path,
        """
from argparse import Namespace
from reader_posttrain import run
p2=Path('new-p2.pt');p3=Path('new-p3.pt');p2.write_bytes(b'NEW synthetic opaque parent 1');p3.write_bytes(b'NEW synthetic opaque parent 2')
receipt(p2,'p2',750);receipt(p3,'p3',320)
a=Namespace(stage='interpolate',p2_parent=p2,p3_parent=p3,include_alpha050=False,
 tokenizer=HERE.parents[1]/'tokenizer/releases/tokenizer_v1/tokenizer.json',out_dir=Path('blend'),execute=False)
report=run(a)
assert report['alphas']==[.75] and report['execution']=='NOT_RUN' and not a.out_dir.exists()
from interpolate import validate
old=Namespace(p2_step750=p2,p3_step320=p3,tokenizer=a.tokenizer,out_dir=Path('old'))
try:validate(old)
except ValueError as e:assert 'Identity mismatch' in str(e)
else:raise AssertionError('historical parent check weakened')
child=Path('new-child.pt');child.write_bytes(b'NEW synthetic opaque child');receipt(child,'blend',0)
a=Namespace(stage='export',parent=child,tokenizer=a.tokenizer,out_dir=Path('export'),execute=False,
 frozen_inference_root=HERE/'runtime-support-v2/frozen_native')
report=run(a);assert report['execution']=='NOT_RUN' and not a.out_dir.exists()
meta=read(str(child)+'.reader.json');meta['config']['n_layers']=12;write(str(child)+'.reader.json',meta)
try:run(a)
except ValueError as e:assert 'architecture' in str(e)
else:raise AssertionError('wrong architecture accepted')
""",
    )


def test_checkpoint_schema_tokenizer_step_and_file_hash(tmp_path):
    run(
        tmp_path,
        """
p=Path('new.pt');p.write_bytes(b'opaque synthetic');original=receipt(p,'p3',320)
for mutation in ('hash','tokenizer','step','config'):
 meta=copy.deepcopy(original)
 if mutation=='hash':meta['sha256']='0'*64
 elif mutation=='tokenizer':meta['tokenizer_sha256']='0'*64
 elif mutation=='step':meta['step']=640
 else:meta['config']['d_model']=768
 write(str(p)+'.reader.json',meta)
 try:checkpoint(p,['p3'],320)
 except ValueError:pass
 else:raise AssertionError('bad metadata accepted:'+mutation)
""",
    )


def test_new_p2_data_plan_and_split_checks(tmp_path):
    run(
        tmp_path,
        """
torch.load=forbidden
from argparse import Namespace
from reader_posttrain import run
from src_placeholder import unused
""".replace(
            "from src_placeholder import unused",
            """
source()
from src.chat_template import load_chat_tokenizer,encode_chat
tokpath=HERE.parents[1]/'tokenizer/releases/tokenizer_v1/tokenizer.json';tok=load_chat_tokenizer(str(tokpath))
def data(prefix,n):
 records=[]
 for i in range(n):
  ms=[{'role':'user','content':f'{prefix} synthetic {i}'},{'role':'assistant','content':'OK'}]
  ids,labels=encode_chat(tok,ms,default_system=None)
  records.append({'audit_id':f'{prefix}-{i}','messages':ms,'shifted_supervised_tokens':sum(v!=-100 for v in labels[1:])})
 return records
def save(path,records):path.write_text(''.join(json.dumps(r)+'\\n' for r in records))
train=Path('train.jsonl');val=Path('val.jsonl');save(train,data('train',64));save(val,data('val',4))
base=Path('base.pt');base.write_bytes(b'NEW opaque base');receipt(base,'pretrain',49590)
a=Namespace(stage='p2',parent=base,train_jsonl=train,validation_jsonl=val,tokenizer=tokpath,
 plan=None,out_dir=Path('p2'),execute=False)
assert run(a)['updates']==4 and not a.out_dir.exists()
save(val,data('train',4))
try:run(a)
except ValueError as e:assert 'overlap' in str(e)
else:raise AssertionError('overlapping split accepted')
""",
        ),
    )


def test_full_p3_new_population_accepted_without_private_freeze(tmp_path):
    run(
        tmp_path,
        """
torch.load=forbidden
from argparse import Namespace
from reader_posttrain import run
source()
from src.chat_template import load_chat_tokenizer,encode_chat
tokpath=HERE.parents[1]/'tokenizer/releases/tokenizer_v1/tokenizer.json';tok=load_chat_tokenizer(str(tokpath))
def record(i,family):
 ms=[{'role':'user','content':f'Synthetic format row {family} {i}'},{'role':'assistant','content':'OK'}]
 _,labels=encode_chat(tok,ms,default_system=None)
 return {'audit_id':f'{family}-{i}','family':family,'messages':ms,'shifted_supervised_tokens':sum(v!=-100 for v in labels[1:])}
train=Path('train.jsonl');replay=Path('replay.jsonl')
train.write_text(''.join(json.dumps(record(i,f))+'\\n' for f in ['COPY','FIELD','MEMBERSHIP','JSON'] for i in range(1792)))
replay.write_text(''.join(json.dumps(record(i,'replay'))+'\\n' for i in range(3072)))
p=Path('new-p2.pt');p.write_bytes(b'NEW synthetic opaque P2');receipt(p,'p2',750)
a=Namespace(stage='p3',parent=p,train_jsonl=train,replay_jsonl=replay,tokenizer=tokpath,plan=None,out_dir=Path('p3'),execute=False)
assert run(a)['updates']==640 and not a.out_dir.exists()
assert not list(Path('.').glob('*FROZEN*'))
""",
    )


def test_packed_bytes_schema_and_pretrain_transition(tmp_path):
    run(
        tmp_path,
        """
import array
from reader_pretrain import packed,validate_transition
root=Path('packed');(root/'train').mkdir(parents=True)
p=root/'train/shard_00000.bin';p.write_bytes(array.array('H',[2,100,3]*1366).tobytes())
meta={'schema':'petitgpt-packed-new-v3','dtype':'uint16','split':'train','tokenizer_sha256':TOKENIZER_SHA,
 'shards':[{'name':p.name,'tokens':4098,'sha256':sha(p)}]}
write(root/'meta.json',meta);assert packed(root/'train','train')['blocks']==2
p.write_bytes(array.array('H',[32000]*4098).tobytes());meta['shards'][0]['sha256']=sha(p);write(root/'meta.json',meta)
try:packed(root/'train','train')
except ValueError as e:assert 'outside' in str(e)
else:raise AssertionError('invalid token IDs accepted')
a={'policy':'new-run','plan':{'identity':'synthetic'},'stage':'stage_a'}
b={**a,'stage':'stage_b'};validate_transition(a,b,38146)
for current,step in [(b,38145),({**b,'plan':{}},38146),({**b,'stage':'stage_a'},38147)]:
 if current==a:continue
 try:validate_transition(a,current,step)
 except RuntimeError:pass
 else:raise AssertionError('invalid handover accepted')
""",
    )


def test_synthetic_cpu_interpolation_and_native_serialization(tmp_path):
    run(
        tmp_path,
        """
from adapter_common import definitions
from interpolate import BLEND
from reader_tensors import save_native
ns={'torch':torch,'hashlib':__import__('hashlib')};definitions(BLEND,['interpolate','state_digest'],ns)
a=torch.tensor([-.75,2.,8.,-16.]);b=torch.tensor([.25,6.,4.,16.]);before=a.clone()
child=ns['interpolate'](a,b,.75)
assert torch.equal(child,torch.tensor([0.,5.,5.,8.])) and torch.equal(a,before)
assert child.data_ptr()!=a.data_ptr() and child.data_ptr()!=b.data_ptr()
sd={'lm_head.weight':child,'tok_emb.weight':child,'fixed':torch.tensor([1],dtype=torch.int64)}
out=Path('toy.safetensors');restored=save_native(sd,out,{'tok_emb.weight':'lm_head.weight'})
assert restored['tok_emb.weight'] is restored['lm_head.weight']
assert ns['state_digest'](sd)==ns['state_digest'](restored)
p=Path('toy.pt');torch.save({'model':sd},p);again=torch.load(p,map_location='cpu',weights_only=True)
assert torch.equal(again['model']['fixed'],sd['fixed'])
bad=dict(sd);bad['tok_emb.weight']=child+1
try:save_native(bad,Path('bad.safetensors'),{'tok_emb.weight':'lm_head.weight'})
except ValueError:pass
else:raise AssertionError('bad tied values accepted')
print('SYNTHETIC_CPU_TENSOR_TESTS=PASS; tiny 4-element arithmetic and file round trips only')
""",
    )


def test_local_task_inputs_and_likelihood_alignment_helpers(tmp_path):
    run(
        tmp_path,
        """
from reader_evaluate import inputs,BENCH
from reader_contracts import module,tokenizer
p=Path('piqa.jsonl');p.write_text(json.dumps({'goal':'Synthetic only','sol1':'é','sol2':'abcd','label':0})+'\\n')
m=Path('tasks.json');write(m,{'schema':'petitgpt-tasks-new-v3','scope':'synthetic',
 'tasks':[{'task':'piqa','revision':'synthetic-v1','split':'test','path':str(p),'rows':1,'sha256':sha(p)}]})
tok=tokenizer(HERE.parents[1]/'tokenizer/releases/tokenizer_v1/tokenizer.json')
scope,identity,requests=inputs(m,tok);assert scope=='synthetic' and requests[0]['prompt']=='Question: Synthetic only\\nAnswer:'
helper=module(BENCH/'runtime/likelihood_checks.py','toy_likelihood')
# Predetermined toy scores/logits test alignment and Unicode denominators; no model/task inference.
assert helper.metrics([-2.,-3.],['é','abcd'],0)['prediction_acc_norm']==1
logits=torch.arange(24,dtype=torch.float32).reshape(1,6,4)/10
v=helper.continuation_logprobs(logits,2,[1,2])
assert torch.equal(v,logits[0,1:3].log_softmax(-1)[torch.arange(2),torch.tensor([1,2])])
assert helper.metrics([-1.,-1.],['a','b'],0)['prediction_acc']==0
""",
    )


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_recovered_pretrain_bindings_and_strict_resume_without_model_work(tmp_path, stage):
    body = """
from argparse import Namespace
from reader_pretrain import bind_trainer,ENDS,STARTS,SEEDS
stage=STAGE
args=Namespace(pretrain_stage=stage,stage_a_dir=Path('A/train'),stage_b_dir=Path('B/train'),
 validation_dir=Path('reference/val'),out_dir=Path('output'),tokenizer=Path('tokenizer.json'),resume=None)
plan={'synthetic_metadata_only':True}
trainer,parsed=bind_trainer(args,plan)
assert parsed.strict_resume_contract and parsed.micro_bsz==8 and parsed.grad_accum==16
assert parsed.schedule_total_steps==49590 and parsed.max_steps==ENDS[stage]
assert parsed.data_stage_start_step==STARTS[stage] and parsed.sampler_seed==SEEDS[stage]
assert not parsed.launch_contract_json and not parsed.stage_authorization_json
assert not parsed.eval_steps and parsed.eval_every==0
assert trainer.GPT.__module__=='src.model'
assert Path(sys.modules['src.model'].__file__).is_relative_to(HERE/('sources/pretrain_'+stage))
# The actual source resume checker runs after the new A/B plan checker. All
# comparisons below use synthetic metadata; no model/optimizer is constructed.
old={'policy':'new-run','plan':plan,'stage':'stage_a'}
new={**old,'stage':'stage_b'}
policy={'stage_a_sampler_seed':20260832,'stage_b_sampler_seed':20260833}
saved={'run_plan':old,'sampler_seed':20260832,'governed_policy':policy,'schedule':{'horizon':49590}}
current={**saved,'run_plan':new,'sampler_seed':20260833}
kw={'strict':True,'checkpoint_step':38146,'allow_schedule_branch':False}
trainer.validate_resume_contract({'run_contract':saved},current,**kw)
bad={**current,'schedule':{'horizon':320}}
try:trainer.validate_resume_contract({'run_contract':saved},bad,**kw)
except RuntimeError:pass
else:raise AssertionError('schedule mismatch bypassed')
assert not args.out_dir.exists()
""".replace("stage=STAGE", f"stage={stage!r}")
    run(tmp_path, body)
