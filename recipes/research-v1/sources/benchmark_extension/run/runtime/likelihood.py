"""Task-local driver reusing the reviewed FP32 forward/scoring methods verbatim."""
import os,sys,json,hashlib,ast,importlib.util,time,gc,math,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R.parent/'source';B=S/'recipes/research-v1/sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2'
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',HF_HOME=str(R/'cache/hf'))
sys.path.insert(0,str(S/'recipes/research-v1/sources/native_inference'))
import torch,numpy as np
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from src.model import GPT,gpt_config_from_checkpoint_dict

def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads((R/p).read_text())
def lines(p):
 with (R/p).open() as f:
  for l in f:yield json.loads(l)
def save(p,x):
 p=R/p;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def extract(p,cls,names,ns):
 body=ast.parse(p.read_text()).body
 if cls:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==cls).body
 nodes=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
 return {n:ns[n] for n in names}
ns={'torch':torch,'np':np,'sdpa_kernel':sdpa_kernel,'SDPBackend':SDPBackend}
extract(B/'runtime/common.py',None,['primary_policy','metric'],ns)
methods=extract(B/'runtime/common.py','Scorer',['_model_call','score'],ns)
load_ns={'torch':torch,'np':np};load_ckpt=extract(S/'recipes/research-v1/sources/benchmark/sft/train_sft.py',None,['load_ckpt'],load_ns)['load_ckpt']
metric=ns['metric']
class Scorer:
 _model_call=methods['_model_call'];score=methods['score']
 def __init__(self,m):
  self.identity=m;self.forward_calls=0
  if m['id'].startswith('Petit'):
   ck=load_ckpt(m['path']);cfg=gpt_config_from_checkpoint_dict(ck.get('config') or ck.get('cfg'))
   for k,v in m['expected_config'].items():assert getattr(cfg,k)==v,(k,getattr(cfg,k),v)
   self.model=GPT(cfg);self.model.load_state_dict(ck['model'],strict=True);del ck
   assert self.model.tok_emb.weight is self.model.lm_head.weight
  else:
   self.model=AutoModelForCausalLM.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='sdpa')
   assert self.model.config._attn_implementation=='sdpa'
   assert self.model.get_input_embeddings().weight is self.model.get_output_embeddings().weight
  self.model.requires_grad_(False).eval().to('cuda')
  assert sum(p.numel() for p in self.model.parameters())==m['expected_unique_parameters']
  assert all(p.dtype==torch.float32 for p in self.model.parameters())
  assert all(not p.requires_grad for p in self.model.parameters()) and not self.model.training

def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['smoke','full']);args=a.parse_args()
 models=read('protocol/MODEL_IDENTITIES.json');pre=read('protocol/LIKELIHOOD_PREFLIGHT.json');data=read('protocol/DATA_IDENTITIES.json')
 # Validate all three model identities before any execution; no score-dependent fallback.
 for m in models:
  for f in m['verified_files']:assert sha(f['path'])==f['sha256'],f['path']
 for task in ['arc_challenge','hellaswag']:
  assert sha(R/task/'normalized_rows.jsonl')==data[task]['normalized_rows_sha256']
  for m in models:assert sha(R/pre[m['id']][task]['file'])==pre[m['id']][task]['sha256']
 source_paths=[B/'runtime/common.py',B/'runtime/likelihood_checks.py',S/'recipes/research-v1/sources/benchmark/sft/train_sft.py',S/'recipes/research-v1/sources/native_inference/src/model.py',S/'recipes/research-v1/sources/native_inference/src/special_tokens.py',B/'source/harness/lm_eval/api/model.py',Path(__file__),R/'runtime/prepare.py',R/'upstream/lm_eval/tasks/hellaswag/utils.py']
 bindings={'source_files':{str(p):sha(p) for p in source_paths},'protocol_files':{p:sha(R/p) for p in ['protocol/MODEL_IDENTITIES.json','protocol/LIKELIHOOD_PREFLIGHT.json','protocol/DATA_IDENTITIES.json','protocol/LIKELIHOOD_PROTOCOL.json','protocol/ENVIRONMENT.json']}}
 frozen=digest(canon(bindings))
 if (R/'protocol/LIKELIHOOD_CODE_FREEZE.json').exists():assert read('protocol/LIKELIHOOD_CODE_FREEZE.json')['fingerprint']==frozen
 else:save('protocol/LIKELIHOOD_CODE_FREEZE.json',{'fingerprint':frozen,**bindings,'unchanged_AST_methods':['Scorer._model_call','Scorer.score','primary_policy','metric','load_ckpt'],'driver_adaptations':'only task-specific inputs, identity guards, portable loading and incremental output; numeric scorer unchanged'})
 if args.stage=='full':assert read('smoke/LIKELIHOOD.json')['status']=='PASS' and read('smoke/LIKELIHOOD.json')['fingerprint']==frozen
 summary=[]
 for m in models:
  mid=m['id'];print('LOAD',args.stage,mid,flush=True);scorer=Scorer(m)
  for task in ['arc_challenge','hellaswag']:
   normalized=list(lines(f'{task}/normalized_rows.jsonl'));encoded=list(lines(pre[mid][task]['file']));n=8 if args.stage=='smoke' else len(normalized)
   out=f'smoke/{task}_{mid}.jsonl' if args.stage=='smoke' else f'{task}/rows_{mid}.jsonl'
   target=R/out;completed=[];start=time.monotonic()
   if target.exists():
    completed=list(lines(out))
    for i,x in enumerate(completed):
     assert x['fingerprint']==frozen and x['id']==normalized[i]['id'] and x['row_sha256']==digest(canon(normalized[i]))
     assert x['choices']==normalized[i]['choices'] and len(x['scores'])==len(x['choices']) and all(math.isfinite(v) for v in x['scores'])
     assert x['candidate_chars']==[len(t) for t in x['choices']]
     z=metric(x['scores'],x['candidate_chars'],x['gold']);assert all(x[k]==v for k,v in z.items())
    assert len(completed)<=n
    if len(completed)<n:save(f'{task}/RESUME_{mid}.json',{'completed_rows':len(completed),'fingerprint':frozen})
   with target.open('a') as f:
    for i in range(len(completed),n):
     d=normalized[i];e=encoded[i];assert e['id']==d['id'] and e['row_sha256']==digest(canon(d));scores=[]
     for p in e['pairs']:scores.append(scorer.score(p['context_ids'],p['continuation_ids']))
     denoms=[len(t) for t in d['choices']];assert denoms==[p['candidate_chars'] for p in e['pairs']]
     record={'model':mid,'task':task,**d,'row_sha256':e['row_sha256'],'scores':scores,'candidate_chars':denoms,'continuation_tokens':[len(p['continuation_ids']) for p in e['pairs']],'fingerprint':frozen,**metric(scores,denoms,d['gold'])}
     assert len(scores)==len(d['choices']) and all(math.isfinite(v) for v in scores)
     f.write(canon(record)+'\n');f.flush();completed.append(record)
     if (i+1)%100==0:
      os.fsync(f.fileno());save('runtime/PROGRESS.json',{'stage':args.stage,'model':mid,'task':task,'completed':i+1,'total':n,'seconds_this_session':time.monotonic()-start});print('PROGRESS',mid,task,i+1,n,round(time.monotonic()-start,1),flush=True)
    os.fsync(f.fileno())
   result={'model':mid,'task':task,'rows':n,'acc_correct':sum(d['acc'] for d in completed),'acc_norm_correct':sum(d['acc_norm'] for d in completed),'candidate_sequences':sum(len(d['choices']) for d in completed),'rows_file':out,'rows_sha256':sha(target),'fingerprint':frozen,'status':'PASS','seconds_this_session':time.monotonic()-start}
   result.update(acc=result['acc_correct']/n,acc_norm=result['acc_norm_correct']/n);summary.append(result)
   save(f'smoke/SUMMARY_{task}_{mid}.json' if args.stage=='smoke' else f'{task}/RESULTS_{mid}.json',result);print('TASK COMPLETE',mid,task,n,flush=True)
  del scorer;gc.collect();torch.cuda.empty_cache()
 save('smoke/LIKELIHOOD.json' if args.stage=='smoke' else 'protocol/LIKELIHOOD_COMPLETION.json',{'status':'PASS','fingerprint':frozen,'stage':args.stage,'results':summary})
 print('STAGE COMPLETE',args.stage,flush=True)
if __name__=='__main__':main()
