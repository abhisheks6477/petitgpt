"""Frozen canonical-chat IFEval generation; inference and verifier environments separate."""
from pathlib import Path
import os,sys,json,hashlib,time,gc,subprocess,argparse,collections
V=Path(__file__).resolve().parents[1];R=V.parents[1];S=R.parent/'source'
sys.path.insert(0,str(R/'runtime'))
from effective_config import prepare,invoke,identity
import fcntl,traceback

# This module imports no evaluator-only packages.
from likelihood import Scorer,sha,canon,digest,read,lines,save
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer
from src.chat_template import load_chat_tokenizer,encode_prompt
from src.accepted_generate import generate as native_generate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full']);a=ap.parse_args()
 lock=(V/'EVALUATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 approval=read('protocol/IFEVAL_TEMPLATE_RESOLUTION.json');assert approval['retain_canonical_template_system'] is True
 models=read('protocol/MODEL_IDENTITIES.json');protocol=read('protocol/IFEVAL_PROTOCOL.json');assert protocol['full_generation_authorized_by_preflight']
 assert read('ifeval/verifier/FIXTURES.json')['status']=='PASS'
 for m in models:
  for f in m['verified_files']:assert sha(f['path'])==f['sha256']
 inputs={m['id']:list(lines(f"ifeval/generation_inputs_{m['id']}.jsonl")) for m in models}
 for mid,rr in inputs.items():
  assert len(rr)==541 and all(r['fits'] for r in rr)
  assert sha(R/f'ifeval/generation_inputs_{mid}.jsonl')==protocol['context_preflight'][mid]['sha256']
 assert sha(R/'ifeval/normalized_rows.jsonl')==read('protocol/DATA_IDENTITIES.json')['ifeval']['normalized_rows_sha256']
 docs=list(lines('ifeval/normalized_rows.jsonl'));selection=docs[:8];keys=[d['key'] for d in selection]
 choice={'rule':'first 8 official rows in original order; frozen before any generation','keys':keys,'instruction_types':sorted(set(t for d in selection for t in d['instruction_id_list']))}
 assert len(choice['instruction_types'])>=4
 if (R/'protocol/IFEVAL_SMOKE_SELECTION.json').exists():assert read('protocol/IFEVAL_SMOKE_SELECTION.json')==choice
 else:save('protocol/IFEVAL_SMOKE_SELECTION.json',choice)
 freeze=read('ifeval/completion_v2/CODE_FREEZE.json');fp=freeze['fingerprint']
 assert digest(canon(freeze['files']))==fp
 for path,expected in freeze['files'].items():assert sha(path)==expected,path
 assert read('ifeval/completion_v2/CONFIG_ONLY_REGRESSIONS.json')['status']=='PASS'
 bridge=read('ifeval/completion_v2/PETITGPT_SMOKE_BRIDGE.json')
 assert bridge['status']=='REUSED_VERIFIED' and bridge['new_fingerprint']==fp
 for path,expected in bridge['evidence_files'].items():assert sha(path)==expected,path
 if a.stage=='full':
  gate=read('ifeval/completion_v2/smoke/IFEVAL.json');assert gate['status']=='PASS' and gate['continuation_fingerprint']==fp
  for receipt in gate['execution_identities']:
   for path,expected in receipt['evidence_files'].items():assert sha(path)==expected,path
 reports=[read('smoke/IFEVAL_SCORED_PetitGPT-alpha075.json')] if a.stage=='smoke' else []
 for m in models:
  if a.stage=='smoke' and m['id']=='PetitGPT-alpha075':continue
  mid=m['id'];native=mid.startswith('Petit');print('LOAD',a.stage,mid,flush=True);scorer=Scorer(m);model=scorer.model
  tok=load_chat_tokenizer(m['tokenizer']) if native else AutoTokenizer.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False)
  if not native:
   effective,config_evidence=prepare(model.generation_config)
   frozen_config=read(f'ifeval/completion_v2/EFFECTIVE_CONFIG_{mid}.json')
   assert config_evidence['effective_config_id']==frozen_config['effective_config_id']
   assert config_evidence['raw_config_id']==frozen_config['raw_config_id']
   config_id=config_evidence['effective_config_id']
  else:
   config_id=digest(canon({'native_accepted_generate':freeze['native_settings']}))
  actual_inputs=inputs[mid] if a.stage=='full' else inputs[mid][:8];expected_docs=docs if a.stage=='full' else selection
  def generate(row,doc):
   messages=[{'role':'user','content':doc['prompt']}];ids=row['prompt_token_ids']
   actual=encode_prompt(tok,messages,default_system=None,mode='full_context') if native else tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
   assert actual==ids and digest(canon(ids))==row['prompt_ids_sha256'] and digest(doc['prompt'])==row['prompt_sha256']
   assert len(ids)+1280<=row['context']
   torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
   with torch.inference_mode(),torch.autocast('cuda',enabled=False),sdpa_kernel([SDPBackend.MATH]):
    assert not torch.is_autocast_enabled('cuda') and not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled()
    if native:
     v=native_generate(model,tok,messages,1280,autocast_enabled=False);new=v['generated_token_ids'];response=v['output_text_for_scoring'];stop=v['stop_reason']
    else:
     x=torch.tensor([ids],dtype=torch.long,device='cuda');out=invoke(model,effective,config_evidence,input_ids=x,attention_mask=torch.ones_like(x))
     assert out[0,:len(ids)].tolist()==ids;new=out[0,len(ids):].tolist();response=tok.decode(new,skip_special_tokens=True)
     eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;stop='eos' if new and new[-1] in eos else 'max_new_tokens'
   assert new and len(new)<=1280 and all(type(v) is int and 0<=v<(32000 if native else model.config.vocab_size) for v in new)
   assert stop=='eos' or len(new)==1280
   return new,response,stop
  path=f'ifeval/completion_v2/generations/{mid}.jsonl' if a.stage=='full' else f'ifeval/completion_v2/smoke/ifeval_{mid}.jsonl';target=R/path;complete=[];start=time.monotonic()
  if target.exists():
   payload=target.read_bytes()
   if payload and not payload.endswith(b'\n'):
    partial=payload.rsplit(b'\n',1)[-1];(V/f'{a.stage}_{mid}.partial_evidence').write_bytes(partial);raise RuntimeError('Partial JSONL record retained; refusing automatic resume: '+str(target))
   complete=list(lines(path))
   for i,g in enumerate(complete):
    row=actual_inputs[i];assert g['effective_config_id']==config_id and g['model_identity_sha256']==digest(canon(m));assert g['key']==row['key'] and g['fingerprint']==fp and g['response_sha256']==digest(g['response']) and g['generated_ids_sha256']==digest(canon(g['generated_token_ids'])) and g['generated_token_count']==len(g['generated_token_ids'])
    assert g['prompt_ids_sha256']==row['prompt_ids_sha256'] and g['rendered_prompt_sha256']==row['rendered_prompt_sha256']
   assert len(complete)<=len(actual_inputs)
   if len(complete)<len(actual_inputs):save(f'ifeval/completion_v2/RESUME_{mid}_{a.stage}.json',{'completed_rows':len(complete),'fingerprint':fp})
  with target.open('a') as f:
   for i in range(len(complete),len(actual_inputs)):
    row=actual_inputs[i];doc=expected_docs[i];begin=time.monotonic();new,response,stop=generate(row,doc)
    g={'model':mid,'key':row['key'],'prompt_sha256':row['prompt_sha256'],'rendered_prompt_sha256':row['rendered_prompt_sha256'],'prompt_ids_sha256':row['prompt_ids_sha256'],'prompt_tokens':row['prompt_tokens'],'generated_token_ids':new,'generated_ids_sha256':digest(canon(new)),'generated_token_count':len(new),'response':response,'response_sha256':digest(response),'stop_reason':stop,'generation_settings':{'do_sample':False,'temperature':0.0,'max_new_tokens':1280,'num_beams':1,'repetition_penalty':1.0,'until':[],'use_cache':True},'seconds':time.monotonic()-begin,'fingerprint':fp,'effective_config_id':config_id,'model_identity_sha256':digest(canon(m))}
    f.write(canon(g)+'\n');f.flush();os.fsync(f.fileno());complete.append(g)
    if a.stage=='smoke' and i==0:
     ids2,text2,stop2=generate(row,doc);same=(ids2==new and text2==response and stop2==stop);save(f'ifeval/completion_v2/smoke/IFEVAL_DETERMINISM_{mid}.json',{'key':row['key'],'pass':same,'first_generated_ids_sha256':digest(canon(new)),'repeat_generated_ids_sha256':digest(canon(ids2))});assert same
    if (i+1)%10==0 or a.stage=='smoke':
     progress={'stage':a.stage,'model':mid,'task':'ifeval','completed':i+1,'total':len(actual_inputs),'seconds_this_session':time.monotonic()-start,'generated_tokens':sum(x['generated_token_count'] for x in complete)};save('ifeval/completion_v2/PROGRESS.json',progress);print('PROGRESS',mid,i+1,len(actual_inputs),round(time.monotonic()-start,1),flush=True)
  del model,scorer,tok,generate;gc.collect();torch.cuda.empty_cache()
  subprocess.run([sys.executable,'-B',str(V/'runtime/verify_ifeval_v2.py'),a.stage,'--model',mid],check=True)
  result=read(f'ifeval/completion_v2/RESULTS_{mid}.json' if a.stage=='full' else f'ifeval/completion_v2/smoke/IFEVAL_SCORED_{mid}.json');reports.append(result);print('MODEL COMPLETE',mid,a.stage,flush=True)
 if a.stage=='smoke':
  receipts=[{'model':'PetitGPT-alpha075','execution_fingerprint':bridge['old_fingerprint'],'acceptance':'REUSED_VERIFIED','evidence_files':{str(V/'PETITGPT_SMOKE_BRIDGE.json'):sha(V/'PETITGPT_SMOKE_BRIDGE.json')}}]
  for m in models[1:]:
   mid=m['id'];ps=[V/f'smoke/ifeval_{mid}.jsonl',V/f'smoke/IFEVAL_DETERMINISM_{mid}.json',V/f'smoke/IFEVAL_SCORED_{mid}.json',V/f'verifier/{mid}_smoke.jsonl']
   assert read(f'ifeval/completion_v2/smoke/IFEVAL_DETERMINISM_{mid}.json')['pass']
   receipts.append({'model':mid,'execution_fingerprint':fp,'acceptance':'NEW_EXECUTION_PASS','evidence_files':{str(p):sha(p) for p in ps}})
  save('ifeval/completion_v2/smoke/IFEVAL.json',{'status':'PASS','stage':a.stage,'continuation_fingerprint':fp,'execution_identities':receipts,'results':reports})
 else:save('ifeval/completion_v2/RESULTS.json',{'status':'PASS','stage':a.stage,'fingerprint':fp,'results':reports})
 print('STAGE COMPLETE',a.stage,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save('ifeval/completion_v2/STOP_RUNTIME.json',{'status':'PARTIAL_STOPPED','stage':sys.argv[1:],'exact_error':traceback.format_exc(),'action':'Stopped; completed rows retained. No retry or protocol change.'});raise
