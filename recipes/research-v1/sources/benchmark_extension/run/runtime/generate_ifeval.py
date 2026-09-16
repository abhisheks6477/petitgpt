"""Frozen canonical-chat IFEval generation; inference and verifier environments separate."""
from pathlib import Path
import os,sys,json,hashlib,time,gc,subprocess,argparse,collections
R=Path(__file__).resolve().parents[1];S=R.parent/'source'
# This module imports no evaluator-only packages.
from likelihood import Scorer,sha,canon,digest,read,lines,save
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer
from src.chat_template import load_chat_tokenizer,encode_prompt
from src.accepted_generate import generate as native_generate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full']);a=ap.parse_args()
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
 paths=[Path(__file__),R/'runtime/likelihood.py',R/'runtime/verify_ifeval.py',R/'protocol/IFEVAL_PROTOCOL.json',R/'protocol/IFEVAL_TEMPLATE_RESOLUTION.json',R/'protocol/IFEVAL_CONTEXT_PREFLIGHT.json',R/'protocol/IFEVAL_SMOKE_SELECTION.json',R/'protocol/MODEL_IDENTITIES.json',R/'protocol/DATA_IDENTITIES.json',R/'protocol/EVALUATOR_ENVIRONMENT.json',R/'protocol/ENVIRONMENT.json']
 paths += [S/'recipes/research-v1/sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/common.py', S/'recipes/research-v1/sources/benchmark/sft/train_sft.py', R/'ifeval/normalized_rows.jsonl']
 paths+=list((S/'recipes/research-v1/sources/native_inference/src').glob('*.py'));paths+=list((R/'upstream/lm_eval/tasks/ifeval').glob('*.py'))
 bindings={str(p):sha(p) for p in paths};fp=digest(canon(bindings))
 freeze={'fingerprint':fp,'files':bindings,'decoder':'PetitGPT released accepted_generate token decoding; HF pinned tok_decode(skip_special_tokens=True), tokenizer cleanup default; no added cleanup','use_cache':True,'batch_size':1,'numeric_policy':'FP32, autocast off, TF32 off, SDPA MATH','max_new_tokens':1280,'sampling':False}
 if (R/'protocol/IFEVAL_CODE_FREEZE.json').exists():assert read('protocol/IFEVAL_CODE_FREEZE.json')==freeze
 else:save('protocol/IFEVAL_CODE_FREEZE.json',freeze)
 if a.stage=='full':assert read('smoke/IFEVAL.json')['status']=='PASS' and read('smoke/IFEVAL.json')['fingerprint']==fp
 reports=[]
 for m in models:
  mid=m['id'];native=mid.startswith('Petit');print('LOAD',a.stage,mid,flush=True);scorer=Scorer(m);model=scorer.model
  tok=load_chat_tokenizer(m['tokenizer']) if native else AutoTokenizer.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False)
  if not native:
   assert model.generation_config.repetition_penalty==1.0
   assert model.generation_config.no_repeat_ngram_size==0 and model.generation_config.encoder_repetition_penalty==1.0
   assert model.generation_config.forced_bos_token_id is None and model.generation_config.forced_eos_token_id is None
   assert model.generation_config.min_length==0 and not model.generation_config.bad_words_ids and not model.generation_config.suppress_tokens
   save(f'protocol/GENERATION_CONFIG_{mid}.json',model.generation_config.to_dict())
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
     x=torch.tensor([ids],dtype=torch.long,device='cuda');out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=1280,do_sample=False,temperature=0.0,num_beams=1,use_cache=True,repetition_penalty=1.0)
     assert out[0,:len(ids)].tolist()==ids;new=out[0,len(ids):].tolist();response=tok.decode(new,skip_special_tokens=True)
     eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;stop='eos' if new and new[-1] in eos else 'max_new_tokens'
   assert new and len(new)<=1280 and all(type(v) is int and 0<=v<(32000 if native else model.config.vocab_size) for v in new)
   assert stop=='eos' or len(new)==1280
   return new,response,stop
  path=f'ifeval/generations/{mid}.jsonl' if a.stage=='full' else f'smoke/ifeval_{mid}.jsonl';target=R/path;complete=[];start=time.monotonic()
  if target.exists():
   complete=list(lines(path))
   for i,g in enumerate(complete):
    row=actual_inputs[i];assert g['key']==row['key'] and g['fingerprint']==fp and g['response_sha256']==digest(g['response']) and g['generated_ids_sha256']==digest(canon(g['generated_token_ids'])) and g['generated_token_count']==len(g['generated_token_ids'])
    assert g['prompt_ids_sha256']==row['prompt_ids_sha256'] and g['rendered_prompt_sha256']==row['rendered_prompt_sha256']
   assert len(complete)<=len(actual_inputs)
   if len(complete)<len(actual_inputs):save(f'ifeval/RESUME_{mid}.json',{'completed_rows':len(complete),'fingerprint':fp})
  with target.open('a') as f:
   for i in range(len(complete),len(actual_inputs)):
    row=actual_inputs[i];doc=expected_docs[i];begin=time.monotonic();new,response,stop=generate(row,doc)
    g={'model':mid,'key':row['key'],'prompt_sha256':row['prompt_sha256'],'rendered_prompt_sha256':row['rendered_prompt_sha256'],'prompt_ids_sha256':row['prompt_ids_sha256'],'prompt_tokens':row['prompt_tokens'],'generated_token_ids':new,'generated_ids_sha256':digest(canon(new)),'generated_token_count':len(new),'response':response,'response_sha256':digest(response),'stop_reason':stop,'generation_settings':{'do_sample':False,'temperature':0.0,'max_new_tokens':1280,'num_beams':1,'repetition_penalty':1.0,'until':[],'use_cache':True},'seconds':time.monotonic()-begin,'fingerprint':fp}
    f.write(canon(g)+'\n');f.flush();os.fsync(f.fileno());complete.append(g)
    if a.stage=='smoke' and i==0:
     ids2,text2,stop2=generate(row,doc);same=(ids2==new and text2==response and stop2==stop);save(f'smoke/IFEVAL_DETERMINISM_{mid}.json',{'key':row['key'],'pass':same,'first_generated_ids_sha256':digest(canon(new)),'repeat_generated_ids_sha256':digest(canon(ids2))});assert same
    if (i+1)%10==0 or a.stage=='smoke':
     progress={'stage':a.stage,'model':mid,'task':'ifeval','completed':i+1,'total':len(actual_inputs),'seconds_this_session':time.monotonic()-start,'generated_tokens':sum(x['generated_token_count'] for x in complete)};save('ifeval/completion_v1/PROGRESS.json',progress);print('PROGRESS',mid,i+1,len(actual_inputs),round(time.monotonic()-start,1),flush=True)
  del model,scorer;gc.collect();torch.cuda.empty_cache()
  subprocess.run([sys.executable,'-B',str(R/'runtime/verify_ifeval.py'),a.stage,'--model',mid],check=True)
  result=read(f'ifeval/RESULTS_{mid}.json' if a.stage=='full' else f'smoke/IFEVAL_SCORED_{mid}.json');reports.append(result);print('MODEL COMPLETE',mid,a.stage,flush=True)
 save('smoke/IFEVAL.json' if a.stage=='smoke' else 'ifeval/RESULTS.json',{'status':'PASS','stage':a.stage,'fingerprint':fp,'results':reports})
 print('STAGE COMPLETE',a.stage,flush=True)
if __name__=='__main__':main()
