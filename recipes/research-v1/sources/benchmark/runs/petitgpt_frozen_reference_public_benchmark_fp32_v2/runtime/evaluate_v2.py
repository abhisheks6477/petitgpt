from common import *
from likelihood_checks import continuation_logprobs,token_parity,sum_parity,same_logits_sum_parity,record_before_check
import datetime,gc,time,itertools,argparse
V1=REPO/'runs/petitgpt_frozen_reference_public_benchmark_v1'
def append(path,x):
 with (ROOT/path).open('a') as f:f.write(canonical(x)+'\n');f.flush()
def verify_frozen():
 for f,h in read('runtime/CODE_FREEZE.json').items():assert sha(ROOT/f)==h,f
 for f,h in read('preparation/FROZEN_V2_FILES.json').items():assert sha(ROOT/f)==h,f
class Observed(Scorer):
 def __init__(self,m):
  primary_policy();super().__init__(m);self.tag={};self.last_return=None;self.bf16=False
  existing=jsonl('runtime/FORWARD_EVENTS.jsonl') if (ROOT/'runtime/FORWARD_EVENTS.jsonl').exists() else []
  self.next_id=sum(e['event']=='invoked' for e in existing)
  self.extra=sum(e['event']=='invoked' and e['category']!='primary' for e in existing)
 def _model_call(self,x,**kw):
  assert not kw
  is_extra=self.tag['category']!='primary'
  if is_extra:assert self.extra<192;self.extra+=1
  self.next_id+=1;fid=self.next_id
  event={'forward_id':fid,'model':self.identity['id'],**self.tag,'input_shape':list(x.shape),'input_ids_sha256':digest(canonical(x[0].tolist())),'route':'BF16_DIAGNOSTIC' if self.bf16 else 'FP32_V2','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  append('runtime/FORWARD_EVENTS.jsonl',{**event,'event':'invoked'})
  if self.bf16:
   torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
   torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=self.model(x)
   self.forward_calls+=1
  else:out=super()._model_call(x)
  self.last_return={**event,'event':'returned','logits_shape':list(out.shape),'logits_dtype':str(out.dtype),'attention_implementation':'native SDPA is_causal=True' if self.identity['id']=='PetitGPT-alpha075' else self.model.config._attn_implementation,'autocast_inside_forward':self.bf16,'SDPA_context':'V1 enabled defaults' if self.bf16 else 'MATH only','cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
  append('runtime/FORWARD_EVENTS.jsonl',self.last_return)
  return out
 def measure(self,ids,C,targets):
  x=torch.tensor(ids,device='cuda',dtype=torch.long).unsqueeze(0);out=self._model_call(x)
  if self.bf16:
   selected=out[0,C-1:C+len(targets)-1].float().log_softmax(-1)
   v=selected.gather(1,torch.tensor(targets,device='cuda')[:,None]).squeeze(1)
  else:v=continuation_logprobs(out,C,targets)
  manual=torch.stack([out[0,C-1+j].float().log_softmax(-1)[t] for j,t in enumerate(targets)]).sum().item()
  record={'input_ids':ids,'C':C,'K':len(targets),'target_ids':targets,'scored_positions':list(range(C-1,C+len(targets)-1)),'token_logprobs':v.cpu().tolist(),'fp32_sum':float(v.sum()),'explicit_position_sum':manual,'actual_forward':self.last_return}
  return record

def run(mid,phase):
 verify_frozen();protocol=read('preparation/TASK_PROTOCOL_FREEZE_V2.json');m=next(x for x in read('preparation/MODEL_SELECTION_FREEZE.json')['models'] if x['id']==mid)
 if phase=='full':assert read('checks/LIKELIHOOD_PARITY_SUMMARY.json')['status']=='PASS'
 scorer=Observed(m)
 identity={'model':mid,'phase':phase,'unique_parameters':scorer.parameters,'tied_embeddings':True,'parameter_dtypes':sorted(set(str(x.dtype) for x in scorer.model.parameters())),'config':scorer.config,'HF_attention_implementation':getattr(getattr(scorer.model,'config',None),'_attn_implementation',None),'environment':check_env()}
 dump(f'runtime/MODEL_IDENTITY_{mid}_{phase}.json',identity)
 if mid=='PetitGPT-alpha075':
  for k,v in m['expected_config'].items():assert scorer.config[k]==v
 fixtures=read('preparation/SYNTHETIC_INPUTS_FREEZE.json')[mid];path=f'checks/FP32_PARITY_{mid}.jsonl'
 if phase=='checks':
  assert not (ROOT/path).exists(),'Never silently rerun completed checks'
  if mid=='PetitGPT-alpha075':
   f=fixtures[1];measurements=[];scorer.bf16=True
   for name,ids in [('unpadded',f['input_ids']),('repeat',f['input_ids']),('future_zeros',f['input_ids']+[0]*7),('future_ordinary',f['input_ids']+f['ordinary_suffix_ids'])]:
    scorer.tag={'category':'bf16_diagnostic','request_id':'synthetic:1','variant':name}
    r=scorer.measure(ids,f['C'],f['continuation_ids']);r.update({'measurement':name,'fixture_index':1,'gated':False})
    r['pairwise_deltas']={x['measurement']:{'per_token_signed_delta':[a-b for a,b in zip(r['token_logprobs'],x['token_logprobs'])],'sequence_signed_delta':r['fp32_sum']-x['fp32_sum'],'bit_identical_token_values':r['token_logprobs']==x['token_logprobs']} for x in measurements}
    append('checks/BF16_SYNTHETIC_DIAGNOSTIC.jsonl',r);measurements.append(r)
   scorer.bf16=False;primary_policy()
  for f in fixtures:
   base=None
   for name,ids in [('unpadded',f['input_ids']),('repeat',f['input_ids']),('future_zeros',f['input_ids']+[0]*7),('future_ordinary',f['input_ids']+f['ordinary_suffix_ids'])]:
    scorer.tag={'category':'fp32_synthetic','request_id':f'synthetic:{f["fixture_index"]}','variant':name}
    r=scorer.measure(ids,f['C'],f['continuation_ids']);r.update({'fixture_index':f['fixture_index'],'measurement':name})
    gates={'same_logits_sum':same_logits_sum_parity(r['fp32_sum'],r['explicit_position_sum'])}
    if base is not None:gates['versus_unpadded']=token_parity(base['token_logprobs'],r['token_logprobs']);r['bit_identical_to_unpadded']=base['token_logprobs']==r['token_logprobs']
    if name=='future_ordinary':gates['same_shape_suffix']=token_parity(zero['token_logprobs'],r['token_logprobs'])
    r['comparisons']=gates;r['pass']=all(g['pass'] for g in gates.values())
    record_before_check(ROOT/path,r,r['pass'])
    if name=='unpadded':base=r
    if name=='future_zeros':zero=r
   scorer.tag={'category':'fp32_reference','request_id':f'synthetic:{f["fixture_index"]}','variant':'full_unchanged_HFLM_helper'}
   val=scorer.reference(f['context_ids'],f['continuation_ids']);gate=sum_parity(base['fp32_sum'],val,f['K'])
   record_before_check(ROOT/path,{'fixture_index':f['fixture_index'],'measurement':'full_unchanged_HFLM_helper','fp32_sum':val,'C':f['C'],'K':f['K'],'input_ids':f['input_ids'],'targets':f['continuation_ids'],'comparison':gate,'actual_forward':scorer.last_return,'pass':gate['pass']},gate['pass'])
   for j in f['prefix_witness_target_indices']:
    scorer.tag={'category':'fp32_prefix','request_id':f'synthetic:{f["fixture_index"]}','variant':f'prefix_target_{j}'}
    # Prefix length is exactly the number of preceding real tokens. No cache or future tokens.
    ids=(f['context_ids']+f['continuation_ids'])[:f['C']+j]
    r=scorer.measure(ids,len(ids),[f['continuation_ids'][j]])
    gate=token_parity([base['token_logprobs'][j]],r['token_logprobs']);r.update({'fixture_index':f['fixture_index'],'measurement':'prefix_witness','original_target_index':j,'comparison':gate,'pass':gate['pass']})
    record_before_check(ROOT/path,r,gate['pass'])
  print('SYNTHETIC_PASS',mid,flush=True)
 texts=[json.loads(s) for s in (V1/'local_data/REQUEST_TEXT.jsonl').read_text().splitlines()]
 encs={r['request_id']:r for r in [json.loads(s) for s in (V1/f'local_data/ENCODINGS_{mid}.jsonl').read_text().splitlines()]}
 out=f'results/RAW_CHOICES_{mid}.jsonl';old=jsonl(out) if (ROOT/out).exists() else [];done={(r['request_id'],r['choice_index']):r for r in old};assert len(done)==len(old)
 policy_hash=sha(ROOT/'preparation/PROTOCOL_AMENDMENT_V2.json');code_hash=sha(ROOT/'runtime/CODE_FREEZE.json');model_hash=m.get('sha256') or next(x['sha256'] for x in m['files'] if x['path'].endswith('/model.safetensors'))
 selected=[r for r in texts if phase=='full' or r['request_id'] in protocol['technical_ids']];new=0;t0=time.monotonic()
 with (ROOT/out).open('a',buffering=1) as stream:
  for n,r in enumerate(selected):
   for j,e in enumerate(encs[r['request_id']]['choices']):
    key=(r['request_id'],j)
    if key in done:continue
    scorer.tag={'category':'primary','request_id':r['request_id'],'choice_index':j,'phase':phase}
    ll=scorer.score(e['context_ids'],e['continuation_ids'])
    row={'model':mid,'task':r['task'],'split':r['split'],'doc_id':r['doc_id'],'request_id':r['request_id'],'choice_index':j,'model_weights_sha256':model_hash,'task_protocol_sha256':sha(ROOT/'preparation/TASK_PROTOCOL_FREEZE_V2.json'),'code_sha256':code_hash,'numeric_policy_sha256':policy_hash,'prompt_sha256':r['prompt_sha256'],'choice_sha256':r['choice_sha256'][j],'continuation_sha256':r['continuation_sha256'][j],'encoding_sha256':e['encoding_sha256'],'C':len(e['context_ids']),'K':len(e['continuation_ids']),'total_tokens':e['tokens'],'continuation_logprob':ll,'normalization_denominator':r['normalization_denominators'][j],'normalized_score':ll/r['normalization_denominators'][j],'forward_id':scorer.next_id}
    stream.write(canonical(row)+'\n');stream.flush();os.fsync(stream.fileno());done[key]=row;new+=1
   if phase=='checks':
    e=encs[r['request_id']]['choices'][0];scorer.tag={'category':'fp32_technical_reference','request_id':r['request_id'],'choice_index':0,'variant':'full_unchanged_HFLM_helper'}
    val=scorer.reference(e['context_ids'],e['continuation_ids']);base=done[(r['request_id'],0)]['continuation_logprob'];gate=sum_parity(base,val,len(e['continuation_ids']))
    record_before_check(ROOT/path,{'measurement':'technical_document_reference','request_id':r['request_id'],'choice_index':0,'C':len(e['context_ids']),'K':len(e['continuation_ids']),'primary_sum':base,'reference_sum':val,'encoding_sha256':e['encoding_sha256'],'comparison':gate,'actual_forward':scorer.last_return,'pass':gate['pass']},gate['pass'])
   if (n+1)%100==0:print(mid,phase,f'{n+1}/{len(selected)} docs; {new} new primaries; {time.monotonic()-t0:.1f}s',flush=True)
 dump(f'runtime/COMPLETE_{mid}_{phase}.json',{'status':'COMPLETE','model':mid,'phase':phase,'new_primary_candidates':new,'forward_calls':scorer.forward_calls,'seconds_primary':time.monotonic()-t0})
 print('COMPLETE',mid,phase,new,scorer.forward_calls,flush=True)
 del scorer;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['checks','full']);ap.add_argument('model');a=ap.parse_args();run(a.model,a.phase)
