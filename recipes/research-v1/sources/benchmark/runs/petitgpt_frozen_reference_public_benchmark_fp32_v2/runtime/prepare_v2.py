from common import *
import copy,datetime
V1=REPO/'runs/petitgpt_frozen_reference_public_benchmark_v1'
assert not (ROOT/'preparation/PROTOCOL_AMENDMENT_V2.json').exists()
assert 'Ran 14 tests' in (ROOT/'checks/CPU_TEST_RESULTS.txt').read_text() and (ROOT/'checks/CPU_TEST_RESULTS.txt').read_text().rstrip().endswith('OK')
primary_policy();dump('runtime/ACTUAL_ENVIRONMENT.json',check_env())
v1=read('preparation/V1_TASK_PROTOCOL_FREEZE.json');models=read('preparation/MODEL_SELECTION_FREEZE.json')['models']
fixtures={};prechecks=[];extra=4
texts=[json.loads(s) for s in (V1/'local_data/REQUEST_TEXT.jsonl').read_text().splitlines()]
index=jsonl('preparation/REQUEST_INDEX.jsonl');assert len(texts)==len(index)==4214
for m in models:
 enc=Encoding(m);stored=[json.loads(s) for s in (V1/f'local_data/ENCODINGS_{m["id"]}.jsonl').read_text().splitlines()];assert len(stored)==4214
 maximum=0;total=0
 for r,idx,er in zip(texts,index,stored):
  assert r['request_id']==idx['request_id']==er['request_id'] and digest(r['prompt'])==idx['prompt_sha256']
  for j,(text,e) in enumerate(zip(r['continuations'],er['choices'])):
   c,k=enc.pair(r['prompt'],text);assert c==e['context_ids'] and k==e['continuation_ids']
   assert enc.tok_encode(r['prompt']+text)==c+k and c and k
   assert e['encoding_sha256']==digest(canonical([c,k]))
   assert text==' '+r['choices'][j] and len(r['choices'][j])==idx['normalization_denominators'][j]
   maximum=max(maximum,len(c)+len(k));total+=1
 assert maximum== (257 if enc.native else 242) and total==13177
 if enc.native:special={i for i,t in enc.tokenizer.get_added_tokens_decoder().items() if t.special}
 else:special=set(enc.tokenizer.all_special_ids)
 records=[]
 for i,f in enumerate(v1['synthetic_fixtures']):
  c,k=enc.pair(f['context'],f['continuation']);ordinary=[t for t in c if t not in special];assert ordinary
  suffix=[ordinary[j%len(ordinary)] for j in range(7)];assert not(set(suffix)&special)
  witnesses=sorted(set([0,len(k)-1]));extra+=5+len(witnesses)
  records.append({'fixture_index':i,**f,'context_ids':c,'continuation_ids':k,'C':len(c),'K':len(k),'input_ids':(c+k)[:-1],'positions':list(range(len(c)-1,len(c)+len(k)-1)),'ordinary_suffix_ids':suffix,'prefix_witness_target_indices':witnesses,'special_ids':sorted(special)})
 fixtures[m['id']]=records;extra+=4
 prechecks.append({'model':m['id'],'all_v1_token_encodings_equal':True,'zero_prefix_mismatches':True,'maximum_total_tokens':maximum,'candidate_sequences':total,'documents':4214})
assert extra<=192
dump('preparation/SYNTHETIC_INPUTS_FREEZE.json',fixtures)
policy={'parameters':'FP32','forward':'FP32','autocast':False,'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False,'cudnn_benchmark':False,'float32_matmul_precision':'highest','SDPA':'MATH','HF_attention_implementation':'sdpa','batch_size':1,'primary_padding':False,'use_cache':False,'compile':False,'log_softmax':'FP32','sum':'FP32','logits_dtype_assertion':'torch.float32 before any cast'}
amend={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'v1_status':'BLOCKED_LIKELIHOOD_PARITY_FAILED','v1_protocol_sha256':sha(ROOT/'preparation/V1_TASK_PROTOCOL_FREEZE.json'),'v1_precision':'FP32 parameters / BF16 autocast forward / FP32 log-softmax','v1_backend_flags':json.loads((V1/'runtime/ACTUAL_ENVIRONMENT.json').read_text())['backend'],'v2_policy':policy,'changes':['Disable CUDA autocast for complete forward','Disable cuDNN TF32 (V1 True)','Explicitly restrict SDPA to MATH (V1 multiple enabled)','Explicit official HF sdpa implementation','Actual logits must be FP32'],'unchanged':'all model/checkpoint/source/tokenizer/task/dataset/request/choice/metric identities','bf16_diagnostic':'exactly four forwards on V1 failing native synthetic fixture; observed separately, no BF16 gate relaxation','gates':{'same_logits':{'atol':1e-5,'rtol':1e-6},'cross_forward_per_token':{'atol':5e-4,'rtol':1e-5,'scale':'symmetric max absolute'},'sequence_sum':'sum of per-token allowances','reference_scalar':{'atol':'K*5e-4','rtol':1e-5,'scale':'symmetric max absolute'}},'technical_extra_forwards_planned':extra,'technical_extra_cap':192,'supplied_check_code_sha256':sha(ROOT/'runtime/likelihood_checks.py')}
dump('preparation/PROTOCOL_AMENDMENT_V2.json',amend)
v2=copy.deepcopy(v1);v2.update({'version':'FP32_V2','utc':amend['utc'],'forward':'FP32; CUDA autocast disabled','numeric_policy':policy,'parity_tolerances':amend['gates'],'padding_fixture':'seven future zero IDs and seven frozen ordinary non-special IDs; compare all original target positions','synthetic_scope':'same three V1 text fixtures; repeat, zero suffix, ordinary suffix, full reference helper, first/last prefix witnesses','length_prechecks':prechecks,'budget':{'documents_per_model':4214,'document_model_evaluations_total':12642,'candidate_sequences_primary_total':39531,'extra_candidate_sequences_planned':extra,'extra_cap':192},'v1_source_freeze':'preparation/V1_TASK_PROTOCOL_FREEZE.json'})
for key in ['tasks','prompt_template','target_delimiter','choice_text','fewshot','chat_template','token_boundary','special_tokens','BOS','EOS','max_total_tokens','batch_size','acc','acc_norm','tie','technical_ids','technical_selection','parameter_dtype','log_softmax_dtype']:assert v2[key]==v1[key],key
dump('preparation/TASK_PROTOCOL_FREEZE_V2.json',v2)
dump('runtime/CODE_FREEZE.json',{str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'runtime').glob('*.py'))})
dump('preparation/FROZEN_V2_FILES.json',{str(f.relative_to(ROOT)):sha(f) for sub in ['preparation','source'] for f in sorted((ROOT/sub).rglob('*')) if f.is_file()})
print('V2_FROZEN',prechecks,'extra_forwards',extra,flush=True)
