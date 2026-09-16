"""Narrow identity refresh and native smoke provenance bridge; no model execution."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.metadata,subprocess,datetime,collections
V=Path(__file__).resolve().parents[1];R=V.parents[1];S=R.parent/'source'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def read(p):return json.loads((R/p).read_text())
def rows(p):return [json.loads(x) for x in (R/p).read_text().splitlines()]
def save(p,x):(V/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
assert not (V/'CODE_FREEZE.json').exists()
old=read('protocol/IFEVAL_CODE_FREEZE.json');assert old['fingerprint']=='9420b9c1fb23ba1c7b76584a9310a80436c38117b6d056b3b373e07b2070fb8b'
assert digest(canon(old['files']))==old['fingerprint']
checks={}
def verify(p,expected):
 assert sha(p)==expected,str(p);checks[str(p)]=expected
for p,h in old['files'].items():verify(p,h)
archives={'PETITGPT_THREE_TASK_BENCHMARK_EXTENSION_V1_EVIDENCE.tar.gz':'87b730e003ca6f1c2a14487e90f193ae1ee7f313306f43f8efc7db3c548d802f','PETITGPT_IFEVAL_CANONICAL_TEMPLATE_CONTINUE_V1_COMPLETION_EVIDENCE.tar.gz':'2fd3f17b9cb9407bc6946fd0df4c1036651f971b646e90aca6fdbf93bb3a5a6a'}
for p,h in archives.items():verify(R.parent/p,h)
# Freeze historical continuation files as they exist before new inference.
for p in (R/'ifeval/completion_v1').rglob('*'):
 if p.is_file():checks[str(p)]=sha(p)
for x in read('ifeval/completion_v1/PRIOR_EVIDENCE_IDENTITIES.json')['verified_existing_files']:
 if x['path']=='runtime/generate_ifeval.py':
  routing=read('ifeval/completion_v1/GENERATOR_OUTPUT_ROUTING.json')
  assert x['sha256']==routing['original_sha256']
  verify(R/'ifeval/completion_v1/generate_ifeval_pre_continuation.py',x['sha256'])
  verify(R/x['path'],routing['effective_sha256'])
 else:verify(R/x['path'],x['sha256'])
models=read('protocol/MODEL_IDENTITIES.json')
for m in models:
 for x in m['verified_files']:verify(x['path'],x['sha256'])
 for base in [R/'ifeval/generations',V/'generations']:
  p=base/(m['id']+'.jsonl');assert not p.exists() or p.stat().st_size==0,str(p)
for m in models[1:]:assert not (R/f"smoke/ifeval_{m['id']}.jsonl").exists()
for mid,item in read('protocol/IFEVAL_PROTOCOL.json')['context_preflight'].items():verify(R/item['file'],item['sha256'])
assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()=='b82ede26bff93423bd05fa08b8cae6ebd62d1dee'
harness=read('protocol/SOURCE_IDENTITIES.json');assert harness['harness_commit']=='b954108c9baaaa934b4ad842033b31a97ee30816'
for item in harness['files']:
 if '/ifeval/' in item['file']:verify(R/item['file'],item['sha256'])
for pkg,expected in read('protocol/ENVIRONMENT.json')['versions'].items():assert importlib.metadata.version(pkg)==expected,pkg
sys.path[:0]=[str(R/'evaluator_deps')]
env=read('protocol/EVALUATOR_ENVIRONMENT.json')
for pkg,expected in env['versions'].items():assert importlib.metadata.version(pkg)==expected,pkg
for group in ['package_archives','resource_files']:
 for p,h in env[group].items():verify(R/p,h)
prev=read('ifeval/completion_v1/ENVIRONMENT_RECHECK.json');assert prev['status']=='PASS'
for pkg,item in prev['packages'].items():
 for x in item['record_verified_files']:
  if 'sha256' in x:verify(x['file'],x['sha256'])
  else:assert x['external_console_entry'] is True and Path(x['file']).exists()==x['exists']
assert read('ifeval/verifier/FIXTURES.json')['status']=='PASS'
for name in ['ifeval/verifier/FIXTURES.json','protocol/IFEVAL_CODE_FREEZE.json','ifeval/STOP.json','protocol/IFEVAL_TEMPLATE_CONFLICT.json']:
 checks[str(R/name)]=sha(R/name)
# Only generators/evaluators, no unrelated process details.
ps=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
active=[line for line in ps.splitlines() if ('python' in line.split()[1] if len(line.split())>1 else False) and any(n in line for n in ['generate_ifeval.py','generate_ifeval_v2.py','verify_ifeval.py','verify_ifeval_v2.py'])]
assert not active,active
smi=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not smi.strip(),smi
# Prove the native generation branch and numeric/formatting statements unchanged.
a=(R/'runtime/generate_ifeval.py').read_text();b=(V/'runtime/generate_ifeval_v2.py').read_text()
native_line="v=native_generate(model,tok,messages,1280,autocast_enabled=False);new=v['generated_token_ids'];response=v['output_text_for_scoring'];stop=v['stop_reason']"
assert native_line in a and native_line in b
unchanged=[line for line in a.splitlines() if any(s in line for s in ['messages=[','actual=encode_prompt','assert actual==ids','assert len(ids)+1280','torch.set_float32','with torch.inference_mode','assert not torch.is_autocast','native_generate(model,'])]
for line in unchanged:assert line in b,line
vo=(R/'runtime/verify_ifeval.py').read_text();vn=(V/'runtime/verify_ifeval_v2.py').read_text()
def fn(text,name):return ast.dump(next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
normalized=vn.replace('ifeval/completion_v2/generations/','ifeval/generations/').replace('ifeval/completion_v2/smoke/','smoke/').replace('ifeval/completion_v2/verifier/','ifeval/verifier/').replace('ifeval/completion_v2/RESULTS_','ifeval/RESULTS_')
assert fn(vo,'score')==fn(normalized,'score') and fn(vo,'fixtures')==fn(vn,'fixtures')
# Tokenizer only: no PetitGPT model reload and no checker rerun.
sys.path.insert(0,str(S/'recipes/research-v1/sources/native_inference'))
from src.chat_template import load_chat_tokenizer,encode_prompt
tok=load_chat_tokenizer(models[0]['tokenizer'])
mid=models[0]['id'];gp=R/f'smoke/ifeval_{mid}.jsonl';cp=R/f'ifeval/verifier/{mid}_smoke.jsonl'
verify(gp,'bfb03e6777a88a455777310b7f340326c8d546a7cada51ef1198ba9e78e62f26');verify(cp,'438cd0a55318922f2e702f1ce7770f126db16cbc11ccaabd9abb5e36ade7d16a')
gens=rows(str(gp));scored=rows(str(cp));docs=rows('ifeval/normalized_rows.jsonl');inputs=rows(f'ifeval/generation_inputs_{mid}.jsonl');keys=read('protocol/IFEVAL_SMOKE_SELECTION.json')['keys']
assert [g['key'] for g in gens]==keys==[s['key'] for s in scored]
settings={'do_sample':False,'temperature':0.0,'max_new_tokens':1280,'num_beams':1,'repetition_penalty':1.0,'until':[],'use_cache':True}
for g,c,d,i in zip(gens,scored,docs,inputs):
 ids=g['generated_token_ids'];before=ids[:-1] if ids[-1]==3 else ids
 assert g['fingerprint']==old['fingerprint'] and g['generation_settings']==settings
 assert g['generated_ids_sha256']==digest(canon(ids)) and len(ids)==g['generated_token_count']
 assert all(type(t) is int and 0<=t<32000 for t in ids) and 0<len(ids)<=1280
 assert g['response']==tok.decode(before,skip_special_tokens=False)
 assert g['response_sha256']==digest(g['response'])==c['response_sha256']
 assert g['stop_reason']==('eos' if ids[-1]==3 else 'max_new_tokens')
 assert g['key']==d['key']==i['key'] and g['prompt_sha256']==digest(d['prompt'])==i['prompt_sha256']
 actual=encode_prompt(tok,[{'role':'user','content':d['prompt']}],default_system=None,mode='full_context')
 assert actual==i['prompt_token_ids'] and digest(canon(actual))==g['prompt_ids_sha256']==i['prompt_ids_sha256']
 assert g['rendered_prompt_sha256']==i['rendered_prompt_sha256']
 assert len(c['inst_level_strict_acc'])==len(c['inst_level_loose_acc'])==len(d['instruction_id_list'])
 assert c['prompt_level_strict_acc']==all(c['inst_level_strict_acc']) and c['prompt_level_loose_acc']==all(c['inst_level_loose_acc'])
receipt=read(f'smoke/IFEVAL_DETERMINISM_{mid}.json');assert receipt['pass'] and receipt['key']==keys[0]
assert receipt['first_generated_ids_sha256']==receipt['repeat_generated_ids_sha256']==gens[0]['generated_ids_sha256']=='16e6ace0d457de6319917ef695365078c19a4aff346f0f679ebfd7afae57ce83'
result=read(f'smoke/IFEVAL_SCORED_{mid}.json');assert result['status']=='PASS' and result['independent_recomputation']=='PASS'
assert result['rows']==8 and result['instruction_count']==sum(len(c['inst_level_strict_acc']) for c in scored)==14
assert result['prompt_strict_correct']==sum(c['prompt_level_strict_acc'] for c in scored)==0
assert result['prompt_loose_correct']==sum(c['prompt_level_loose_acc'] for c in scored)==0
assert result['instruction_strict_correct']==sum(sum(c['inst_level_strict_acc']) for c in scored)==2
assert result['instruction_loose_correct']==sum(sum(c['inst_level_loose_acc']) for c in scored)==3
assert result['generations_sha256']==sha(gp) and result['verifier_rows_sha256']==sha(cp)
for name in [f'smoke/IFEVAL_DETERMINISM_{mid}.json',f'smoke/IFEVAL_SCORED_{mid}.json']:
 checks[str(R/name)]=sha(R/name)
save('IDENTITY_REFRESH.json',{'status':'PASS','checked_files':checks,'old_archives_unchanged':True,'old_and_new_full_rows':0,'active_evaluators':active,'gpu_compute_processes':smi,'package_versions_unchanged':True,'fixtures_reused':True,'native_model_loads':0,'native_smoke_generations':0,'native_checker_reruns':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
save('NATIVE_EQUIVALENCE.json',{'status':'PASS','scope':'PetitGPT-alpha075 existing eight smoke rows ONLY','old_fingerprint':old['fingerprint'],'identical_native_and_numeric_statements':unchanged,'all_old_frozen_sources_unchanged':True,'checker_AST_identical_except_output_paths':True,'settings':settings,'saved_rows_and_decoding_verified':True,'old_checker_boolean_aggregation_verified':True,'fresh_native_verifier_executions':0})
assert read('ifeval/completion_v2/CONFIG_ONLY_REGRESSIONS.json')['status']=='PASS'
bindings=dict(old['files'])
for p in (V/'runtime').glob('*.py'):bindings[str(p)]=sha(p)
for name in ['IDENTITY_REFRESH.json','NATIVE_EQUIVALENCE.json','CONFIG_ONLY_REGRESSIONS.json','INSTALLED_CONFIG_SOURCE.json','REPAIR.diff']+[f"EFFECTIVE_CONFIG_{m['id']}.json" for m in models[1:]]:bindings[str(V/name)]=sha(V/name)
for p,h in read('ifeval/completion_v2/INSTALLED_CONFIG_SOURCE.json')['sources'].items():verify(p,h);bindings[p]=h
fp=digest(canon(bindings));freeze=dict(old);freeze.update(fingerprint=fp,files=bindings,old_fingerprint=old['fingerprint'],semantic_protocol_unchanged=True,implementation_changed='HF effective GenerationConfig compatibility and continuation output routing only',native_settings=settings)
save('CODE_FREEZE.json',freeze)
evidence={str(p):sha(p) for p in [gp,cp,R/f'smoke/IFEVAL_DETERMINISM_{mid}.json',R/f'smoke/IFEVAL_SCORED_{mid}.json',V/'NATIVE_EQUIVALENCE.json',V/'IDENTITY_REFRESH.json']}
save('PETITGPT_SMOKE_BRIDGE.json',{'model':mid,'status':'REUSED_VERIFIED','old_fingerprint':old['fingerprint'],'new_fingerprint':fp,'scope':'Accept only the exact eight original PetitGPT smoke records under native implementation equivalence; original fingerprints preserved','evidence_files':evidence,'old_execution':True,'new_model_loads':0,'new_generations':0,'new_checker_passes':0})
print('IDENTITIES PASS; NATIVE SMOKE REUSED_VERIFIED; new fingerprint',fp,flush=True)
