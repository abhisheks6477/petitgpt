from pathlib import Path
import difflib
V=Path(__file__).resolve().parents[1];R=V.parents[1];prefix='ifeval/completion_v2/'
old=(R/'runtime/generate_ifeval.py').read_text();s=old
s=s.replace("R=Path(__file__).resolve().parents[1];S=R.parent/'source'", "V=Path(__file__).resolve().parents[1];R=V.parents[1];S=R.parent/'source'\nsys.path.insert(0,str(R/'runtime'))\nfrom effective_config import prepare,invoke,identity\nimport fcntl,traceback\n")
s=s.replace("approval=read('protocol/IFEVAL_TEMPLATE_RESOLUTION.json')", "lock=(V/'EVALUATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n approval=read('protocol/IFEVAL_TEMPLATE_RESOLUTION.json')")
a=s.index(' paths=[Path(__file__)');b=s.index(' reports=[]',a)
s=s[:a]+''' freeze=read('ifeval/completion_v2/CODE_FREEZE.json');fp=freeze['fingerprint']
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
''' +s[b:]
s=s.replace("reports=[]", "reports=[read('smoke/IFEVAL_SCORED_PetitGPT-alpha075.json')] if a.stage=='smoke' else []")
s=s.replace(" for m in models:\n  mid=m['id'];native=", " for m in models:\n  if a.stage=='smoke' and m['id']=='PetitGPT-alpha075':continue\n  mid=m['id'];native=")
a=s.index('  if not native:\n');b=s.index('  actual_inputs=',a)
s=s[:a]+'''  if not native:
   effective,config_evidence=prepare(model.generation_config)
   frozen_config=read(f'ifeval/completion_v2/EFFECTIVE_CONFIG_{mid}.json')
   assert config_evidence['effective_config_id']==frozen_config['effective_config_id']
   assert config_evidence['raw_config_id']==frozen_config['raw_config_id']
   config_id=config_evidence['effective_config_id']
  else:
   config_id=digest(canon({'native_accepted_generate':freeze['native_settings']}))
''' +s[b:]
s=s.replace("out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=1280,do_sample=False,temperature=0.0,num_beams=1,use_cache=True,repetition_penalty=1.0)","out=invoke(model,effective,config_evidence,input_ids=x,attention_mask=torch.ones_like(x))")
s=s.replace("f'ifeval/generations/{mid}.jsonl'", "f'ifeval/completion_v2/generations/{mid}.jsonl'")
s=s.replace("f'smoke/ifeval_{mid}.jsonl'", "f'ifeval/completion_v2/smoke/ifeval_{mid}.jsonl'")
s=s.replace("  if target.exists():\n   complete", "  if target.exists():\n   payload=target.read_bytes()\n   if payload and not payload.endswith(b'\\n'):\n    partial=payload.rsplit(b'\\n',1)[-1];(V/f'{a.stage}_{mid}.partial_evidence').write_bytes(partial);raise RuntimeError('Partial JSONL record retained; refusing automatic resume: '+str(target))\n   complete")
s=s.replace("row=actual_inputs[i];assert g['key']", "row=actual_inputs[i];assert g['effective_config_id']==config_id and g['model_identity_sha256']==digest(canon(m));assert g['key']")
s=s.replace("save(f'ifeval/RESUME_{mid}.json'", "save(f'ifeval/completion_v2/RESUME_{mid}_{a.stage}.json'")
s=s.replace("'fingerprint':fp}\n    f.write", "'fingerprint':fp,'effective_config_id':config_id,'model_identity_sha256':digest(canon(m))}\n    f.write")
s=s.replace("f'smoke/IFEVAL_DETERMINISM_{mid}.json'", "f'ifeval/completion_v2/smoke/IFEVAL_DETERMINISM_{mid}.json'")
s=s.replace("'ifeval/completion_v1/PROGRESS.json'", "'ifeval/completion_v2/PROGRESS.json'")
s=s.replace('del model,scorer;', 'del model,scorer,tok,generate;')
s=s.replace("str(R/'runtime/verify_ifeval.py')", "str(V/'runtime/verify_ifeval_v2.py')")
s=s.replace("f'ifeval/RESULTS_{mid}.json'", "f'ifeval/completion_v2/RESULTS_{mid}.json'")
s=s.replace("f'smoke/IFEVAL_SCORED_{mid}.json'", "f'ifeval/completion_v2/smoke/IFEVAL_SCORED_{mid}.json'")
s=s.replace(" save('smoke/IFEVAL.json' if a.stage=='smoke' else 'ifeval/RESULTS.json',{'status':'PASS','stage':a.stage,'fingerprint':fp,'results':reports})",''' if a.stage=='smoke':
  receipts=[{'model':'PetitGPT-alpha075','execution_fingerprint':bridge['old_fingerprint'],'acceptance':'REUSED_VERIFIED','evidence_files':{str(V/'PETITGPT_SMOKE_BRIDGE.json'):sha(V/'PETITGPT_SMOKE_BRIDGE.json')}}]
  for m in models[1:]:
   mid=m['id'];ps=[V/f'smoke/ifeval_{mid}.jsonl',V/f'smoke/IFEVAL_DETERMINISM_{mid}.json',V/f'smoke/IFEVAL_SCORED_{mid}.json',V/f'verifier/{mid}_smoke.jsonl']
   assert read(f'ifeval/completion_v2/smoke/IFEVAL_DETERMINISM_{mid}.json')['pass']
   receipts.append({'model':mid,'execution_fingerprint':fp,'acceptance':'NEW_EXECUTION_PASS','evidence_files':{str(p):sha(p) for p in ps}})
  save('ifeval/completion_v2/smoke/IFEVAL.json',{'status':'PASS','stage':a.stage,'continuation_fingerprint':fp,'execution_identities':receipts,'results':reports})
 else:save('ifeval/completion_v2/RESULTS.json',{'status':'PASS','stage':a.stage,'fingerprint':fp,'results':reports})''')
s=s.replace("if __name__=='__main__':main()",'''if __name__=='__main__':
 try:main()
 except Exception:
  save('ifeval/completion_v2/STOP_RUNTIME.json',{'status':'PARTIAL_STOPPED','stage':sys.argv[1:],'exact_error':traceback.format_exc(),'action':'Stopped; completed rows retained. No retry or protocol change.'});raise''')
(V/'runtime/generate_ifeval_v2.py').write_text(s)
vo=(R/'runtime/verify_ifeval.py').read_text();v=vo.replace("R=Path(__file__).resolve().parents[1]","V=Path(__file__).resolve().parents[1];R=V.parents[1]")
for a,b in [("f'ifeval/generations/{mid}.jsonl'","f'ifeval/completion_v2/generations/{mid}.jsonl'"),("f'smoke/ifeval_{mid}.jsonl'","f'ifeval/completion_v2/smoke/ifeval_{mid}.jsonl'"),("f'ifeval/verifier/{mid}_{stage}.jsonl'","f'ifeval/completion_v2/verifier/{mid}_{stage}.jsonl'"),("f'ifeval/RESULTS_{mid}.json'","f'ifeval/completion_v2/RESULTS_{mid}.json'"),("f'smoke/IFEVAL_SCORED_{mid}.json'","f'ifeval/completion_v2/smoke/IFEVAL_SCORED_{mid}.json'")]:v=v.replace(a,b)
# Fixtures are preserved as source history, but this routing entrypoint cannot execute them.
v=v.replace("choices=['fixtures','smoke','full']","choices=['smoke','full']")
(V/'runtime/verify_ifeval_v2.py').write_text(v)
(V/'REPAIR.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='runtime/generate_ifeval.py',tofile='ifeval/completion_v2/runtime/generate_ifeval_v2.py'))+''.join(difflib.unified_diff(vo.splitlines(True),v.splitlines(True),fromfile='runtime/verify_ifeval.py',tofile='ifeval/completion_v2/runtime/verify_ifeval_v2.py')))
compile(s,'generate_ifeval_v2.py','exec');compile(v,'verify_ifeval_v2.py','exec')
print('NEW routing scripts created; originals unchanged')
