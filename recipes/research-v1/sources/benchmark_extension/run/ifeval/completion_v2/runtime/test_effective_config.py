"""Config-only regressions against installed Transformers; no model or tensor execution."""
from pathlib import Path
import copy, inspect, json, hashlib, traceback
import transformers
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin
from effective_config import prepare, invoke, validate, checked_update, NEUTRAL, ABSENT, CANONICAL
V=Path(__file__).resolve().parents[1];R=V.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):(V/p).write_text(json.dumps(x,indent=2)+'\n')
checks=[]
def check(name,fn):
 fn();checks.append({'test':name,'status':'PASS'});print('PASS',name,flush=True)
def rejects(fn):
 try:fn()
 except (ValueError,TypeError):return
 raise AssertionError('Expected rejection')
models=json.loads((R/'protocol/MODEL_IDENTITIES.json').read_text())
try:
 assert transformers.__version__=='5.15.0'
 source={str(Path(inspect.getsourcefile(x))):sha(inspect.getsourcefile(x)) for x in [GenerationConfig,GenerationMixin]}
 save('INSTALLED_CONFIG_SOURCE.json',{'version':transformers.__version__,'sources':source,'functions':{name:inspect.getsource(fn) for name,fn in [('GenerationConfig.__init__',GenerationConfig.__init__),('GenerationConfig._get_default_generation_params',GenerationConfig._get_default_generation_params),('GenerationConfig.update',GenerationConfig.update),('GenerationMixin._prepare_generation_config',GenerationMixin._prepare_generation_config)]}})
 for m in models[1:]:
  p=Path(m['path'])/'generation_config.json';before=sha(p);raw=GenerationConfig.from_pretrained(m['path'],local_files_only=True);snapshot=copy.deepcopy(raw.to_dict())
  assert all(getattr(raw,k) is None for k in ['repetition_penalty','no_repeat_ngram_size','encoder_repetition_penalty','min_length'])
  old_failed=False
  try:assert raw.repetition_penalty==1.0
  except AssertionError:old_failed=True
  assert old_failed
  cfg,ev=prepare(raw);ev.update(model=m['id'],snapshot_path=str(p),snapshot_sha256=before,snapshot_json=json.loads(p.read_text()),old_none_assertion_reproduced=True)
  save('EFFECTIVE_CONFIG_'+m['id']+'.json',ev)
  check(m['id']+' actual local None defaults resolve',lambda:validate(cfg))
  class Receiver:
   generation_config=raw
   def generate(self,**kwargs):
    assert set(kwargs)=={'input_ids','attention_mask','generation_config'}
    assert kwargs['input_ids']=='sentinel-input' and kwargs['attention_mask']=='sentinel-mask'
    resolved,left=GenerationMixin._prepare_generation_config(self,kwargs['generation_config'])
    assert left=={} and resolved.to_dict()==ev['effective']
    validate(resolved);self.seen=kwargs['generation_config'].to_dict();return 'received'
  receiver=Receiver()
  check(m['id']+' production call exact effective config',lambda:exec("assert invoke(receiver,cfg,ev,input_ids='sentinel-input',attention_mask='sentinel-mask')=='received'"))
  assert receiver.seen==ev['effective']
  for key,value in NEUTRAL.items():
   neutral=copy.deepcopy(raw);setattr(neutral,key,value);prepare(neutral)
  check(m['id']+' explicit neutral controls accepted',lambda:None)
  for key,value in [('repetition_penalty',1.2),('repetition_penalty',0),('encoder_repetition_penalty',1.2),('no_repeat_ngram_size',2),('encoder_no_repeat_ngram_size',2),('min_length',1),('min_new_tokens',1),('length_penalty',.9),('token_healing',True),('use_mtp',True),('is_assistant',True)]:
   bad=copy.deepcopy(raw);setattr(bad,key,value);check(m['id']+' rejects explicit '+key+'='+str(value),lambda bad=bad:rejects(lambda:prepare(bad)))
  for key in ABSENT:
   bad=copy.deepcopy(raw);setattr(bad,key,['forbidden']);check(m['id']+' rejects enabled '+key,lambda bad=bad:rejects(lambda:prepare(bad)))
  for key in CANONICAL:
   bad=copy.deepcopy(raw);setattr(bad,key,999);check(m['id']+' rejects changed '+key,lambda bad=bad:rejects(lambda:prepare(bad)))
  override=copy.deepcopy(raw);override.do_sample=True;override.num_beams=4;override.max_new_tokens=40
  fixed,_=prepare(override);assert fixed.do_sample is False and fixed.num_beams==1 and fixed.max_new_tokens==1280
  check(m['id']+' frozen overrides intentional',lambda:None)
  for key,value in [('do_sample',True),('min_new_tokens',1),('repetition_penalty',1.2)]:
   changed=copy.deepcopy(cfg);setattr(changed,key,value);check(m['id']+' rejects postprepare '+key,lambda changed=changed:rejects(lambda:invoke(receiver,changed,ev,input_ids='sentinel-input',attention_mask='sentinel-mask')))
  check(m['id']+' unknown config kwargs rejected',lambda:rejects(lambda:checked_update(copy.deepcopy(cfg),{'no_such_generation_setting':123})))
  check(m['id']+' extra production kwargs rejected',lambda:rejects(lambda:invoke(receiver,cfg,ev,input_ids='sentinel-input',attention_mask='sentinel-mask',do_sample=True)))
  assert sha(p)==before and raw.to_dict()==snapshot and receiver.generation_config.to_dict()==snapshot
  check(m['id']+' raw object and file unchanged',lambda:None)
 save('CONFIG_ONLY_REGRESSIONS.json',{'status':'PASS','tests':checks,'count':len(checks),'model_loads':0,'tensor_executions':0,'installed_resolver_used':True,'expected_warning':'temperature=0.0 ignored with do_sample=False; retained in config-only.log, no global warning suppression','helper_sha256':sha(V/'runtime/effective_config.py')})
except Exception:
 save('CONFIG_ONLY_REGRESSIONS.json',{'status':'FAIL','tests':checks,'error':traceback.format_exc()});raise
