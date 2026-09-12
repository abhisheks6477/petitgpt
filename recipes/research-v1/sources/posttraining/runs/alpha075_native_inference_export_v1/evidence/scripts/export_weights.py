import sys,json,hashlib,dataclasses,os
from pathlib import Path
R=Path('./runs/alpha075_native_inference_export_v1'); B=R/'petitgpt-alpha075-native-research-v1'; E=R/'evidence'
sys.path.insert(0,'.')
import torch
from sft.train_sft import load_ckpt
from src.model import GPT,gpt_config_from_checkpoint_dict,audit_gpt_parameter_count
from src.special_tokens import SPECIAL_TOKEN_IDS,assert_tokenizer_contract
from safetensors.torch import save_model,load_model
from safetensors import safe_open
assert sys.executable=='./.venv/bin/python' and sys.prefix=='./.venv'
bind=json.loads((E/'frozen/INPUT_BINDINGS.json').read_text())
ckpath=bind['source_checkpoint']['path']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert sha(ckpath)==bind['source_checkpoint']['sha256']
ck=load_ckpt(ckpath);cfg=gpt_config_from_checkpoint_dict(ck.get('config') or ck.get('cfg'))
for k,v in bind['config_required'].items(): assert getattr(cfg,k)==v,(k,getattr(cfg,k))
sd=ck['model']; assert not any(k.startswith('_orig_mod.') for k in sd)
model=GPT(cfg).eval(); model.load_state_dict(sd,strict=True)
audit=audit_gpt_parameter_count(model,cfg);assert audit['actual_total']==bind['unique_parameters']
assert model.tok_emb.weight is model.lm_head.weight
assert torch.equal(sd['tok_emb.weight'],sd['lm_head.weight'])
assert all(t.dtype==torch.float32 for t in sd.values())
source_checks=[]
for name,t in model.state_dict().items():
 assert t.dtype==sd[name].dtype and t.shape==sd[name].shape and torch.equal(t,sd[name]),name
 source_checks.append(name)
(B/'config.json').write_text(json.dumps(dataclasses.asdict(cfg),indent=2)+'\n')
save_model(model,str(B/'model.safetensors'))
with safe_open(str(B/'model.safetensors'),framework='pt',device='cpu') as f: aliases=f.metadata(); stored=list(f.keys())
reloaded=GPT(cfg).eval();load_model(reloaded,str(B/'model.safetensors'),strict=True,device='cpu')
assert reloaded.tok_emb.weight is reloaded.lm_head.weight
assert reloaded.tok_emb.weight.data_ptr()==reloaded.lm_head.weight.data_ptr()
assert audit_gpt_parameter_count(reloaded,cfg)['actual_total']==124635456
assert set(model.state_dict())==set(reloaded.state_dict())==set(sd)
comparisons=[]
for name,t in sd.items():
 other=reloaded.state_dict()[name]
 equal=t.dtype==other.dtype and t.shape==other.shape and torch.equal(t,other)
 comparisons.append({'name':name,'dtype':str(t.dtype),'shape':list(t.shape),'equal':equal,'max_absolute_difference':0.0 if equal else float((t-other).abs().max()),'value_sha256':hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()})
 assert equal,name
buffers=[]
for (name,t),(othername,other) in zip(model.named_buffers(),reloaded.named_buffers()):
 assert name==othername and torch.equal(t,other)
 buffers.append({'name':name,'dtype':str(t.dtype),'shape':list(t.shape),'equal':True,'persistent':False})
assert_tokenizer_contract(str(B/'tokenizer.json'))
(B/'special_tokens_map.json').write_text(json.dumps({'format':'descriptive native metadata; no HF AutoModel claim','tokens':SPECIAL_TOKEN_IDS},indent=2))
meta={'checkpoint_sha256':sha(ckpath),'export_sha256':sha(B/'model.safetensors'),'export_bytes':(B/'model.safetensors').stat().st_size,'config_sha256':sha(B/'config.json'),'tokenizer_sha256':sha(B/'tokenizer.json'),'config':dataclasses.asdict(cfg),'normalization':{'class':'RMSNorm','eps':1e-6},'cache':'native RoPE nonpersistent buffers; KV cache at 3 heads, absolute offsets; see byte-identical src/model.py','aliases':aliases,'stored_tensor_count':len(stored),'named_state_entries':len(sd),'unique_parameters':audit,'shared_parameter_object':True,'shared_storage':True,'source_strict_load':True,'source_to_constructed_comparisons':len(source_checks),'source_to_reloaded_comparisons':comparisons,'nonpersistent_buffer_comparisons':buffers,'counters':{'source_checkpoint_loads':1,'model_constructions':2,'gpu_model_loads':0,'tensor_comparisons':len(source_checks)+len(comparisons)+len(buffers),'standalone_witness_forwards':0,'generation_forwards':0,'completions':0}}
(E/'TENSOR_EQUALITY.json').write_text(json.dumps(meta,indent=2))
(E/'export_imports.json').write_text(json.dumps({k:getattr(v,'__file__',None) for k,v in sys.modules.items() if k.startswith(('src','sft'))},indent=2))
print(json.dumps({k:v for k,v in meta.items() if k not in ['source_to_reloaded_comparisons','nonpersistent_buffer_comparisons']},indent=2))
