from __future__ import annotations
import os,sys,json,hashlib,pathlib,ast,types,collections,logging
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(REPO))
import torch,numpy as np
from torch.nn.attention import sdpa_kernel, SDPBackend
from likelihood_checks import continuation_logprobs

def primary_policy():
 torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False

from tokenizers import Tokenizer
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm

def dump(p,x):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def read(p): return json.loads((ROOT/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def jsonl(p):return [json.loads(s) for s in (ROOT/p).read_text().splitlines()]
def write_jsonl(p,rows):
 with (ROOT/p).open('w') as f:
  for x in rows:f.write(canonical(x)+'\n')
def check_env():
 import platform,importlib.metadata as m,subprocess
 assert sys.executable=='./.venv/bin/python' and sys.prefix=='./.venv'
 assert platform.python_version()=='3.10.12'
 for k,v in {'torch':'2.11.0+cu126','numpy':'2.2.6','tokenizers':'0.22.2','transformers':'5.15.0','datasets':'5.0.1','huggingface_hub':'1.27.0','safetensors':'0.8.0','pyarrow':'25.0.1'}.items():assert m.version(k)==v
 return {'sys_executable':sys.executable,'sys_prefix':sys.prefix,'python':platform.python_version(),'versions':{k:m.version(k) for k in ['torch','numpy','tokenizers','transformers','datasets','huggingface_hub','safetensors','pyarrow']},'lm_eval':'not installed; native protocol-compatible evaluator','cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'parameters':'FP32','forward':'FP32; autocast disabled; math SDPA enforced inside forward','log_softmax':'FP32','batch_size':1,'backend':{'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'memory_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),'math_sdp':torch.backends.cuda.math_sdp_enabled(),'float32_matmul_precision':torch.get_float32_matmul_precision()},'note':'Enabled backend policy does not identify actual dispatched kernels.'}

def extract(rel,cls,names,namespace):
 tree=ast.parse((ROOT/'source/harness'/rel).read_text());body=tree.body
 if cls:body=next(x for x in body if isinstance(x,ast.ClassDef) and x.name==cls).body
 nodes=[x for x in body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in names]
 assert len(nodes)==len(names)
 # Preserve complete function/class AST; only defer annotations and omit containing module imports.
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),rel,'exec'),namespace)
 return [namespace[n] for n in names]
NS={'torch':torch,'F':torch.nn.functional,'np':np,'collections':collections,'tqdm':tqdm,'eval_logger':logging.getLogger('reference')}
encode_reference,=extract('lm_eval/api/model.py','TemplateLM',['_encode_pair'],NS)
extract('lm_eval/models/utils.py',None,['Collator'],NS)
extract('lm_eval/models/utils_hf.py',None,['pad_and_concat'],NS)
select_reference,ll_reference=extract('lm_eval/models/huggingface.py','HFLM',['_select_cont_toks','_loglikelihood_tokens'],NS)

class Encoding:
 backend='causal'
 def __init__(self,m):
  self.native=m['id']=='PetitGPT-alpha075'
  if self.native:self.tokenizer=Tokenizer.from_file(m['tokenizer'])
  else:
   self.tokenizer=AutoTokenizer.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False)
   for attr in ['add_bos_token','add_eos_token']:
    if hasattr(self.tokenizer,attr):setattr(self.tokenizer,attr,False)
 def tok_encode(self,text,add_special_tokens=False):
  assert add_special_tokens is False
  x=self.tokenizer.encode(text,add_special_tokens=False)
  return x.ids if self.native else x
 def pair(self,ctx,cont):return encode_reference(self,ctx,cont)

class Scorer:
 backend='causal';batch_size=1;rank=0;logits_cache=False;max_length=2048;device='cuda';softmax_dtype=torch.float32
 _select_cont_toks=select_reference
 def __init__(self,m):
  check_env();self.identity=m;self.forward_calls=0;self.model=None
  if m['id']=='PetitGPT-alpha075':
   from src.model import GPT,gpt_config_from_checkpoint_dict
   from sft.train_sft import load_ckpt
   ck=load_ckpt(m['path']);cfg=gpt_config_from_checkpoint_dict(ck.get('config') or ck.get('cfg'))
   self.model=GPT(cfg);self.model.load_state_dict(ck['model'],strict=True);self.config=vars(cfg)
   assert self.model.tok_emb.weight is self.model.lm_head.weight
   del ck
  else:
   self.model=AutoModelForCausalLM.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='sdpa')
   assert self.model.config._attn_implementation=='sdpa'
   assert self.model.get_input_embeddings().weight is self.model.get_output_embeddings().weight
   self.config=self.model.config.to_dict()
  self.model.requires_grad_(False);self.model.eval();self.model.to('cuda')
  self.parameters=sum(p.numel() for p in self.model.parameters());assert self.parameters==m['expected_unique_parameters']
  assert all(p.dtype==torch.float32 for p in self.model.parameters())
  self.cache_hook=types.SimpleNamespace(add_partial=lambda *a:None)
 def _model_call(self,x,**kwargs):
  assert not kwargs and x.shape[0]==1
  primary_policy()
  self.forward_calls+=1
  with torch.inference_mode(),torch.autocast('cuda',enabled=False),sdpa_kernel([SDPBackend.MATH]):
   assert not torch.is_autocast_enabled('cuda')
   assert torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled()
   if self.identity['id']=='PetitGPT-alpha075':out=self.model(x)
   else:out=self.model(input_ids=x,use_cache=False).logits
   assert out.dtype==torch.float32, 'Actual returned logits must be FP32 without casting'
   return out
 def score(self,c,k,pad=0,return_logits=False):
  assert c and k and len(c)+len(k)<=2048
  inp=torch.tensor((c+k)[:-1]+[0]*pad,device='cuda').unsqueeze(0)
  logits=self._model_call(inp)[0]
  selected=logits[len(c)-1:len(c)+len(k)-1].float().log_softmax(-1)
  tokens=selected.gather(1,torch.tensor(k,device='cuda').unsqueeze(1)).squeeze(1)
  value=float(tokens.sum())
  assert np.isfinite(value)
  return (value,logits,tokens.cpu().tolist()) if return_logits else value
 def reference(self,c,k):return ll_reference(self,[(None,c,k)],disable_tqdm=True)[0][0]
 def generate_until(self,*a,**k):raise NotImplementedError('Generation is outside this likelihood-only run')
 def loglikelihood_rolling(self,*a,**k):raise NotImplementedError('Rolling likelihood is outside this run')

def metric(scores,denoms,gold):
 assert len(scores)==len(denoms) and all(d>0 for d in denoms)
 p=int(np.argmax(scores));pn=int(np.argmax(np.array(scores)/np.array(denoms)))
 return {'prediction_acc':p,'prediction_acc_norm':pn,'acc':int(p==gold),'acc_norm':int(pn==gold)}
