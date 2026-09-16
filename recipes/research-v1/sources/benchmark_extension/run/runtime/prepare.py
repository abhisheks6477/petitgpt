from pathlib import Path
import os,sys,json,hashlib,ast,importlib.util,collections,platform,importlib.metadata,subprocess
R=Path(__file__).resolve().parents[1]; S=R.parent/'source'
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HOME']=str(R/'cache/hf');os.environ['HF_DATASETS_CACHE']=str(R/'cache/datasets')
sys.path.insert(0,str(S/'recipes/research-v1/sources/native_inference'))
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from src.chat_template import load_chat_tokenizer,encode_prompt

def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): (R/p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def jsonl(p,v):
 if (R/p).exists():
  assert (R/p).read_text()==''.join(canon(x)+'\n' for x in v);return
 with (R/p).open('x') as f:
  for x in v:f.write(canon(x)+'\n')
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
h=module(R/'upstream/lm_eval/tasks/hellaswag/utils.py','pinned_hellaswag')
meta=json.loads((R/'protocol/DATA_DOWNLOADS.json').read_text());rows={}
for name,n in [('arc_challenge',1172),('hellaswag',10042),('ifeval',541)]:
 m=meta[name];p=Path(m['source_file']);assert sha(p)==m['source_sha256']
 raw=([json.loads(s) for s in p.read_text().splitlines()] if name=='ifeval' else pq.read_table(p).to_pylist());assert len(raw)==n,(name,len(raw))
 normal=[]
 if name=='hellaswag':
  class MapAdapter:
   def map(self,fn):return [fn(d) for d in raw]
  mapped=h.process_docs(MapAdapter())
 for i,d in enumerate(raw):
  if name=='arc_challenge':
   cs=d['choices']['text'];ls=d['choices']['label'];assert len(cs)==len(ls)==len(set(ls));gold=ls.index(str(d['answerKey']))
   x={'row_index':i,'id':str(d['id']),'prompt':f"Question: {d['question']}\nAnswer:",'choices':cs,'gold':gold}
  elif name=='hellaswag':
   pd=mapped[i];cs=pd['choices'];gold=pd['gold'];assert len(cs)==4
   # Execute upstream map itself, then independently check its published preprocessing.
   def equivalent(t):
    import re
    return re.sub(r'\[.*?\]','',t.strip().replace(' [title]','. ')).replace('  ',' ')
   assert pd['query']==equivalent(d['activity_label']+': '+d['ctx_a']+' '+d['ctx_b'].capitalize())
   assert cs==[equivalent(t) for t in d['endings']]
   x={'row_index':i,'id':f'hellaswag:validation:{i}','source_ind':d['ind'],'prompt':pd['query'],'choices':cs,'gold':gold}
  else:
   assert type(d['key']) is int and isinstance(d['prompt'],str) and d['prompt']
   assert isinstance(d['instruction_id_list'],list) and d['instruction_id_list'] and all(isinstance(t,str) for t in d['instruction_id_list'])
   assert isinstance(d['kwargs'],list) and len(d['kwargs'])==len(d['instruction_id_list']) and all(isinstance(t,dict) for t in d['kwargs'])
   x={k:d[k] for k in ['key','prompt','instruction_id_list','kwargs']}
  if name!='ifeval':assert x['prompt'] and all(isinstance(t,str) and len(t)>0 for t in cs) and 0<=gold<len(cs)
  normal.append(x)
 ids=[x['key'] if name=='ifeval' else x['id'] for x in normal];assert len(set(ids))==n
 rows[name]=normal;jsonl(f'{name}/normalized_rows.jsonl',normal)
 m.update(rows=n,split='test' if name=='arc_challenge' else 'validation' if name=='hellaswag' else 'train',ordering='source physical/original row order; unchanged',normalized_rows_sha256=sha(R/name/'normalized_rows.jsonl'),canonical_rows_sha256=digest(canon(normal)),row_order_sha256=digest(canon(ids)))
 if name=='ifeval':m.update(instruction_id_distribution=dict(collections.Counter(t for d in normal for t in d['instruction_id_list'])),total_instructions=sum(len(d['instruction_id_list']) for d in normal))
 else:m['choice_count_distribution']=dict(collections.Counter(len(d['choices']) for d in normal))
 print('DATA PASS',name,n,flush=True)
save('protocol/DATA_IDENTITIES.json',meta)
# Exact unchanged pinned boundary method, extracted from reviewed public source.
b=S/'recipes/research-v1/sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2'
p=b/'source/harness/lm_eval/api/model.py';tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TemplateLM');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_encode_pair');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
class Encoding:
 backend='causal'
 pair=ns['_encode_pair']
 def __init__(self,t,native):self.tokenizer=t;self.native=native
 def tok_encode(self,text,add_special_tokens=False):
  assert add_special_tokens is False
  v=self.tokenizer.encode(text,add_special_tokens=False);return v.ids if self.native else v
models=json.loads((R/'protocol/MODEL_IDENTITIES.json').read_text());check={};icheck={}
for m in models:
 mid=m['id'];native=mid.startswith('Petit');tok=Tokenizer.from_file(m['tokenizer']) if native else AutoTokenizer.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False)
 for a in ['add_bos_token','add_eos_token']:
  if not native and hasattr(tok,a):setattr(tok,a,False)
 enc=Encoding(tok,native);check[mid]={}
 for name in ['arc_challenge','hellaswag']:
  encoded=[];max_len=0
  for d in rows[name]:
   pairs=[]
   for text in d['choices']:
    c,k=enc.pair(d['prompt'],' '+text);full=enc.tok_encode(d['prompt']+' '+text)
    assert c and k and c+k==full and max(full)<(32000 if native else 49152)
    assert len(full)<=2048,(name,mid,d['id'],len(full))
    max_len=max(max_len,len(full));pairs.append({'context_ids':c,'continuation_ids':k,'candidate_chars':len(text)})
   encoded.append({'row_index':d['row_index'],'id':d['id'],'row_sha256':digest(canon(d)),'pairs':pairs})
  out=f'{name}/encodings_{mid}.jsonl';jsonl(out,encoded);check[mid][name]={'status':'PASS','rows':len(encoded),'max_total_tokens':max_len,'file':out,'sha256':sha(R/out),'candidates':sum(len(x['pairs']) for x in encoded)}
  print('BOUNDARIES PASS',mid,name,max_len,flush=True)
 chat=load_chat_tokenizer(m['tokenizer']) if native else AutoTokenizer.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False)
 context=2048 if native else json.loads((Path(m['path'])/'config.json').read_text())['max_position_embeddings']
 inp=[]
 for d in rows['ifeval']:
  messages=[{'role':'user','content':d['prompt']}]
  if native:
   ids=encode_prompt(chat,messages,default_system=None,mode='full_context');assert ids==encode_prompt(chat,messages,default_system=None,mode='full_context');rendered=chat.decode(ids,skip_special_tokens=False);assert ids[:2]==[2,5] and ids[-1]==6
  else:
   ids=chat.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False);rendered=chat.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   assert ids==chat.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
  inp.append({'key':d['key'],'prompt_sha256':digest(d['prompt']),'rendered_prompt_sha256':digest(rendered),'prompt_token_ids':ids,'prompt_ids_sha256':digest(canon(ids)),'prompt_tokens':len(ids),'context':context,'budget':1280,'fits':len(ids)+1280<=context})
 out=f'ifeval/generation_inputs_{mid}.jsonl';jsonl(out,inp)
 fail=[{'key':d['key'],'prompt_tokens':d['prompt_tokens']} for d in inp if not d['fits']]
 icheck[mid]={'rows':len(inp),'context':context,'max_prompt_tokens':max(x['prompt_tokens'] for x in inp),'overflow_count':len(fail),'overflow_rows':fail,'file':out,'sha256':sha(R/out),'chat_template_sha256':sha(S/'recipes/research-v1/sources/native_inference/src/chat_template.py') if native else digest(chat.chat_template)}
 print('IFEVAL CONTEXT',mid,icheck[mid]['max_prompt_tokens'],len(fail),flush=True)
save('protocol/LIKELIHOOD_PREFLIGHT.json',check);save('protocol/IFEVAL_CONTEXT_PREFLIGHT.json',icheck)
policy=json.loads((S/'configs/research-v1/benchmark_protocol.json').read_text());policy['tasks']=[meta[n] for n in ['arc_challenge','hellaswag']];policy['prompt_template']={'arc_challenge':'Question: <exact official question>\nAnswer:','hellaswag':'pinned process_docs query'};policy['choice_text']='exact normalized candidate from task definition; ARC official text; HellaSwag pinned preprocessed ending';policy['acc_norm']='FP32 continuation sum / Python len(candidate text before leading delimiter); HellaSwag candidate is the pinned-preprocessed ending, not token count'
save('protocol/LIKELIHOOD_PROTOCOL.json',policy)
save('protocol/IFEVAL_PROTOCOL.json',{'harness_commit':'b954108c9baaaa934b4ad842033b31a97ee30816','messages':'one official verbatim user prompt; zero system/fewshot messages; assistant generation prefix','PetitGPT_formatter':'released native_inference/src/chat_template.py encode_prompt(default_system=None, mode=full_context); [BOS] <|user|> user text <|assistant|>','SmolLM_formatters':'pinned AutoTokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)','max_new_tokens':1280,'do_sample':False,'temperature':0,'num_beams':1,'repetition_penalty':1.0,'until':[],'canonical_eos':True,'generation_numeric':'FP32 parameters and forward, autocast/TF32 disabled, SDPA MATH','context_preflight':icheck,'full_generation_authorized_by_preflight':all(x['overflow_count']==0 for x in icheck.values()),'metrics':['prompt_level_strict_acc','inst_level_strict_acc','prompt_level_loose_acc','inst_level_loose_acc']})
import torch
save('protocol/ENVIRONMENT.json',{'python':platform.python_version(),'executable':sys.executable,'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','tokenizers','transformers','datasets','huggingface_hub','safetensors','pyarrow']},'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'historical_environment_modified':False})
print('PREPARATION COMPLETE',flush=True)
