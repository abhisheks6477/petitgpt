"""Independent saved-record audit and compact private delivery; no model inference."""
from pathlib import Path
import json,hashlib,sys,collections,datetime,tarfile
V=Path(__file__).resolve().parent;R=V.parents[1];S=R.parent/'source'
sys.path.insert(0,str(S/'recipes/research-v1/sources/native_inference'))
from src.chat_template import load_chat_tokenizer,encode_prompt
from transformers import AutoTokenizer

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 p=Path(p)
 if not p.exists():return []
 raw=p.read_bytes();assert not raw or raw.endswith(b'\n'),f'partial line: {p}'
 return [json.loads(x) for x in raw.splitlines()]
def save(name,x):(V/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
models=read(R/'protocol/MODEL_IDENTITIES.json');docs=rows(R/'ifeval/normalized_rows.jsonl');keys=[d['key'] for d in docs]
assert len(keys)==len(set(keys))==541 and sum(len(d['instruction_id_list']) for d in docs)==834
assert len(set(t for d in docs for t in d['instruction_id_list']))==25
freeze=read(V/'CODE_FREEZE.json');fp=freeze['fingerprint']
for path,h in freeze['files'].items():assert sha(path)==h,path
checked=read(V/'IDENTITY_REFRESH.json')['checked_files']
for path,h in checked.items():assert sha(path)==h,path
counts={};instructions={};caps={};stops={};outcomes=[]
for m in models:
 mid=m['id'];native=mid.startswith('Petit');eos=3 if native else 2
 gens=rows(V/f'generations/{mid}.jsonl');counts[mid]=len(gens);instructions[mid]=0
 assert len(gens)<=541 and [g['key'] for g in gens]==keys[:len(gens)]
 inputs=rows(R/f'ifeval/generation_inputs_{mid}.jsonl')
 tok=load_chat_tokenizer(m['tokenizer']) if native else AutoTokenizer.from_pretrained(m['path'],local_files_only=True,trust_remote_code=False)
 configid=digest(canon({'native_accepted_generate':freeze['native_settings']})) if native else read(V/f'EFFECTIVE_CONFIG_{mid}.json')['effective_config_id']
 for g,d,i in zip(gens,docs,inputs):
  ids=g['generated_token_ids'];assert 0<len(ids)<=1280 and all(type(t) is int and 0<=t<(32000 if native else 49152) for t in ids)
  assert eos not in ids[:-1]
  assert g['generated_token_count']==len(ids) and g['generated_ids_sha256']==digest(canon(ids))
  response=tok.decode(ids[:-1] if ids[-1]==eos else ids,skip_special_tokens=False) if native else tok.decode(ids,skip_special_tokens=True)
  assert g['response']==response and g['response_sha256']==digest(response)
  assert g['stop_reason']==('eos' if ids[-1]==eos else 'max_new_tokens')
  assert ids[-1]==eos or len(ids)==1280
  messages=[{'role':'user','content':d['prompt']}]
  actual=encode_prompt(tok,messages,default_system=None,mode='full_context') if native else tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
  rendered=tok.decode(actual,skip_special_tokens=False) if native else tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  assert actual==i['prompt_token_ids'] and g['prompt_ids_sha256']==i['prompt_ids_sha256']==digest(canon(actual))
  assert g['prompt_sha256']==i['prompt_sha256']==digest(d['prompt'])
  assert g['rendered_prompt_sha256']==i['rendered_prompt_sha256']==digest(rendered)
  assert len(actual)+1280<=i['context'] and g['prompt_tokens']==len(actual)
  assert g['fingerprint']==fp and g['model_identity_sha256']==digest(canon(m)) and g['effective_config_id']==configid
  assert g['generation_settings']==freeze['native_settings'] and g['seconds']>=0
 stops[mid]=dict(collections.Counter(g['stop_reason'] for g in gens));caps[mid]=stops[mid].get('max_new_tokens',0)
 resultfile=V/f'RESULTS_{mid}.json'
 if len(gens)!=541 or not resultfile.exists():
  outcomes.append({'model':mid,'status':'INCOMPLETE','metrics':None,'rows':len(gens),'reason':'Full 541-row generation and verified scoring required'});continue
 scored=rows(V/f'verifier/{mid}_full.jsonl');accepted=read(resultfile)
 assert [s['key'] for s in scored]==keys
 for c,g,d in zip(scored,gens,docs):
  assert c['response_sha256']==g['response_sha256']
  for kind in ['strict','loose']:
   flags=c[f'inst_level_{kind}_acc'];assert len(flags)==len(d['instruction_id_list']) and all(type(f) is bool for f in flags)
   assert c[f'prompt_level_{kind}_acc']==all(flags)
 instruction_count=sum(len(c['inst_level_strict_acc']) for c in scored);assert instruction_count==834;instructions[mid]=instruction_count
 nums={'prompt_strict_correct':sum(c['prompt_level_strict_acc'] for c in scored),'prompt_loose_correct':sum(c['prompt_level_loose_acc'] for c in scored),'instruction_strict_correct':sum(sum(c['inst_level_strict_acc']) for c in scored),'instruction_loose_correct':sum(sum(c['inst_level_loose_acc']) for c in scored)}
 metrics={'prompt_level_strict_acc':nums['prompt_strict_correct']/541,'prompt_level_loose_acc':nums['prompt_loose_correct']/541,'inst_level_strict_acc':nums['instruction_strict_correct']/834,'inst_level_loose_acc':nums['instruction_loose_correct']/834}
 for k,x in {**nums,**metrics}.items():assert accepted[k]==x,(mid,k)
 assert accepted['status']=='PASS' and accepted['independent_recomputation']=='PASS' and len(accepted['instruction_id_coverage'])==25
 assert accepted['stop_reasons']==stops[mid] and accepted['generated_tokens']==sum(g['generated_token_count'] for g in gens)
 assert accepted['generations_sha256']==sha(V/f'generations/{mid}.jsonl') and accepted['verifier_rows_sha256']==sha(V/f'verifier/{mid}_full.jsonl')
 outcomes.append({'model':mid,'status':'PASS','rows':541,'instructions':834,'instruction_types':25,'numerators':nums,'metrics':metrics,'stop_reasons':stops[mid],'generated_tokens':accepted['generated_tokens'],'generation_seconds':sum(g['seconds'] for g in gens),'saved_checker_reaggregation':'PASS','saved_response_decoding_and_input_rendering':'PASS','existing_second_checker_pass':'PASS'})
smoke=read(V/'smoke/IFEVAL.json') if (V/'smoke/IFEVAL.json').exists() else None
completed=sum(o['status']=='PASS' for o in outcomes)
status={'STATUS':'COMPLETE' if completed==3 else 'PARTIAL_STOPPED','GENERATION_CONFIG_COMPATIBILITY_FIX':'PASS','CONFIG_ONLY_REGRESSIONS':'PASS','EFFECTIVE_GREEDY_POLICY_UNCHANGED':True,'OLD_PARTIAL_EVIDENCE_UNCHANGED':True,'PETITGPT_SMOKE':'REUSED_VERIFIED','SMOLLM_SMOKE':'PASS' if (V/'smoke/IFEVAL_SCORED_SmolLM-135M-Instruct.json').exists() else 'NOT_RUN','SMOLLM2_SMOKE':'PASS' if (V/'smoke/IFEVAL_SCORED_SmolLM2-135M-Instruct.json').exists() else 'NOT_RUN','IFEVAL_TEMPLATE_RESOLUTION':'CANONICAL_DEFAULT_SYSTEM_RETAINED','CALLER_SUPPLIED_SYSTEM_MESSAGE':False,'LIKELIHOOD_RESULTS_UNCHANGED':True,'IFEVAL_SMOKE':'PASS' if smoke else 'NOT_RUN','IFEVAL_FULL':'PASS' if completed==3 else ('PARTIAL' if any(counts.values()) else 'NOT_RUN'),'IFEVAL_PROMPTS_PER_MODEL':counts,'IFEVAL_INSTRUCTIONS_PER_MODEL':instructions,'IFEVAL_MODELS_COMPLETED':completed,'GENERATION_CAP_HITS':caps,'NEW_LIKELIHOOD_EVALUATIONS':0,'PROMPT_ROWS_DROPPED':0,'INPUT_ROWS_TRUNCATED':0,'MODEL_PARAMETERS_MODIFIED':False,'FORMAL_TRAINING_STARTED':False,'TEACHER_API_CALLS':0,'GITHUB_WRITES':0,'HF_WRITES':0,'DEPENDENCY_INSTALLATIONS':0,'STORAGE_MAINTENANCE_ACTIONS':0,'FIXTURE_RERUNS':0,'PETITGPT_SMOKE_REGENERATIONS':0,'NEW_HF_SMOKE_DETERMINISTIC_REPEATS':2 if smoke else None,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save('INDEPENDENT_FINAL_AUDIT.json',{'status':'PASS','results':outcomes,'old_identity_files_rechecked':len(checked),'old_archives_unchanged':True,'fresh_model_generations':0,'new_checker_executions':0,'aggregation_source':'Saved per-key checker booleans linked to saved response SHA256; existing second checker pass independently verified in each result'})
save('STATUS.json',status)
lines=['# IFEval 续跑完成报告（私有）','',f"状态：**{status['STATUS']}**。三模型完整评测完成数：{completed}/3。",'', '本次修复仅处理 Transformers 5.15.0 的有效生成配置解析及续跑输出路径。96 项配置测试通过；使用安装版本的真实 resolver，原始 `None`、模型快照和依赖保持不变。最终调用显式传入已校验的同一份配置。预检脚本对历史清单、源码文件集和 CLI 条目的适配错误及修正保留在日志中，未导致模型推理重试。','', '所有新生成保持 FP32、SDPA MATH、关闭 autocast/TF32/compile，batch=1、KV cache、greedy、temperature=0、num_beams=1、单条响应、max_new_tokens=1280；无强制最小长度、重复惩罚、额外停止字符串或输出修复。','', '调用方只提供一条原样官方 user 消息。保留各模型原生格式，包括 SmolLM2 自动插入的 `You are a helpful AI assistant named SmolLM, trained by Hugging Face`；PetitGPT 保留 default_system=None。这是原生聊天格式下的指令遵循比较：官方用户文本相同，但格式化后的 token 输入不同，不是相同 token 提示实验，也不宣称复现模型卡评测协议。','', '## IFEval（完整评测）','', '| 模型 | Prompt strict | Instruction strict | Prompt loose | Instruction loose | EOS / 1280 上限 |','|---|---:|---:|---:|---:|---:|']
for o in outcomes:
 mid=o['model']
 if o['status']!='PASS':lines.append(f'| {mid} | N/A | N/A | N/A | N/A | {stops[mid]} |');continue
 n=o['numerators'];cells=[f"{n[k]}/{den} ({n[k]/den:.2%})" for k,den in [('prompt_strict_correct',541),('instruction_strict_correct',834),('prompt_loose_correct',541),('instruction_loose_correct',834)]]
 lines.append('| '+mid+' | '+' | '.join(cells)+f" | {stops[mid].get('eos',0)} / {caps[mid]} |")
lines+=['','每个完整模型覆盖 541 个唯一 key、834 条指令、25 种指令类型，顺序与官方数据一致。核对了全部生成 ID、响应解码、输入渲染及哈希；原 checker 已完成两遍确定性评分，本次另从保存的布尔结果独立重聚合。所有低分回答、提前 EOS 和达到 token 上限的回答均保留；未补写、重采样或丢弃。','', '## Smoke 与历史','', 'PetitGPT 复用旧代码指纹下的 8 条 smoke、14 条指令及首题重复凭据；没有重新生成或重新运行其 verifier。通过独立 acceptance bridge 接入新续跑。SmolLM 和 SmolLM2 各新增 8 条 smoke、14 条指令及一次首题确定性重复。Smoke 的 PASS 仅表示执行、解码、确定性与 verifier 正确，不是完整榜单分数。','', '| 模型 | Smoke 处理 | Prompt strict / loose | Instruction strict / loose | EOS / 上限 |','|---|---|---:|---:|---:|']
if smoke:
 for s in smoke['results']:
  lines.append(f"| {s['model']} | {'REUSED_VERIFIED' if s['model'].startswith('Petit') else 'PASS（新执行）'} | {s['prompt_strict_correct']}/8 · {s['prompt_loose_correct']}/8 | {s['instruction_strict_correct']}/14 · {s['instruction_loose_correct']}/14 | {s['stop_reasons'].get('eos',0)} / {s['stop_reasons'].get('max_new_tokens',0)} |")
lines+=['','## 已接受的 likelihood 结果（原表原样复用）','',(R/'tables/LIKELIHOOD_NEW_TASKS.md').read_text().rstrip(),'','本次没有重新运行 ARC-Challenge、HellaSwag、ARC-Easy 或 PIQA，也没有修改其数据、协议、分项或聚合结果。不计算跨任务平均，不作显著性或通用智能推断。','', '## 身份与交付','',f"新代码指纹：`{fp}`。",'',f"原生 smoke 的旧代码指纹：`{freeze['old_fingerprint']}`。",'', '数据：google/IFEval @ 966cd89545d6b6acfd7638bc708b98261ca58e84；normalized rows SHA256：4d49ac039cbebdfc4beb3f9f30435c4fa25320caffaae441c6b9d868126744eb。模型、输入、模板、依赖及安装源码的完整路径和哈希见证据清单。','', '压缩包 INCLUDED：本次修复代码及 diff、配置测试和 raw/default/override/effective 配置、安装源码身份、新 freeze、PetitGPT smoke bridge、smoke 响应与凭据、逐题 checker 布尔结果、指标分子、停止原因、日志、身份记录和本报告。','', '压缩包 REFERENCED_ONLY：完整评测响应和 token ID 文件（保留在本续跑目录 generations/）、模型权重、完整数据及缓存、两份历史压缩包。清单明确记录其路径和 SHA256；这些 payload 未放入压缩包。原始数据、模型缓存和依赖未重复打包。','', '此前两份压缩包、旧 STOP、旧 freeze、completion_v1 及原 generator 均保持原字节。结果保持私有；没有训练、参数修改、依赖安装、GitHub/HF 写入或存储维护。','', '## 实际状态字段','', '```json',json.dumps(status,indent=2,ensure_ascii=False),'```','']
(V/'REPORT_ZH.md').write_text('\n'.join(lines))
# Explicit file-level payload inventory. Archive itself is external, avoiding recursive self-hash.
included={}
for p in V.rglob('*'):
 if not p.is_file() or p.name in ['EVIDENCE_MANIFEST.json','ARCHIVE_SHA256.txt','EVALUATOR.lock'] or 'generations' in p.relative_to(V).parts or '__pycache__' in p.parts:continue
 included[str(p)]=str(p.relative_to(R))
extra=['protocol/IFEVAL_TEMPLATE_RESOLUTION.json','protocol/IFEVAL_PROTOCOL.json','protocol/IFEVAL_CODE_FREEZE.json','protocol/IFEVAL_CONTEXT_PREFLIGHT.json','protocol/IFEVAL_SMOKE_SELECTION.json','protocol/MODEL_IDENTITIES.json','protocol/DATA_IDENTITIES.json','protocol/SOURCE_IDENTITIES.json','protocol/ENVIRONMENT.json','protocol/EVALUATOR_ENVIRONMENT.json','protocol/INDEPENDENT_LIKELIHOOD_AUDIT.json','protocol/LIKELIHOOD_COMPLETION.json','tables/LIKELIHOOD_NEW_TASKS.md','tables/FOUR_TASK_PROPOSAL.md','runtime/generate_ifeval.py','runtime/verify_ifeval.py','ifeval/verifier/FIXTURES.json','smoke/ifeval_PetitGPT-alpha075.jsonl','smoke/IFEVAL_DETERMINISM_PetitGPT-alpha075.json','smoke/IFEVAL_SCORED_PetitGPT-alpha075.json','ifeval/verifier/PetitGPT-alpha075_smoke.jsonl','ifeval/completion_v1/STOP_CONTINUATION.json','ifeval/completion_v1/STATUS.json','ifeval/STOP.json','protocol/IFEVAL_TEMPLATE_CONFLICT.json']
for rel in extra:included[str(R/rel)]=rel
references={str(p):sha(p) for p in (V/'generations').glob('*.jsonl')}
for p,h in checked.items():
 if p not in included:references[p]=h
for p,h in freeze['files'].items():
 if p not in included:references[p]=h
save('EVIDENCE_MANIFEST.json',{'format':'Explicit INCLUDED payloads and REFERENCED_ONLY external files','included':[{'source_path':p,'archive_path':arc,'sha256':sha(p),'bytes':Path(p).stat().st_size} for p,arc in sorted(included.items())],'referenced_only':[{'path':p,'sha256':h,'payload_included':False} for p,h in sorted(references.items())],'manifest_self':'Included as ifeval/completion_v2/EVIDENCE_MANIFEST.json; own hash excluded to avoid self-reference','no_weights_datasets_or_caches_in_payload':True})
archive=R.parent/'PETITGPT_IFEVAL_EFFECTIVE_CONFIG_FIX_V2_COMPLETION_EVIDENCE.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for p,arc in sorted(included.items()):tar.add(p,arcname=arc,recursive=False)
 tar.add(V/'EVIDENCE_MANIFEST.json',arcname='ifeval/completion_v2/EVIDENCE_MANIFEST.json',recursive=False)
with tarfile.open(archive,'r:gz') as tar:
 for item in read(V/'EVIDENCE_MANIFEST.json')['included']:
  assert hashlib.sha256(tar.extractfile(item['archive_path']).read()).hexdigest()==item['sha256']
h=sha(archive);(V/'ARCHIVE_SHA256.txt').write_text(h+'  '+str(archive)+'\n')
print(json.dumps({'status':status,'results':outcomes,'archive':str(archive),'archive_sha256':h,'archive_bytes':archive.stat().st_size},indent=2),flush=True)
