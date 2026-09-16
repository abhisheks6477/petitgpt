"""Completion-only audit/report/package; never runs any model or checker."""
from pathlib import Path
import json,hashlib,collections,datetime,tarfile,subprocess
C=Path(__file__).resolve().parent;R=C.parents[1];W=R.parent
OLD_SHA='87b730e003ca6f1c2a14487e90f193ae1ee7f313306f43f8efc7db3c548d802f'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def pc(x):return f'{100*x:.2f}%'
def main():
 old=W/'PETITGPT_THREE_TASK_BENCHMARK_EXTENSION_V1_EVIDENCE.tar.gz';assert sha(old)==OLD_SHA
 prior=read(C/'PRIOR_EVIDENCE_IDENTITIES.json');route=read(C/'GENERATOR_OUTPUT_ROUTING.json');prior_checks=[]
 for e in prior['verified_existing_files']:
  p=R/e['path'];expected=route['effective_sha256'] if e['path']=='runtime/generate_ifeval.py' else e['sha256'];assert sha(p)==expected,str(p)
  prior_checks.append({'file':e['path'],'sha256':sha(p),'unchanged':expected==e['sha256']})
 assert sha(C/'generate_ifeval_pre_continuation.py')==route['original_sha256']
 assert sha(R/'MANIFEST.json')==prior['prior_manifest_sha256']
 models=read(R/'protocol/MODEL_IDENTITIES.json');model_files=[]
 for m in models:
  for f in m['verified_files']:
   p=Path(f['path']);assert sha(p)==f['sha256'];model_files.append({'file':str(p),'sha256':f['sha256']})
 env=read(R/'protocol/EVALUATOR_ENVIRONMENT.json')
 for p,h in {**env['resource_files'],**env['package_archives']}.items():assert sha(R/p)==h
 for dist in read(C/'ENVIRONMENT_RECHECK.json')['packages'].values():
  for e in dist['record_verified_files']:
   if 'sha256' in e:assert sha(Path(e['file']))==e['sha256']
 freeze=read(R/'protocol/IFEVAL_CODE_FREEZE.json')
 for p,h in freeze['files'].items():assert sha(Path(p))==h
 save(C/'FINAL_IDENTITY_AUDIT.json',{'status':'PASS','old_archive_sha256':OLD_SHA,'prior_manifest_unchanged':True,'old_STOP_unchanged':True,'prior_file_checks':prior_checks,'likelihood_results_data_protocols_code_unchanged':True,'new_likelihood_evaluations':0,'generator_only_change':route,'models':model_files,'verifier_dependencies_resources_unchanged':True,'IFEval_fingerprint':freeze['fingerprint']})
 docs=rows(R/'ifeval/normalized_rows.jsonl');assert len(docs)==541 and sha(R/'ifeval/normalized_rows.jsonl')=='4d49ac039cbebdfc4beb3f9f30435c4fa25320caffaae441c6b9d868126744eb'
 keys=[d['key'] for d in docs];ids=set(t for d in docs for t in d['instruction_id_list']);assert len(set(keys))==541 and sum(len(d['instruction_id_list']) for d in docs)==834 and len(ids)==25
 metrics=[];generation_counts={};scored_counts={};caps={};audits=[]
 for m in models:
  mid=m['id'];gp=R/f'ifeval/generations/{mid}.jsonl';sp=R/f'ifeval/verifier/{mid}_full.jsonl';rp=R/f'ifeval/RESULTS_{mid}.json';ins=rows(R/f'ifeval/generation_inputs_{mid}.jsonl')
  rr=rows(gp) if gp.exists() else [];generation_counts[mid]=len(rr);caps[mid]=sum(g['stop_reason']=='max_new_tokens' for g in rr);scored_counts[mid]=0
  assert len(rr)<=541 and [g['key'] for g in rr]==keys[:len(rr)]
  for i,g in enumerate(rr):
   inp=ins[i];assert g['fingerprint']==freeze['fingerprint'] and g['model']==mid
   assert g['prompt_sha256']==digest(docs[i]['prompt']) and g['prompt_ids_sha256']==inp['prompt_ids_sha256'] and g['rendered_prompt_sha256']==inp['rendered_prompt_sha256']
   assert g['response_sha256']==digest(g['response']) and g['generated_ids_sha256']==digest(canon(g['generated_token_ids'])) and g['generated_token_count']==len(g['generated_token_ids'])
   assert 0<g['generated_token_count']<=1280 and g['prompt_tokens']+1280<=inp['context']
   assert g['generation_settings']=={'do_sample':False,'temperature':0.0,'max_new_tokens':1280,'num_beams':1,'repetition_penalty':1.0,'until':[],'use_cache':True}
   eos=3 if mid.startswith('Petit') else 2
   assert (g['stop_reason']=='eos' and g['generated_token_ids'][-1]==eos) or (g['stop_reason']=='max_new_tokens' and g['generated_token_count']==1280 and g['generated_token_ids'][-1]!=eos)
  if sp.exists():
   ss=rows(sp);assert [s['key'] for s in ss]==keys[:len(ss)]
   for i,s in enumerate(ss):
    assert s['response_sha256']==rr[i]['response_sha256'];n=len(docs[i]['instruction_id_list'])
    for mode in ['strict','loose']:
     flags=s[f'inst_level_{mode}_acc'];assert len(flags)==n and all(type(v) is bool for v in flags);assert s[f'prompt_level_{mode}_acc']==all(flags)
   scored_counts[mid]=sum(len(s['inst_level_strict_acc']) for s in ss)
   if len(rr)==len(ss)==541 and rp.exists():
    stored=read(rp);calc={'rows':541,'instruction_count':834,'prompt_strict_correct':sum(all(s['inst_level_strict_acc']) for s in ss),'prompt_loose_correct':sum(all(s['inst_level_loose_acc']) for s in ss),'instruction_strict_correct':sum(sum(s['inst_level_strict_acc']) for s in ss),'instruction_loose_correct':sum(sum(s['inst_level_loose_acc']) for s in ss)}
    for mode in ['strict','loose']:
     calc[f'prompt_level_{mode}_acc']=calc[f'prompt_{mode}_correct']/541;calc[f'inst_level_{mode}_acc']=calc[f'instruction_{mode}_correct']/834
    assert all(stored[k]==v for k,v in calc.items())
    assert stored['generations_sha256']==sha(gp) and stored['verifier_rows_sha256']==sha(sp) and stored['independent_recomputation']=='PASS'
    assert set(stored['instruction_id_coverage'])==ids
    metrics.append(stored);audits.append({'model':mid,'status':'PASS','generation_keys':541,'scored_instructions':834,'instruction_types':25,'decoded_response_and_ID_hashes':'PASS','saved_checker_flags_aggregation':'PASS','pinned_verifier_second_pass':stored['independent_recomputation'],'checker_rows_sha256':sha(sp),'generations_sha256':sha(gp)})
 complete=len(metrics)==3
 smoke_details=[]
 for m in models:
  mid=m['id'];gp=R/f'smoke/ifeval_{mid}.jsonl';sp=R/f'ifeval/verifier/{mid}_smoke.jsonl';rp=R/f'smoke/IFEVAL_SCORED_{mid}.json'
  if gp.exists():
   gg=rows(gp);ss=rows(sp);stored=read(rp);assert [g['key'] for g in gg]==keys[:8] and [v['key'] for v in ss]==keys[:8]
   for i,(g,v) in enumerate(zip(gg,ss)):
    assert g['fingerprint']==freeze['fingerprint'] and digest(g['response'])==g['response_sha256']==v['response_sha256']
    assert digest(canon(g['generated_token_ids']))==g['generated_ids_sha256'] and len(g['generated_token_ids'])==g['generated_token_count']
    assert len(v['inst_level_strict_acc'])==len(v['inst_level_loose_acc'])==len(docs[i]['instruction_id_list'])
   n=sum(len(v['inst_level_strict_acc']) for v in ss);ps=sum(all(v['inst_level_strict_acc']) for v in ss);pl=sum(all(v['inst_level_loose_acc']) for v in ss);si=sum(sum(v['inst_level_strict_acc']) for v in ss);li=sum(sum(v['inst_level_loose_acc']) for v in ss)
   assert (stored['prompt_strict_correct'],stored['prompt_loose_correct'],stored['instruction_strict_correct'],stored['instruction_loose_correct'],stored['instruction_count'])==(ps,pl,si,li,n)
   det=read(R/f'smoke/IFEVAL_DETERMINISM_{mid}.json');assert det['pass'] and det['first_generated_ids_sha256']==det['repeat_generated_ids_sha256']
   smoke_details.append({'model':mid,'rows':8,'instructions':n,'status':'PASS','deterministic_repeat':'PASS','prompt_strict':f'{ps}/8','prompt_loose':f'{pl}/8','instruction_strict':f'{si}/{n}','instruction_loose':f'{li}/{n}','stop_reasons':dict(collections.Counter(g['stop_reason'] for g in gg)),'independent_aggregation':'PASS','benchmark_result':False})
  else:smoke_details.append({'model':mid,'rows':0,'instructions':0,'status':'BLOCKED_BEFORE_GENERATION' if mid=='SmolLM-135M-Instruct' else 'NOT_RUN','benchmark_result':False})
 save(C/'SMOKE_ACTUAL_RESULTS.json',{'models':smoke_details,'full_benchmark_results':False})
 smoke=read(R/'smoke/IFEVAL.json') if (R/'smoke/IFEVAL.json').exists() else None
 smoke_state='PASS' if smoke and smoke['status']=='PASS' else 'FAIL' if (C/'STOP_CONTINUATION.json').exists() else 'NOT_RUN'
 if smoke_state=='PASS':
  assert smoke['fingerprint']==freeze['fingerprint']
  for m in models:
   det=read(R/f"smoke/IFEVAL_DETERMINISM_{m['id']}.json");assert det['pass'] and det['first_generated_ids_sha256']==det['repeat_generated_ids_sha256']
 if complete:assert smoke_state=='PASS' and read(R/'ifeval/RESULTS.json')['status']=='PASS'
 full='PASS' if complete else 'PARTIAL' if sum(generation_counts.values()) else 'NOT_RUN'
 status={'STATUS':'COMPLETE' if complete else 'PARTIAL_STOPPED','IFEVAL_TEMPLATE_RESOLUTION':'CANONICAL_DEFAULT_SYSTEM_RETAINED','CALLER_SUPPLIED_SYSTEM_MESSAGE':False,'LIKELIHOOD_RESULTS_UNCHANGED':True,'NEW_LIKELIHOOD_EVALUATIONS':0,'IFEVAL_SMOKE':smoke_state,'IFEVAL_FULL':full,'IFEVAL_PROMPTS_PER_MODEL':generation_counts,'IFEVAL_INSTRUCTIONS_PER_MODEL':scored_counts,'IFEVAL_MODELS_COMPLETED':len(metrics),'PROMPT_ROWS_DROPPED':0,'INPUT_ROWS_TRUNCATED':0,'GENERATION_CAP_HITS':caps,'MODEL_PARAMETERS_MODIFIED':False,'FORMAL_TRAINING_STARTED':False,'GITHUB_WRITES':0,'HF_WRITES':0,'STORAGE_MAINTENANCE_ACTIONS':0,'NETWORK_DOWNLOADS':0,'DEPENDENCY_INSTALLATIONS':0,'FIXTURE_RERUNS':0,'SMOKE_PROMPTS_PER_MODEL':{v['model']:v['rows'] for v in smoke_details},'SMOKE_INSTRUCTIONS_PER_MODEL':{v['model']:v['instructions'] for v in smoke_details},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save(C/'STATUS.json',status);save(C/'INDEPENDENT_COMPLETION_AUDIT.json',{'status':'PASS' if complete else 'PARTIAL','implementation':'independent standard-library aggregation from saved per-prompt checker flags; no model or checker execution','models':audits,'completed_smoke_audits':smoke_details})
 table='| Model | Prompt strict | Inst strict | Prompt loose | Inst loose |\n|---|---:|---:|---:|---:|\n'
 for m in models:
  v=next((x for x in metrics if x['model']==m['id']),None);table+='| '+m['id']+' | '+(' | '.join(pc(v[k]) for k in ['prompt_level_strict_acc','inst_level_strict_acc','prompt_level_loose_acc','inst_level_loose_acc']) if v else 'N/A | N/A | N/A | N/A')+' |\n'
 table+='\n| Model | Prompt strict 分子/541 | Inst strict 分子/834 | Prompt loose 分子/541 | Inst loose 分子/834 |\n|---|---:|---:|---:|---:|\n'
 for v in metrics:table+=f"| {v['model']} | {v['prompt_strict_correct']}/541 | {v['instruction_strict_correct']}/834 | {v['prompt_loose_correct']}/541 | {v['instruction_loose_correct']}/834 |\n"
 table='以下为正式 IFEval 指标；未完成全量的模型全部记为 N/A。\n\n'+table
 (C/'IFEVAL.md').write_text(table)
 report='# PetitGPT IFEval 续跑报告\n\n状态：**'+status['STATUS']+'**。'+('三模型各 541 条官方提示、834 条指令均已完成生成和评分。' if complete else '执行遇到问题停止，实际完成数量见状态和错误记录。')+'\n\n'
 report+='本次仅执行 IFEval 剩余工作；ARC-Challenge、HellaSwag 及旧 ARC-Easy/PIQA 未重新评测，已有数据、协议、代码、逐题结果和聚合值保持原样。旧 PARTIAL 证据包、原 STOP/NOT_RUN 记录、旧报告与旧状态均保留。\n\n'
 report+='## 格式决策与固定协议\n\n调用者向每个模型只提供一条未改变的官方 user message，不手工加入 system 或 few-shot 消息。各模型保留原生 formatter；PetitGPT 使用 default_system=None；SmolLM2 保留模板自动插入的原文：\n\n> You are a helpful AI assistant named SmolLM, trained by Hugging Face\n\n模板 SHA256：`872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788`。决策在零生成状态下记录，已纳入生成指纹。只解决旧 messages 描述的歧义，没有改变数据、解码、预算、数值和校验器规则。\n\n'
 report+='源码：`b82ede26bff93423bd05fa08b8cae6ebd62d1dee`；harness：`b954108c9baaaa934b4ad842033b31a97ee30816`。数据：`google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84`，train split，541 prompts / 834 instructions / 25 types。规范化行 SHA256：`4d49ac039cbebdfc4beb3f9f30435c4fa25320caffaae441c6b9d868126744eb`。\n\n'
 report+='GPU：RTX 4090；FP32 参数与 forward，SDPA MATH，autocast/TF32 关闭，batch1，KV cache 开启；greedy，do_sample=false，temperature=0，num_beams=1，repetition_penalty=1.0，max_new_tokens=1280；canonical EOS，无自定义文本停止串。原生解码路径保持不变，没有答案清理、修复或重采样。\n\n'
 report+='最长格式化提示：PetitGPT 373、SmolLM 365、SmolLM2 386 tokens；全部可容纳固定 1280-token 预算。原 generation_inputs 三文件哈希保持不变。模型身份见 MODEL_IDENTITIES；执行前后重新验证权重/config/tokenizer，参数文件未变。\n\n'
 report+='## Smoke 与 IFEval 结果\n\n前八条官方提示的三模型 smoke：**'+smoke_state+'**；每模型第一条重复生成用于验证确定性。smoke 与正式评测文件分开，低分、提前 EOS 和触及 cap 均不作为基础设施失败。旧 25 类 verifier fixtures 依身份检查继续有效，本次没有重跑。\n\n'+table+'\n'
 report+='实际 smoke 记录（仅工程检查，不是 benchmark）：\n\n| Model | 已完成提示 | 已评分指令 | 状态 |\n|---|---:|---:|---|\n'
 for v in smoke_details:report+=f"| {v['model']} | {v['rows']} | {v['instructions']} | {v['status']} |\n"
 for v in smoke_details:
  if v['rows']:report+=f"\n{v['model']} smoke：prompt strict {v['prompt_strict']}，prompt loose {v['prompt_loose']}，instruction strict {v['instruction_strict']}，instruction loose {v['instruction_loose']}；停止原因 {v['stop_reasons']}；确定性重复和独立聚合均 PASS。\n"
 report+='\n正式全量停止计数（未开始则无记录）：\n\n'
 report+='| Model | EOS | 1280-token cap | 生成 token 总数（含 EOS） |\n|---|---:|---:|---:|\n'
 for v in metrics:report+=f"| {v['model']} | {v['stop_reasons'].get('eos',0)} | {v['stop_reasons'].get('max_new_tokens',0)} | {v['generated_tokens']} |\n"
 if not metrics:report+='\n全量尚未开始，因此没有全量 EOS/cap 分布；已完成 smoke 的停止原因列在上方。\n'
 report+='\n对实际完成的 smoke/全量响应，原 pinned strict/loose verifier 在 seed=0 下评分并执行第二次 verifier 复算；另从已落盘 checker 布尔结果独立计算分子与分母。没有将未生成的响应或基础设施失败计为普通 checker 错误或成功。完整运行要求每模型 541 个唯一 key、834 条指令和全部 25 类覆盖。所有低分响应及其 token ID 均保留。\n\n'
 if not complete and (C/'STOP_CONTINUATION.json').exists():report+='停止原因：当前 Transformers 5.15.0 将未设置的生成参数保存为 None，现成生成器要求 repetition_penalty 原始值直接等于 1.0，导致调用 generate 前的兼容性断言失败。按本次停止规则没有修补断言、重试模型或开始全量。\n\n停止证据：\n\n```json\n'+json.dumps(read(C/'STOP_CONTINUATION.json'),ensure_ascii=False,indent=2)+'\n```\n\n'
 report+='## 已接受 likelihood 结果（原样复用）\n\n'+(R/'tables/LIKELIHOOD_NEW_TASKS.md').read_text()+'\n'+(R/'tables/FOUR_TASK_PROPOSAL.md').read_text()+'\n'
 report+='## 解释边界\n\n这是原生 chat 格式下的指令遵从比较：官方 user prompt 相同，但格式化后的输入不同，包括 SmolLM2 的默认 system。这不是 token 输入完全一致的实验，也不声称复现官方 model-card 协议。IFEval 与 likelihood 分开，不计算跨任务平均分，不据此声称统计显著性或总体智能优劣。HellaSwag acc_norm 沿用项目字符归一化口径。\n\n'
 report+='本次无网络下载、依赖重装、训练、公开写入或存储维护。现成生成器仅将进度输出转到本 completion 目录，以保留历史进度文件；旧源码副本及改动前后哈希均存档。依赖 RECORD 检查发现 pip --target 的外部 nltk CLI 路径不在该位置；该 CLI 不参与运行，所有实际导入包文件哈希及导入检查通过。\n\n'
 report+='## 交付与证据边界\n\n运行目录：`'+str(R)+'`。本次报告/状态/表：`'+str(C)+'`。旧归档 SHA256 仍为 `'+OLD_SHA+'`。\n\n精简包包含：格式决策、协议/身份、环境复核、实际 smoke 摘要与确定性结果、聚合结果、每条 prompt 的 compact checker outcomes、独立复算、报告/状态/表和执行日志。\n\n仅按哈希引用：完整响应与生成 token ID、完整格式化输入、数据源/规范化数据、likelihood 全量逐题结果及模型文件；它们保留在原运行树或原模型目录。精简包不含权重、数据集 payload、完整响应、完整 token ID 或缓存。文件级 INCLUDED/REFERENCED 列表见 EVIDENCE_MANIFEST.json。\n\n'
 report+='```text\n'+'\n'.join(k+'='+(str(v).lower() if isinstance(v,bool) else json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else str(v)) for k,v in status.items() if k.isupper())+'\n```\n'
 (C/'REPORT_ZH.md').write_text(report)
 included=[C/p for p in ['REPORT_ZH.md','STATUS.json','IFEVAL.md','INDEPENDENT_COMPLETION_AUDIT.json','FINAL_IDENTITY_AUDIT.json','PRIOR_EVIDENCE_IDENTITIES.json','ENVIRONMENT_RECHECK.json','GENERATOR_OUTPUT_ROUTING.json','SMOKE_ACTUAL_RESULTS.json','PRECHECK_NOTE.json','deliver.py','check_environment.py','generate_ifeval_pre_continuation.py']]
 included+=list(C.glob('*.log'))
 included+=[R/'protocol'/p for p in ['IFEVAL_TEMPLATE_RESOLUTION.json','IFEVAL_TEMPLATE_CONFLICT.json','IFEVAL_PROTOCOL.json','IFEVAL_CONTEXT_PREFLIGHT.json','IFEVAL_CODE_FREEZE.json','IFEVAL_SMOKE_SELECTION.json','MODEL_IDENTITIES.json','DATA_IDENTITIES.json','SOURCE_IDENTITIES.json','EVALUATOR_ENVIRONMENT.json','ENVIRONMENT.json']]
 included+=[R/'runtime/generate_ifeval.py',R/'runtime/verify_ifeval.py',R/'ifeval/STOP.json',R/'ifeval/verifier/FIXTURES.json',R/'tables/LIKELIHOOD_NEW_TASKS.md',R/'tables/FOUR_TASK_PROPOSAL.md']
 included+=list((R/'ifeval/verifier').glob('*_full.jsonl'));included+=list((R/'ifeval/verifier').glob('*_smoke.jsonl'));included+=list((R/'ifeval').glob('RESULTS*.json'));included+=list((R/'smoke').glob('IFEVAL*.json'))
 included+=list((R/'protocol').glob('GENERATION_CONFIG_*.json'))
 if (C/'STOP_CONTINUATION.json').exists():included.append(C/'STOP_CONTINUATION.json')
 included=sorted(set(p for p in included if p.exists()))
 referenced=list((R/'ifeval/generations').glob('*.jsonl'))+list((R/'ifeval').glob('generation_inputs_*.jsonl'))+[R/'ifeval/normalized_rows.jsonl',R/'data/ifeval.jsonl']+list((R/'smoke').glob('ifeval_*.jsonl'))
 referenced+=list((R/'arc_challenge').glob('rows_*.jsonl'))+list((R/'hellaswag').glob('rows_*.jsonl'))
 im=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p),'disposition':'INCLUDED'} for p in included]
 rm=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p),'disposition':'REFERENCED_ONLY'} for p in referenced]
 rm+=[{'path':str(Path(f['path'])),'bytes':f['bytes'],'sha256':f['sha256'],'disposition':'REFERENCED_ONLY'} for m in models for f in m['verified_files']]
 save(C/'EVIDENCE_MANIFEST.json',{'included':im,'referenced':rm,'prior_archive':{'path':str(old),'sha256':OLD_SHA,'disposition':'REFERENCED_ONLY'},'self_note':'This manifest is included; its hash is protected by the archive SHA256, avoiding self-reference.'})
 archive=W/'PETITGPT_IFEVAL_CANONICAL_TEMPLATE_CONTINUE_V1_COMPLETION_EVIDENCE.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'x:gz') as t:
  for p in included+[C/'EVIDENCE_MANIFEST.json']:t.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
 ah=sha(archive);Path(str(archive)+'.sha256').write_text(ah+'  '+archive.name+'\n')
 with tarfile.open(archive,'r:gz') as t:
  for member in t.getmembers():assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==sha(R/Path(member.name).relative_to('evidence'))
 assert sha(old)==OLD_SHA
 print(json.dumps({'status':status,'report':str(C/'REPORT_ZH.md'),'archive':str(archive),'sha256':ah,'archive_bytes':archive.stat().st_size,'metrics':metrics},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
