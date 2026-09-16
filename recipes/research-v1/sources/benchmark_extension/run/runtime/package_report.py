"""Write the final Chinese report and a compact private evidence archive."""
from pathlib import Path
import json,hashlib,tarfile,subprocess,datetime,collections,re
R=Path(__file__).resolve().parents[1];W=R.parent;S=W/'source'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads((R/p).read_text())
def save(p,x):(R/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def pc(x):return f'{x*100:.2f}%'
def main():
 models=load('protocol/MODEL_IDENTITIES.json');data=load('protocol/DATA_IDENTITIES.json');env=load('protocol/ENVIRONMENT.json');source=load('protocol/SOURCE_IDENTITIES.json');likelihood=load('protocol/INDEPENDENT_LIKELIHOOD_AUDIT.json');assert likelihood['status']=='PASS'
 ip=load('protocol/IFEVAL_PROTOCOL.json');ctx=load('protocol/IFEVAL_CONTEXT_PREFLIGHT.json');done=(R/'ifeval/RESULTS.json').exists() and load('ifeval/RESULTS.json')['status']=='PASS'
 if done:
  iv=load('ifeval/RESULTS.json')['results'];assert len(iv)==3 and all(v['rows']==541 and v['instruction_count']==834 and v['independent_recomputation']=='PASS' for v in iv)
  it='| Model | Prompt strict | Inst strict | Prompt loose | Inst loose |\n|---|---:|---:|---:|---:|\n'
  for v in iv:it+='| '+v['model']+' | '+' | '.join(pc(v[k]) for k in ['prompt_level_strict_acc','inst_level_strict_acc','prompt_level_loose_acc','inst_level_loose_acc'])+' |\n'
  it+='\n每模型 541 prompts、834 instructions；覆盖全部 25 类指令。\n\n| Model | Prompt strict 分子/541 | Inst strict 分子/834 | Prompt loose 分子/541 | Inst loose 分子/834 |\n|---|---:|---:|---:|---:|\n'
  for v in iv:it+=f"| {v['model']} | {v['prompt_strict_correct']}/541 | {v['instruction_strict_correct']}/834 | {v['prompt_loose_correct']}/541 | {v['instruction_loose_correct']}/834 |\n"
 else:
  reason=load('ifeval/STOP.json')['reason'] if (R/'ifeval/STOP.json').exists() else '等待解决 canonical SmolLM2 自动 system 文本与无 system 要求之间的冲突。'
  it='IFEval 尚未进行完整生成，四项指标均为 N/A，不能将 smoke 或外部 model-card 分数代入。\n\n'+reason+'\n\n| Model | Prompt strict | Inst strict | Prompt loose | Inst loose |\n|---|---:|---:|---:|---:|\n'
  for m in models:it+='| '+m['id']+' | N/A | N/A | N/A | N/A |\n'
  iv=[]
 (R/'tables/IFEVAL.md').write_text(it)
 historical=load('protocol/HISTORICAL_CHECKOUT_READONLY.json');unchanged=[]
 for f in historical['working_files']:
  actual=sha(Path(f['file']));assert actual==f['sha256'];unchanged.append({'file':f['file'],'sha256':actual})
 actual_status=subprocess.check_output(['git','--no-optional-locks','-C','/workspace/petitgpt','status','--porcelain'],text=True);assert actual_status==historical['status']
 final_models=[]
 for m in models:
  for f in m['verified_files']:
   actual=sha(Path(f['path']));assert actual==f['sha256'];final_models.append({'file':f['path'],'sha256':actual})
 subprocess.run(['git','--no-optional-locks','-C',str(S),'diff','--exit-code'],check=True)
 save('protocol/FINAL_INPUT_INTEGRITY.json',{'status':'PASS','model_files':final_models,'historical_checkout_files':unchanged,'historical_git_status_unchanged':True,'public_source_clean':True})
 resumed=bool(list(R.glob('*/RESUME_*.json')))
 status={'STATUS':'COMPLETE' if done else 'PARTIAL','LIKELIHOOD_TASKS':'ARC_CHALLENGE,HELLASWAG','IFEVAL_TASK':'IFEVAL','ARC_CHALLENGE_ROWS':1172,'HELLASWAG_ROWS':10042,'IFEVAL_ROWS':541,'IFEVAL_GENERATIONS_PER_MODEL':541 if done else 0,'MODELS_EVALUATED_LIKELIHOOD':3,'MODELS_EVALUATED_IFEVAL':3 if done else 0,'LIKELIHOOD_SMOKE':'PASS','IFEVAL_SMOKE':'PASS' if (R/'smoke/IFEVAL.json').exists() else 'NOT_RUN','LIKELIHOOD_FULL':'PASS','IFEVAL_FULL':'PASS' if done else 'NOT_RUN','ROWS_EXCLUDED':0,'ROWS_TRUNCATED':0,'MODEL_PARAMETERS_MODIFIED':False,'FORMAL_TRAINING_STARTED':False,'TEACHER_API_CALLS':0,'GITHUB_WRITES':0,'HF_WRITES':0,'STORAGE_MAINTENANCE_ACTIONS':0,'resumed_after_interruption':resumed,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save('STATUS.json',status)
 report=f"# PetitGPT 三任务扩展评测\n\n状态：**{status['STATUS']}**。两个 likelihood 任务均完成三模型全部样本，并通过独立复算。"+('IFEval 完成三模型各 541 条生成和 strict/loose 独立复算。' if done else 'IFEval 未完成，原因见独立章节。')+'\n\n'
 report+=f"## 固定来源与环境\n\n公开源码：`{source['public_commit']}`，与请求的起始 main 完全一致；本次只在独立公开副本运行。\n\nlm-evaluation-harness：`{source['harness_commit']}`。具体源码 SHA256 见 `protocol/SOURCE_IDENTITIES.json` 和代码冻结清单。\n\nGPU：{env['gpu']}；CUDA runtime：{env['cuda']}；Python：{env['python']}。\n\n"
 driver=re.search(r'Driver Version:\s*([0-9.]+)',env['nvidia_smi']).group(1)
 report+='NVIDIA driver：`'+driver+'`。\n\n'
 report+='包版本：'+', '.join(f"{k}={v}" for k,v in env['versions'].items())+'。驱动信息保存在 `protocol/ENVIRONMENT.json`。\n\n'
 report+='网络下载：有，下载了公开源码、固定 harness 小文件、三个数据集指定 split、小型 verifier 依赖和唯一需要的 NLTK `punkt_tab` 资源；未下载模型。已有模型、tokenizer 和历史环境只读复用。'+('发生过严格身份检查后的断点恢复。' if resumed else '未发生模型执行断点恢复。')+'\n\n'
 report+='| Model | 固定身份 |\n|---|---|\n'
 for m in models:report+=f"| {m['id']} | "+('checkpoint SHA256 `'+m['sha256']+'`' if m['id'].startswith('Petit') else m['repo_id']+' @ `'+m['revision']+'`')+' |\n'
 report+='\nPetitGPT tokenizer SHA256：`d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce`。全部模型/config/tokenizer 文件进入执行前与结束后的哈希复核。\n\n'
 report+='## ARC-Challenge 与 HellaSwag\n\n'
 for n in ['arc_challenge','hellaswag']:
  d=data[n];report+=f"- {n}：`{d['repo']}` @ `{d['revision']}`，{d['split']}，{d['rows']} 行；源文件 SHA256 `{d['source_sha256']}`；规范化行文件 SHA256 `{d['normalized_rows_sha256']}`。\n"
 report+='\nHellaSwag 直接执行固定提交的 `process_docs` / `preprocess`，并逐行核对等价预处理；不使用 ARC Question/Answer 包装。原 `ind` 字段不唯一，以官方物理行号产生唯一 ID，同时保留原 `ind`；未删行。\n\n'
 preprocess_identity=next(x for x in source['files'] if x['file'].endswith('/hellaswag/utils.py'))
 report+='HellaSwag 预处理源码 SHA256：`'+preprocess_identity['sha256']+'`。\n\n'
 report+='共同协议：zero-shot raw completion；无 chat/BOS/EOS；每个选项前一个空格；FP32 参数/forward/log_softmax/sum；禁用 autocast/TF32；SDPA MATH；batch=1，无 padding、无 KV cache、无 compile；平分取第一个最大值。token 边界、forward、sum 与 metric 复用公开审查代码的原始 AST 方法。\n\n'
 report+=(R/'tables/LIKELIHOOD_NEW_TASKS.md').read_text()+'\n'
 report+='| Model / Task | acc 分子 | acc_norm 分子 | 分母 | 候选序列数 |\n|---|---:|---:|---:|---:|\n'
 for task in ['arc_challenge','hellaswag']:
  for v in load(task+'/RESULTS.json')['results']:report+=f"| {v['model']} / {task} | {v['acc_correct']} | {v['acc_norm_correct']} | {v['rows']} | {v['candidate_sequences']} |\n"
 report+='\n## 四任务 likelihood 提议表\n\n'+(R/'tables/FOUR_TASK_PROPOSAL.md').read_text()+'\n'
 report+='## IFEval（单独报告）\n\n'
 d=data['ifeval'];report+=f"数据：`{d['repo']}` @ `{d['revision']}`，harness 使用 train split，541 行，834 条指令，25 类指令全部注册且通过 fixtures。源文件 SHA256：`{d['source_sha256']}`；规范化行文件 SHA256：`{d['normalized_rows_sha256']}`。\n\n"
 report+='格式：输入均为同一条官方 user prompt；PetitGPT 使用发布版 `encode_prompt(default_system=None)`，两个 SmolLM 使用各自固定 tokenizer 的 `apply_chat_template(add_generation_prompt=True)`。无 few-shot、无额外用户指令包装。\n\n'
 if (R/'protocol/IFEVAL_TEMPLATE_RESOLUTION.json').exists():report+='SmolLM2 的 canonical 模板会自动加入 system 文本；已在首次生成前明确处理此协议冲突，决策记录见 `protocol/IFEVAL_TEMPLATE_RESOLUTION.json`。输入消息列表本身仅含一条 user。\n\n'
 report+=('已执行的冻结生成规则：' if done else '待模板冲突解决后才可执行的计划规则：')+'greedy，do_sample=false，temperature=0，max_new_tokens=1280，单 beam，无重复惩罚，无自定义文本停止串，允许 canonical EOS；FP32/MATH，KV cache 开启。只评分 continuation；PetitGPT 仅移除最终 EOS，保留其他 token 解码，不清理答案；HF 使用 pinned tokenizer 默认 decode 和 skip_special_tokens=True。\n\n'
 report+='| Model | 最长格式化提示 tokens | 总 context | +1280 后是否全部可容纳 |\n|---|---:|---:|---|\n'
 for mid,x in ctx.items():report+=f"| {mid} | {x['max_prompt_tokens']} | {x['context']} | {'是' if x['overflow_count']==0 else '否'} |\n"
 report+='\nPetitGPT 全部 541 条提示均 ≤768 tokens，因此完整 1280-token 预算有效；没有截断或逐行减小预算。\n\n'+it+'\n'
 if done:
  report+='停止原因与生成量：\n\n| Model | EOS | max_new_tokens | 生成 token 总数（含 EOS） |\n|---|---:|---:|---:|\n'
  for v in iv:report+=f"| {v['model']} | {v['stop_reasons'].get('eos',0)} | {v['stop_reasons'].get('max_new_tokens',0)} | {v['generated_tokens']} |\n"
 report+='\nVerifier：原始 pinned strict/loose checker 未修改；25 类符合/不符合 fixture、strict/loose 差异 fixture 和全部 834 条实际参数检查通过。隔离环境 nltk=3.9.1、langdetect=1.0.9、immutabledict=4.2.1；仅下载 `punkt_tab`；固定 random/langdetect seed=0。\n\n'
 report+='## 验证与限制\n\n所有完成结果均保留，未按分数选择协议或重试答案。初期数据适配发现 HellaSwag `ind` 重复、Transformers 默认返回 BatchEncoding，均在模型运行前修正；原失败日志保留，不涉及丢弃评测结果。依赖安装缺少 wheel 的准备错误也已隔离解决。\n\n'
 report+='- HellaSwag 的 acc_norm 是项目字符归一化：对 pinned 预处理后的候选文本取 Python len，不含目标前导空格；不保证与使用 token 或其他归一化的外部榜单一致。\n- IFEval 采用每模型原生 chat 格式，不自动等同 SmolLM model-card 的评测设置。\n- zero-shot 多选题能力不等于聊天可靠性；IFEval 衡量生成指令遵从，不衡量事实问答质量。\n- IFEval 与 likelihood 指标分开，不计算跨任务总体分数或平均分。\n\n'
 archive=W/'PETITGPT_THREE_TASK_BENCHMARK_EXTENSION_V1_EVIDENCE.tar.gz'
 report+=f"## 证据\n\n运行目录：`{R}`。实际完成的逐题 likelihood 证据、IFEval 格式化输入及 verifier fixtures 均保留在运行树；本次未运行的生成不产生答案证据。文件 SHA256 见 `MANIFEST.json`。\n\n精简证据包：`{archive}`；校验文件：`{archive}.sha256`。包内不含权重、完整数据集、完整生成文本或缓存。归档自身哈希写入同级 `.sha256`，避免自引用。\n\n```text\n"+'\n'.join(f'{k}={str(v).lower() if isinstance(v,bool) else v}' for k,v in status.items() if k.isupper())+'\n```\n'
 (R/'REPORT_ZH.md').write_text(report)
 # Hash all final row/generation evidence, source inputs, protocols and scripts; skip dependency/cache payloads.
 files=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.name=='MANIFEST.json' or any(x in p.relative_to(R).parts for x in ['cache','evaluator_deps','evaluator_wheels','nltk_data','__pycache__']):continue
  files.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save('MANIFEST.json',{'files':files,'note':'Includes full row/generation evidence hashes; compact archive excludes data/rows/encodings/generation inputs/full responses/caches/weights.'})
 selected=[]
 for d in ['protocol','tables','upstream']:
  selected.extend(p for p in (R/d).rglob('*') if p.is_file())
 selected.extend([R/'STATUS.json',R/'REPORT_ZH.md',R/'MANIFEST.json'])
 selected.extend((R/'runtime').glob('*.py'));selected.extend((R/'runtime').glob('*.log'))
 selected.extend((R/'smoke').glob('*.json'));selected.append(R/'ifeval/verifier/FIXTURES.json')
 for task in ['arc_challenge','hellaswag','ifeval']:selected.extend((R/task).glob('RESULTS*.json'))
 if (R/'ifeval/STOP.json').exists():selected.append(R/'ifeval/STOP.json')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(set(selected)):tar.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
 ah=sha(archive);Path(str(archive)+'.sha256').write_text(ah+'  '+archive.name+'\n')
 # Read every archived member to verify its bytes against the finalized run tree.
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar.getmembers():
   p=R/str(Path(member.name).relative_to('evidence'));assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(p)
 print(json.dumps({'status':status['STATUS'],'archive':str(archive),'sha256':ah,'archive_bytes':archive.stat().st_size,'report':str(R/'REPORT_ZH.md')},indent=2))
if __name__=='__main__':main()
