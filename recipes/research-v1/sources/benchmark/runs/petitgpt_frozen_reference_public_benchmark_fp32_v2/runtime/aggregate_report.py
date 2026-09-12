"""Independent stored-row aggregation/reporting only; no model imports or forwards."""
import pathlib,json,hashlib,collections,csv,math,datetime
R=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
def rows(p):return [json.loads(s) for s in (R/p).read_text().splitlines()] if (R/p).exists() else []
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):(R/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def write(p,rs):
 with (R/p).open('w') as f:
  for r in rs:f.write(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n')
p=read('preparation/TASK_PROTOCOL_FREEZE_V2.json');amend=read('preparation/PROTOCOL_AMENDMENT_V2.json');models=read('preparation/MODEL_SELECTION_FREEZE.json')['models'];index=rows('preparation/REQUEST_INDEX.jsonl');env=read('runtime/ACTUAL_ENVIRONMENT.json');summary=read('checks/LIKELIHOOD_PARITY_SUMMARY.json')
events=rows('runtime/FORWARD_EVENTS.jsonl');calls=[e for e in events if e['event']=='invoked'];returns={e['forward_id']:e for e in events if e['event']=='returned'}
assert len({e['forward_id'] for e in calls})==len(calls) and len(returns)==sum(e['event']=='returned' for e in events)
primary=[e for e in calls if e['category']=='primary'];extras=[e for e in calls if e['category']!='primary'];assert len(extras)<=192
assert all(e['input_shape'][0]==1 for e in calls)
for e in returns.values():
 if e['route']=='FP32_V2':assert e['logits_dtype']=='torch.float32' and e['autocast_inside_forward'] is False and e['SDPA_context']=='MATH only' and not e['cudnn_allow_tf32'] and not e['cuda_matmul_allow_tf32']
choices=[];items=[];results=[];coverage={};by_global={};codehash=sha(R/'runtime/CODE_FREEZE.json');policyhash=sha(R/'preparation/PROTOCOL_AMENDMENT_V2.json');taskhash=sha(R/'preparation/TASK_PROTOCOL_FREEZE_V2.json')
for m in models:
 mid=m['id'];raw=rows(f'results/RAW_CHOICES_{mid}.jsonl');by={}
 for c in raw:
  key=(c['request_id'],c['choice_index']);assert key not in by;by[key]=c;by_global[(mid,*key)]=c
  assert c['model']==mid and c['code_sha256']==codehash and c['numeric_policy_sha256']==policyhash and c['task_protocol_sha256']==taskhash
  assert math.isfinite(c['continuation_logprob']) and c['C']>0 and c['K']>0 and c['C']+c['K']==c['total_tokens']<=2048
  assert c['forward_id'] in returns and returns[c['forward_id']]['route']=='FP32_V2'
 expected={(r['request_id'],j) for r in index for j in range(len(r['choice_sha256']))};assert set(by)<=expected
 model_items=[]
 for r in index:
  cs=[by.get((r['request_id'],j)) for j in range(len(r['choice_sha256']))]
  for j,c in enumerate(cs):
   if c is not None:
    assert c['prompt_sha256']==r['prompt_sha256'] and c['choice_sha256']==r['choice_sha256'][j] and c['continuation_sha256']==r['continuation_sha256'][j]
    assert c['normalization_denominator']==r['normalization_denominators'][j]>0 and c['normalized_score']==c['continuation_logprob']/c['normalization_denominator']
  if any(c is None for c in cs):
   for c in cs:
    if c:choices.append({**c,'gold':r['gold'],'prediction_acc':None,'prediction_acc_norm':None})
   continue
  a=max(range(len(cs)),key=lambda j:cs[j]['continuation_logprob']);n=max(range(len(cs)),key=lambda j:cs[j]['normalized_score'])
  item={'model':mid,'task':r['task'],'split':r['split'],'doc_id':r['doc_id'],'request_id':r['request_id'],'gold':r['gold'],'choice_count':len(cs),'prediction_acc':a,'prediction_acc_norm':n,'acc':int(a==r['gold']),'acc_norm':int(n==r['gold']),'raw_max_tie_count':sum(c['continuation_logprob']==cs[a]['continuation_logprob'] for c in cs),'norm_max_tie_count':sum(c['normalized_score']==cs[n]['normalized_score'] for c in cs)}
  model_items.append(item);items.append(item)
  for c in cs:choices.append({**c,'gold':r['gold'],'prediction_acc':a,'prediction_acc_norm':n})
 coverage[mid]={'primary_candidates':len(raw),'complete_documents':len(model_items),'exact_full_candidate_coverage':set(by)==expected,'raw_file_sha256':sha(R/f'results/RAW_CHOICES_{mid}.jsonl') if raw else None}
 for t in p['tasks']:
  sub=[r for r in model_items if r['task']==t['task']];n=len(sub);complete=n==t['rows']
  results.append({'model':mid,'task':t['task'],'split':t['split'],'status':'COMPLETE' if complete else 'PARTIAL' if n else 'UNMEASURED','documents':n,'expected_documents':t['rows'],'candidate_sequences':sum(c['task']==t['task'] for c in raw),'acc_correct':sum(r['acc'] for r in sub) if n else None,'acc':sum(r['acc'] for r in sub)/n if n else None,'acc_norm_correct':sum(r['acc_norm'] for r in sub) if n else None,'acc_norm':sum(r['acc_norm'] for r in sub)/n if n else None,'historical_or_author_reported':None})
assert len(primary)==len(choices)==len(by_global)
assert {(e['model'],e['request_id'],e['choice_index']) for e in primary}==set(by_global)
complete=summary['status']=='PASS' and all(v['exact_full_candidate_coverage'] for v in coverage.values())
if complete:
 assert len(choices)==39531 and len(items)==12642 and len(calls)==len(returns)
 assert len(extras)==amend['technical_extra_forwards_planned']
status='PUBLIC_BENCHMARK_FP32_COMPLETE' if complete else 'BLOCKED_FP32_LIKELIHOOD_VALIDATION_FAILED'
# Preserve every original bound file and full read-only root snapshot.
for f in read('preparation/READ_ONLY_INPUT_SNAPSHOT.json'):assert pathlib.Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],f['path']
for group in ['original_model_and_source_files','local_data_files','bound_v1_files']:
 for f in read('preparation/INPUT_BINDINGS_V2.json')[group]:assert sha(f['path'])==f['sha256'],f['path']
for f,h in read('runtime/CODE_FREEZE.json').items():assert sha(R/f)==h,f
for f,h in read('preparation/FROZEN_V2_FILES.json').items():assert sha(R/f)==h,f
bf=rows('checks/BF16_SYNTHETIC_DIAGNOSTIC.jsonl');assert len(bf)==4
same_shape=bf[3]['pairwise_deltas']['future_zeros'];repeat=bf[1]['pairwise_deltas']['unpadded']
parity_stats={}
for m in models:
 rs=rows(f'checks/FP32_PARITY_{m["id"]}.jsonl');tokenerrs=[];sumerrs=[];failed=[]
 for r in rs:
  if not r['pass']:failed.append(r)
  gs=list(r.get('comparisons',{}).values())+([r['comparison']] if 'comparison' in r else [])
  for g in gs:
   if 'max_token_absolute_error' in g:tokenerrs.append(g['max_token_absolute_error'])
   if 'absolute_error' in g:sumerrs.append(g['absolute_error'])
 parity_stats[m['id']]={'records':len(rs),'failed_records':len(failed),'maximum_token_error':max(tokenerrs,default=None),'maximum_scalar_error':max(sumerrs,default=None)}
counts={'document_model_evaluations':len(items),'unique_primary_candidate_sequences':len(choices),'technical_extra_forward_calls':len(extras),'bf16_diagnostic_forward_calls':sum(e['route']=='BF16_DIAGNOSTIC' for e in calls),'fp32_forward_calls':sum(e['route']=='FP32_V2' for e in calls),'total_forward_invocations':len(calls),'completed_forward_returns':len(returns),'forward_categories':dict(collections.Counter(e['category'] for e in calls)),'primary_retries':0,'technical_retries':0,'generated_completions':0,'training_updates':0,'backward_calls':0,'optimizer_constructions':0,'new_checkpoints':0,'paid_api_requests':0,'downloads':0,'truncated_answers':0,'dropped_official_rows':0}
write('results/CHOICE_LOGPROBS.jsonl',choices);write('results/PER_ITEM_METRICS.jsonl',items)
dump('results/RESULTS.json',{'status':status,'complete':complete,'numeric_policy':'FP32_V2','results':results,'counters':counts,'fixed_reference':'PetitGPT-alpha075 unchanged','v1_status':'BLOCKED_LIKELIHOOD_PARITY_FAILED'})
with (R/'results/RESULTS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
dump('checks/FINAL_AUDIT.json',{'status':'PASS_FOR_REPORTED_STATE','coverage':coverage,'independent_metric_recomputation':True,'model_source_input_code_hashes_unchanged':True,'V1_and_recovery_package_original_bytes_preserved':True,'no_missing_returned_calls':len(calls)==len(returns),'parity':parity_stats,'counters':counts})
header='| Model | Task | N / expected | acc | acc_norm | Status |\n|---|---|---:|---:|---:|---|\n'
def val(r,k):return 'Not measured' if r[k] is None else f"{r[k+'_correct']}/{r['documents']} ({100*r[k]:.2f}%)"
table=header+'\n'.join(f"| {r['model']} | {r['task']} / {r['split']} | {r['documents']} / {r['expected_documents']} | {val(r,'acc')} | {val(r,'acc_norm')} | {r['status']} |" for r in results)
protocol=f'''The native protocol-compatible evaluator uses the unchanged V1 official source snapshot at EleutherAI/lm-evaluation-harness commit {p['harness_commit']}; lm_eval is not installed and full installed-harness execution is not claimed. Task texts, candidate order, token policies and metric semantics are byte/identity-bound to V1. Only the new measurement's numerical path was amended by the owner before model outputs.

Zero-shot raw completion: Question: <unchanged question or goal> followed by a newline and Answer:. Each candidate continuation is one ASCII space plus the unchanged official answer text. No chat/system/roles, few-shot examples, BOS insertion, scored EOS, letter-label likelihood, generation, rationale, cleanup, filtering or truncation. Each model uses its own tokenizer, add_special_tokens=False, and the unchanged AST-extracted TemplateLM._encode_pair. All stored request encodings were regenerated in memory for comparison and matched V1 exactly, with zero context-prefix mismatches.

Input is (context_ids+continuation_ids)[:-1]. All K target IDs use logit positions C-1 through C+K-2, including the first target; context and future suffixes are not scored. The primary route is unpadded batch 1, no KV cache, no compile, FP32 parameters and actual FP32 logits from the entire forward with CUDA autocast explicitly disabled. Both CUDA matmul and cuDNN TF32 are disabled, cuDNN benchmark is disabled, float32 matmul precision is highest, and sdpa_kernel([SDPBackend.MATH]) wraps every FP32 forward. External models explicitly use official sdpa attention, use_cache=False, trust_remote_code=False, and exact local unquantized safetensors. Native sources and positions are unchanged. Returned logits are asserted FP32 before casting; selected log-softmax and sum are FP32.

acc is first argmax of full continuation log-likelihood sums; acc_norm is first argmax after division by Python len(ORIGINAL candidate TEXT), Unicode characters excluding the added delimiter. No native-token normalization, averaging of metrics or selection of the larger metric. Final aggregation independently recomputes ties, predictions and both metrics from stored numeric records.

ARC-Easy: allenai/ai2_arc @ 210d026faf9955653af8916fad021475a3f00453, config ARC-Easy, test, 2376 rows and 9501 candidates/model (2365 rows with 4 choices, 7 with 3, 4 with 5). PIQA: baber/piqa @ 142f6d7367fd9877f0fb3b5734ea6a545f54cdd1, validation, 1838 rows and 3676 candidates/model. The official parquet order is retained; PIQA derived IDs are validation: plus six-digit row index because its data lacks an upstream ID column. Frozen parquet/request/encoding files were read locally, with no network download. Maximum combined token counts remain 257/242/242 under the common 2048 limit.

The same four preselected V1 technical IDs were used for each model: {', '.join(p['technical_ids'])}. All their candidates were stored as primary FP32 measurements and reused in the full pass; the first candidate of each document was separately checked against the full unchanged pinned HFLM._loglikelihood_tokens helper using the V2 forward wrapper. No gold labels, candidate ranks or accuracy selected any numerical setting.
'''
checks=f'''The supplied 14 CPU tests passed without writing into the input package. Each of the same three V1 synthetic fixtures was measured unpadded, repeated exactly, with seven future zero IDs, with seven frozen ordinary non-special token IDs at the same padded shape, by the full unchanged HFLM helper, and by first/last target prefix-only witnesses (one distinct prefix if K=1). Every completed measurement and comparison was persisted before asserting its gate. Exact synthetic inputs, targets, C/K, scored positions, per-token logprobs, dtypes and shapes are retained.

Same-logits FP32 vector-gather versus explicit position sums use atol=1e-5, rtol=1e-6. Cross-forward comparison checks EVERY token with 5e-4 + 1e-5*max(abs(a),abs(b)); the sequence bound sums the per-token allowances. Scalar-only helper comparisons use K*5e-4 + 1e-5*max(abs(a),abs(b)). Repeated-input bit identity is recorded separately. These new FP32 engineering tolerances are not an enlarged BF16 gate or a numerical theorem. The failed V1 gate remains failed.

FP32 validation summary: {json.dumps(parity_stats,sort_keys=True)}.

Exactly four observational BF16 forwards used V1 backend settings on the already failing native fixture. They were not primary benchmark scores and were not gated against FP32 or retried. Scalar totals in order (original, repeat, future zeros, future ordinary): {[r['fp32_sum'] for r in bf]}. Original-repeat signed sequence delta: {repeat['sequence_signed_delta']}; token values bit identical: {repeat['bit_identical_token_values']}. Same-shape ordinary-versus-zero suffix signed sequence delta: {same_shape['sequence_signed_delta']}; maximum absolute per-token change: {max(abs(x) for x in same_shape['per_token_signed_delta'])}. All signed per-token deltas are retained, so cancellation is not hidden. Difference from V1's old unpadded scalar (-28.37740707397461): {bf[0]['fp32_sum']-(-28.37740707397461)}; difference from its padded scalar (-28.241230010986328): {bf[2]['fp32_sum']-(-28.241230010986328)}. No retry was used to force old scalar reproduction.
'''
limitations='''If FP32 witnesses are stable, that is evidence compatible with reduced-precision/numerical-path sensitivity, not proof that a particular kernel is guilty. Any same-shape future-suffix effect must be interpreted from its explicit recorded values; shape-only noise is not presumed. These small probes do not prove correctness for all padded shapes or historical training runs. Matching enabled flags across different official models do not imply identical actual low-level kernels.

This task concerns likelihood evaluation, not a new training-defect verdict. Prior BF16 training/generation and Stage N/O remain unchanged. Evaluating stored FP32 weights with FP32 arithmetic does not retrain or improve weights. Alpha075 remains the preselected instruction research reference regardless of ordering; the declared five-point search is closed, not mathematically exhausted. Accepted Base and other model candidates were not loaded. No claim is made that alpha075 contains later DeepSeek response-KD weights.

ARC-Easy and PIQA are previously used project diagnostics, not untouched tests. No decontamination or training-overlap audit was performed. Training data, compute, tokenizers and architectures are unmatched. Raw multiple-choice accuracy does not establish free-generation reliability, chat/tool competence, a model-size ceiling, global-best checkpoint status or SOTA. No earlier natural scores or 189-task generated-answer reviews were revised. Historical/author-reported results are omitted and remain null rather than zero; no model-card reproduction is claimed. There is no additional task, shot count, model selection, generation, training, export, upload or publication.
'''
identity='\n'.join('- '+m['id']+': '+f"{m['expected_unique_parameters']:,}"+' unique tied parameters; '+('SHA256 '+m['sha256'] if 'sha256' in m else 'revision '+m['revision'])+'.' for m in models)
report=f'''# Fixed public benchmark — FP32 V2

Status: {status}. All displayed formal scores are FP32 V2 only. V1 remains BLOCKED_LIKELIHOOD_PARITY_FAILED with its original null metrics and files unchanged. {'The fixed three-model/two-task measurement is complete.' if complete else 'The measurement is incomplete; partial or unmeasured entries are explicitly marked and do not establish a complete three-model ranking.'}

{table}

## Frozen measurement protocol

{protocol}

## Numerical validation and BF16 observations

{checks}

## Interpretation and limitations

{limitations}

## Models and actual runtime

{identity}

Native model: strict GPT state-dict load through the accepted restricted checkpoint loader; 30 layers, width 576, 9 Q/3 KV heads, FFN 1536, vocabulary 32000, context 2048, tied embeddings. The public models use their official architectures and existing exact safetensors snapshots. Unique parameter counts and FP32 parameter dtypes were measured after loading; one model was resident on GPU at a time. Checks preceded full passes for all models, with fixed technical primary scores reused.

Actual interpreter {env['sys_executable']}; prefix {env['sys_prefix']}; Python {env['python']}. Versions: {json.dumps(env['versions'],sort_keys=True)}. GPU: {env['gpu']}; driver {env['driver']}; torch CUDA {env['cuda']}. ACTUAL_ENVIRONMENT records outer-process flags; returned-forward events and identities additionally record the enforced MATH context, disabled autocast, actual FP32 logits and official attention implementation. This is policy and dtype evidence, not a profiler attribution for every kernel. No dependencies, driver or shared sources were changed.

## Counters and independent audit

{json.dumps(counts,indent=2)}

The planned formal scope is 4214 documents and 13177 candidate likelihoods per model, totalling 12642 document-model evaluations and 39531 unique primary candidates. Technical forwards are counted separately, including all four observational BF16 calls and every FP32 synthetic/repeat/padding/suffix/prefix/helper call; the cap is 192. No formal score was retried, overwritten or selected from repeated runs. Final audit verifies exact primary cardinality and unique IDs, all choices per complete document, finite scores, task/input/code/policy bindings, original denominators, stable first-index ties, recomputed acc/acc_norm, forward IDs/returns, and unchanged original inputs.

## Artifact and delivery provenance

The input package and V1 root remained read-only, including unchanged V1 archive SHA256 1778084b32efa098cffa226c51074840ee0a1df4ec751f8b4d7cb10f664ec1c6. The V2 numerical amendment, task/model/request freezes, actual source/code hashes, supplied checker, complete partial-or-passed evidence and common-helper unified diff are included. The private evaluator and orchestration are new; no V1 prepare/driver/finalizer module was executed. Immutable manifests exclude actively appended logs.

Official task and helper source is the local pinned snapshot, not a current-main fetch. See source/PROVENANCE_AND_LICENSES.md and source/harness/LICENSE.md. [Pinned ARC task](https://github.com/EleutherAI/lm-evaluation-harness/blob/{p['harness_commit']}/lm_eval/tasks/arc/arc_easy.yaml), [pinned PIQA task](https://github.com/EleutherAI/lm-evaluation-harness/blob/{p['harness_commit']}/lm_eval/tasks/piqa/piqa.yaml), [pinned HF helper](https://github.com/EleutherAI/lm-evaluation-harness/blob/{p['harness_commit']}/lm_eval/models/huggingface.py).

Complete datasets, request text and token encodings remain only in the bound V1 local_data directory. The compact evidence package contains request IDs/hashes and synthetic fixtures, not weights, full datasets, request-text dumps, model caches, credentials, environments or nested archives. PIQA's pinned repository had no README/license file; no new redistribution permission is inferred. The archive contains an internal nonrecursive payload manifest and is extracted to verify safe regular relative paths, exact file-set coverage, sizes and hashes; its SHA256 and final external receipt are outside it. No files are uploaded or published. Work stops after this measurement.
'''
(R/'report/PUBLIC_BENCHMARK_FP32_REPORT_EN.md').write_text(report)
(R/'report/PROTOCOL_AND_LIMITATIONS.md').write_text('# Protocol, numerical evidence and limits\n\n'+protocol+'\n'+checks+'\n'+limitations)
zh=f'''# FP32 公共基准恢复评测

状态：{status}。本次正式成绩仅使用预先授权的 FP32 前向、关闭 TF32、math SDPA、batch 1、无 padding 协议。V1 的 BF16 校验失败记录和未测量结果保持不变。

{table}

acc 使用完整候选对数似然，acc_norm 按原始候选答案的 Unicode 字符数归一化，分别报告。{'三个模型的固定数值校验均通过，已完成全部正式测量。' if complete else '当前未完成全部测量，部分或缺失成绩不能当作完整排名。'}正式候选 {len(choices):,} 个，完整题目—模型评测 {len(items):,} 次，技术额外前向 {len(extras)} 次，其中 BF16 诊断恰为 4 次；总前向 {len(calls):,} 次。14 项 CPU 测试通过，所有原始输入字节在结束时复核保持不变。

BF16 四次诊断的序列总分依次为 {[round(r['fp32_sum'],8) for r in bf]}；同形状普通后缀与零后缀的差为 {same_shape['sequence_signed_delta']:.8g}。完整逐 token 差值与 FP32 重复、后缀、前缀和参考实现校验证据已保存。稳定的 FP32 见证与低精度/数值路径敏感性相容，不能证明某个内核是根因，也不能推广为全部历史训练的正确性证明。

alpha075 仍是事先固定的研究参考。ARC/PIQA 曾用作历史诊断，不是未接触测试；原始多选准确率不等同于可靠自由生成、聊天能力或模型容量上限。未训练、改权重、选新模型、重新评审自然任务、下载新数据、导出或发布。
'''
(R/'report/OWNER_SUMMARY_ZH.md').write_text(zh)
print(json.dumps({'status':status,'results':results,'counters':counts,'parity':parity_stats},indent=2))
