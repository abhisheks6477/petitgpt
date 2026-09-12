"""One final CPU-only materialization from immutable eligible pool + bounded reviews.
No corpus rescreening, data rewriting, inference, optimizer or dataset-code execution.
"""
from __future__ import annotations
import ast,copy,importlib.util,json,math,re,sys,time
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('p2_builder',OUT/'prepare_p2_data.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
TASKS=b.TASKS

def vector(r):return [r['task_supervised_tokens'].get(t,0) for t in TASKS]
def pure_function_evidence(text):
    blocks=[body for lang,body in b.old.code_blocks(text) if lang.strip().lower() in ('','python','python3','py')]
    if not blocks and re.match(r'\s*(?:import\s|from\s|def\s)',text):blocks=[text]
    functions=[]
    for code in blocks:
        try:tree=ast.parse(code)
        except SyntaxError:continue
        for fn in ast.walk(tree):
            if isinstance(fn,ast.FunctionDef):
                bad=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'print','input','open','exec','eval','__import__'} for n in ast.walk(fn))
                if not bad:functions.append(fn.name)
    return sorted(set(functions))

def apply_attribution(r,finding):
    before=dict(r['task_supervised_tokens']);changes=[]
    explicit_task=finding.get('task');overrides=finding.get('per_turn_task_overrides',{})
    if r['p2_task']=='PYTHON_BASIC' and explicit_task is None:
        evidence={str(t['message_index']):pure_function_evidence(r['messages'][t['message_index']]['content']) for t in r['per_turn']}
        if not any(evidence.values()):
            assistant='\n'.join(m['content'] for m in r['messages'] if m['role']=='assistant')
            actual_code=bool(b.old.code_blocks(assistant) or re.search(r'`[^`\n]*(?:\w\(|\w\s*=)[^`\n]*`',assistant) or re.match(r'\s*(?:def |import |from |\w+\s*=)',assistant))
            if actual_code:return 'DEFER_BASIC_PYTHON_NONFUNCTION_RECIPE_SCOPE',None
            explicit_task='GENERAL_QA'
        else:
            r['python_function_ast_evidence']=evidence
            # Keep explanations attached to a real Python function as contextual
            # Python targets, with an explicit code-containing lower bound.
            r['python_function_code_turn_lower_tokens']=sum(t['supervised_tokens_with_eos'] for t in r['per_turn'] if evidence[str(t['message_index'])])
    tt=Counter()
    for t in r['per_turn']:
        task=overrides.get(str(t['message_index']),explicit_task or t['task'])
        if task!=t['task']:changes.append({'message_index':t['message_index'],'before':t['task'],'after':task})
        t['task']=task;tt[task]+=t['supervised_tokens_with_eos']
    r['task_supervised_tokens']=dict(tt)
    r['p2_task']=explicit_task or max(TASKS,key=lambda t:tt[t])
    b.require(sum(tt.values())==r['shifted_supervised_tokens'],'attribution denominator changed')
    return None,({'audit_id':r['audit_id'],'before':before,'after':dict(tt),'per_turn_changes':changes} if before!=dict(tt) else None)

def calibrated_choose(pool,target):
    buckets={t:sorted([r for r in pool if r['p2_task']==t],key=lambda r:(r['max_assistant_content_tokens']>128,b.rank(r['audit_id'],'selection'))) for t in TASKS}
    prefixes={}
    for t,rs in buckets.items():
        vs=[[0]*5]
        for r in rs:vs.append([x+y for x,y in zip(vs[-1],vector(r))])
        prefixes[t]=vs
    # All search operations below are arithmetic on cumulative five-dimensional
    # target vectors. They neither reclassify nor recheck corpus content.
    caps=[len(buckets[t]) for t in TASKS]
    counts=[min(caps[i],int(target*b.WEIGHTS[t])) for i,t in enumerate(TASKS)]
    for i in range(5):
        add=min(target-sum(counts),caps[i]-counts[i]);counts[i]+=add
    b.require(sum(counts)==target,'eligible split supply below target')
    weights=[b.WEIGHTS[t] for t in TASKS]
    bounds=[(.40,.48),(.17,.25),(.19,.27),(.05,.10),(.03,.08)]
    def amounts(ns):return [sum(prefixes[t][ns[i]][j] for i,t in enumerate(TASKS)) for j in range(5)]
    def loss(ns):
        amounts_=amounts(ns);total=sum(amounts_);fs=[n/total for n in amounts_]
        return sum((f-w)**2 for f,w in zip(fs,weights))+100*sum(max(0,lo-f,f-hi)**2 for f,(lo,hi) in zip(fs,bounds))
    steps=0;initial_counts=list(counts);initial_vector=amounts(counts)
    for stride in (128,32,8,1):
        for _ in range(400):
            current=loss(counts);best=current;choice=None
            for i in range(5):
                if counts[i]<stride:continue
                for j in range(5):
                    if i==j or counts[j]+stride>caps[j]:continue
                    trial=list(counts);trial[i]-=stride;trial[j]+=stride;score=loss(trial)
                    if score<best-1e-16:best=score;choice=trial
            if choice is None:break
            counts=choice;steps+=1
    chosen=[r for i,t in enumerate(TASKS) for r in buckets[t][:counts[i]]]
    chosen.sort(key=lambda r:b.rank(r['audit_id'],'dataset-order'))
    return chosen,{'method':'one_bounded_coordinate_calibration_of_actual_five_task_prefix_vectors','strides':[128,32,8,1],'max_arithmetic_steps_per_stride':400,'accepted_arithmetic_steps':steps,'initial_counts':dict(zip(TASKS,initial_counts)),'final_counts':dict(zip(TASKS,counts)),'initial_target_vector':dict(zip(TASKS,initial_vector)),'final_target_vector':dict(zip(TASKS,amounts(counts))),'available_rows':dict(zip(TASKS,caps)),'no_content_rescreening':True}

def main():
    start=time.perf_counter();b.require(not (OUT/'FROZEN_MANIFEST.json').exists(),'already frozen')
    draft=b.read(OUT/'DRAFT_MANIFEST.json');preview=b.rows(OUT/'PREVIEW60_FULL_MESSAGES.jsonl');by_preview={r['audit_id']:r for r in preview}
    b.require(b.sha(OUT/'PREVIEW60_FULL_MESSAGES.jsonl')==draft['preview']['sha256'],'original preview changed')
    reviews=sorted(OUT.glob('PREVIEW_REVIEW_*.json'));covered=set();findings={};review_identities={}
    for p in reviews:
        doc=b.read(p);b.require(doc.get('preview_sha256',doc.get('source_preview_sha256'))==draft['preview']['sha256'],'review preview identity mismatch')
        indices=doc.get('fully_read_preview_indices',doc.get('preview_indices'));
        if indices is None:indices=list(range(doc['range'][0],doc['range'][1]+1))
        b.require(not covered.intersection(indices),'overlapping reviewer assignments');covered.update(indices)
        review_identities[p.name]={'sha256':b.sha(p),'fully_read_preview_indices':indices}
        for f in doc['findings']:
            aid=f['audit_id'];b.require(aid in by_preview,'review finding outside original preview')
            b.require(by_preview[aid]['preview_index'] in indices,'review finding outside assignment')
            b.require(aid not in findings,'duplicate review finding');
            if f.get('messages_sha256'):b.require(f['messages_sha256']==by_preview[aid]['messages_sha256'],'review messages identity mismatch')
            findings[aid]=f
    b.require(covered==set(range(1,61)),'full deterministic 60-row review incomplete')
    check_path=OUT.parent/'content_checks/TWO_FUNCTION_CONTENT_CHECK_RESULTS.json';function_checks=b.read(check_path)
    b.require(function_checks['status']=='BOUNDED_CONTENT_CHECKS_PASSED' and function_checks['preview_sha256']==draft['preview']['sha256'],'bounded function checks incomplete or changed')
    for c in function_checks['candidates']:
        b.require(c['functional_status']=='FUNCTIONAL_PASS' and c['real_exit_code']==0 and c['messages_sha256_from_preview']==by_preview[c['audit_id']]['messages_sha256'],'bounded function check failed or identity mismatch')
    checked_ids={c['audit_id'] for c in function_checks['candidates']}
    b.require(all(f['audit_id'] in checked_ids for f in findings.values() if f['action']=='KEEP_PENDING_FUNCTION_TEST'),'unresolved required function check')
    eligible=b.rows(OUT/'ELIGIBLE_POOL.jsonl');by_aid={r['audit_id']:r for r in eligible}
    deferred_groups={by_aid[aid]['group_id'] for aid,f in findings.items() if f['action'].startswith('DEFER')}
    decisions=[];pool=[];attribution=[]
    for r in eligible:
        b.require(r['messages_sha256']==b.msgsha(r),'eligible original message identity mismatch')
        f=findings.get(r['audit_id'],{});reason=None
        if r['group_id'] in deferred_groups:reason='BOUNDED_PREVIEW_'+f.get('reason','related_prompt_group_of_explicit_preview_deferral')
        if reason is None:reason,change=apply_attribution(r,f)
        else:change=None
        if change:attribution.append(change)
        if reason:decisions.append({'audit_id':r['audit_id'],'source':r['source'],'original_row_index':r['original_row_index'],'messages_sha256':r['messages_sha256'],'group_id':r['group_id'],'decision':'DEFER','reason':reason})
        else:pool.append(r)
    train,train_cal=calibrated_choose([r for r in pool if r['assigned_split']=='train'],12000)
    val,val_cal=calibrated_choose([r for r in pool if r['assigned_split']=='validation'],500)
    allrows=train+val;stats={'train':b.summarize(train),'validation':b.summarize(val),'candidate':b.summarize(allrows)}
    for split,rs in [('train',train),('validation',val),('candidate',allrows)]:
        stats[split]['python_function_attribution']={'function_AST_rows':sum(bool(r.get('python_function_ast_evidence')) for r in rs),'code_containing_turn_target_lower':sum(r.get('python_function_code_turn_lower_tokens',0) for r in rs),'contextual_function_target_upper':sum(r['task_supervised_tokens'].get('PYTHON_BASIC',0) for r in rs),'AST_is_not_functional_correctness':True}
    ts=stats['train'];guards={'train_rows_8000_16000':8000<=len(train)<=16000,'validation_rows_300_800':300<=len(val)<=800,'python_5_10':.05<=ts['task_supervised_fraction']['PYTHON_BASIC']<=.10,'magpie_le30':ts['source_supervised_fraction'].get('smol-magpie-ultra-short',0)<=.30,'short128_ge60':ts['assistant_messages_le128_fraction']>=.60,'short256_ge90':ts['assistant_messages_le256_fraction']>=.90,'zero_cross_split_groups':not ({r['group_id'] for r in train}&{r['group_id'] for r in val}),'no_deferred_group':not ({r['group_id'] for r in allrows}&deferred_groups)}
    b.require(all(guards.values()),'unresolved final data guard: '+json.dumps(guards))
    # The eligible pool was formatter-counted once. Verify only final selected
    # rows against the actual trainer API, without inference or truncation.
    import torch
    torch.set_num_threads(4)
    from src.chat_template import load_chat_tokenizer
    from src.special_tokens import PAD_ID
    from sft.train_sft import build_example
    b.require(b.sha(b.old.TOKENIZER)==b.old.TOKENIZER_SHA,'tokenizer identity changed')
    tok=load_chat_tokenizer(str(b.old.TOKENIZER))
    for r in allrows:
        x,y,w=build_example(r,tok,2048,None,1.0,[],'contains_any')
        b.require(len(x)==2048 and int((x!=PAD_ID).sum())==r['sequence_tokens'] and bool((y[r['sequence_tokens']:]==-100).all()) and int((y[1:]!=-100).sum())==r['shifted_supervised_tokens'] and float(w)==1.0,'actual trainer sequence/target denominator changed')
    for name,rs in [('P2_TRAIN.jsonl',train),('P2_VALIDATION.jsonl',val),('P2_CANDIDATE.jsonl',allrows),('P2_FINAL_DEFERRALS.jsonl',decisions),('P2_TASK_ATTRIBUTION_CORRECTIONS.jsonl',attribution)]:b.save_rows(name,rs)
    names=['P2_TRAIN.jsonl','P2_VALIDATION.jsonl','P2_CANDIDATE.jsonl','P2_FINAL_DEFERRALS.jsonl','P2_TASK_ATTRIBUTION_CORRECTIONS.jsonl']
    selected={r['audit_id'] for r in allrows}
    manifest={**{k:v for k,v in draft.items() if k not in ('train','validation','candidate','guard_status','files','wall_seconds','status','CONTENT_REVIEW_PENDING')},**stats,'status':'FROZEN_BOUNDED_CONTENT_REVIEW_COMPLETE','CONTENT_REVIEW_PENDING':False,'TRAINING_STARTED':False,'CONTENT_QUALITY_UNIVERSALLY_VERIFIED':False,'guard_status':guards,'files':{n:{'sha256':b.sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in names},'review_identities':review_identities,'bounded_function_checks':{'path':str(check_path),'sha256':b.sha(check_path),'audits':sorted(checked_ids),'pass_scope':'only recorded original and boundary tests, not general correctness'},'original_preview_preserved':True,'original_preview_final_selected_count':len(selected&set(by_preview)),'review_defer_count':sum(f['action'].startswith('DEFER') for f in findings.values()),'final_deferrals_count':len(decisions),'task_attribution_corrections_count':len(attribution),'calibration':{'train':train_cal,'validation':val_cal},'eligible_after_bounded_review_and_python_attribution':len(pool),'eligible_pool_input_sha256':b.sha(OUT/'ELIGIBLE_POOL.jsonl'),'finalizer_sha256':b.sha(__file__),'actual_final_build_example_rows_verified':len(allrows),'wall_seconds':time.perf_counter()-start}
    manifest['limitations']=[s for s in draft['limitations'] if 'pending full-message preview' not in s and s!='Python uses inherited AST/scope checks only; no candidate program executed.']+['Selection/finalization uses static AST/scope checks only. A separate bounded check executed only full-read preview #8 and #25; its exact 21 passing cases and process exits are recorded in bounded_function_checks. Other programs remain functionally unverified.','The complete fixed 60-row preview has been reviewed once; this cannot estimate population accuracy. Only the recorded findings and arithmetic mix were changed after preview.','Python function AST evidence establishes structure and scope, not program correctness; code-containing per-turn lower bound is recorded separately from contextual Python targets.']
    b.save('FROZEN_MANIFEST.json',manifest)
    b.save('FINAL_PREVIEW_MEMBERSHIP.json',{'original_preview_sha256':draft['preview']['sha256'],'rows':[{'preview_index':r['preview_index'],'audit_id':r['audit_id'],'finally_selected':r['audit_id'] in selected,'review_action':findings.get(r['audit_id'],{}).get('action'),'final_split':next((v['assigned_split'] for v in allrows if v['audit_id']==r['audit_id']),None)} for r in preview]})
    print(json.dumps({'status':manifest['status'],'train':stats['train'],'validation':stats['validation'],'files':manifest['files'],'guards':guards}),flush=True)

if __name__=='__main__':main()
