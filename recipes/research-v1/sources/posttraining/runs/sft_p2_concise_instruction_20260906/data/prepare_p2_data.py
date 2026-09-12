"""Deterministic CPU-only P2 candidate selection and pending full-message preview.
No teacher, network, model, optimizer, corpus-code execution or message rewriting.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,math,re,sys,time
from collections import Counter,defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('.');OUT=Path(__file__).resolve().parent
PREV=ROOT/'runs/D4_completion_and_SFT_pilot_handoff/pilot_v2_preparation'
D4=ROOT/'runs/D4_flash_focused_fastpass/d4_flash_focused_fastpass_v1'
D1=Path('artifacts/instruction_preparation/d1_2_targeted_content_quality_cleanup_v1/output')
D2=Path('artifacts/instruction_preparation/d2_semantic_quality_sample_v1/output')
SFT99=ROOT/'runs/sft_pilot_99step_preparation_20260906'
OLD_SCRIPT=PREV/'selection/build_pilot_v2.py'
spec=importlib.util.spec_from_file_location('prior_pilot_scope',OLD_SCRIPT);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sys.path.insert(0,str(ROOT))
SEED=20260906;DOMAIN='SFT_P2_CONCISE_INSTRUCTION_V1'
TASKS=['GENERAL_QA','INSTRUCTION','TEXT_TRANSFORM','PYTHON_BASIC','PRACTICAL_CHAT']
WEIGHTS=dict(zip(TASKS,[.45,.20,.22,.08,.05]))
EXTRA={'smol-contraints','smol-summarize-20k','smollm-rewrite-30k','everyday-conversations'}
NUM={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(p):return old.sha(p)
def rank(x,purpose='rank'):return hashlib.sha256(f'{DOMAIN}\0{SEED}\0{purpose}\0{x}'.encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return [json.loads(l) for l in f if l.strip()]
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def save_rows(name,data):
    with (OUT/name).open('w') as f:
        for row in data:f.write(canonical(row)+'\n')
def require(x,msg):
    if not x:raise RuntimeError(msg)
def identity(r):return (r.get('source'),r.get('original_row_index'))
def msgsha(r):return hashlib.sha256(canonical(r['messages']).encode()).hexdigest()
def asnum(x):return int(x) if x.isdigit() else NUM[x.lower()]
def words(s):return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?",s)
def sentences(s):
    s=re.sub(r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.',lambda m:m[0][:-1],s)
    s=re.sub(r'P\.S\.', 'PS',s)
    return [v.strip() for v in re.split(r'[.!?]+(?=\s|$)',s) if v.strip()]

# Finite common template checks. These are syntax/explicit-constraint checks,
# never a claim to universally validate natural-language instructions.
COUNT_RE=re.compile(r'\b(exactly|at least|at most|less than|more than|no more than|up to)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(words?|sentences?|bullet points?|paragraphs?|lines?|placeholders?)\b',re.I)
def compare(n,op,k):return {'exactly':n==k,'at least':n>=k,'at most':n<=k,'less than':n<k,'more than':n>k,'no more than':n<=k,'up to':n<=k}[op.lower()]

def constraint_checks(prompt,answer,strict=False):
    checks=[]; failures=[]
    for m in COUNT_RE.finditer(prompt):
        op,k,unit=m.groups();k=asnum(k);unit=unit.lower()
        if unit.startswith('word'): n=len(words(answer))
        elif unit.startswith('sentence'):n=len(sentences(answer))
        elif unit.startswith('bullet'):n=len(re.findall(r'^\s*[-*+]\s+\S',answer,re.M))
        elif unit.startswith('paragraph'):n=len([p for p in re.split(r'\n\s*\n',answer) if p.strip()])
        elif unit.startswith('line'):n=len([p for p in answer.splitlines() if p.strip()])
        else:n=len(re.findall(r'\[[^\[\]\n]+\]',answer))
        ok=compare(n,op,k);checks.append({'kind':unit,'requirement':m[0],'actual':n,'pass':ok})
        if not ok:failures.append('explicit_'+unit+'_count')
    for m in re.finditer(r'\bword\s+["“]([^"”]+)["”]\s+should appear\s+(at least|exactly|at most)\s+(\d+)\s+times?',prompt,re.I):
        word,op,k=m.groups();n=len(re.findall(r'(?<!\w)'+re.escape(word)+r'(?!\w)',answer,re.I));ok=compare(n,op,int(k))
        checks.append({'kind':'word_frequency','word':word,'actual':n,'required':int(k),'pass':ok})
        if not ok:failures.append('explicit_word_frequency')
    for m in re.finditer(r'\b(?:include|contain) keywords\s*\[([^\]]+)\]',prompt,re.I):
        if m[1].lower()=='forbidden_words':continue
        missing=[w.strip(' \"\'') for w in m[1].split(',') if not re.search(r'(?<!\w)'+re.escape(w.strip(' \"\''))+r'(?!\w)',answer,re.I)]
        checks.append({'kind':'required_keywords','missing':missing,'pass':not missing})
        if missing:failures.append('explicit_required_keyword')
    forbidden=re.search(r'\[forbidden_words\]\s*(?:are|is)\s*:\s*([^\n.]+)',prompt,re.I)
    if forbidden:
        bad=[w.strip(' \"\'') for w in forbidden[1].split(',') if re.search(r'(?<!\w)'+re.escape(w.strip(' \"\''))+r'(?!\w)',answer,re.I)]
        checks.append({'kind':'forbidden_words','present':bad,'pass':not bad})
        if bad:failures.append('explicit_forbidden_word')
    if re.search(r'\ball lowercase\b|\bno capital letters\b',prompt,re.I):
        ok=not any(c.isupper() for c in answer);checks.append({'kind':'lowercase','pass':ok})
        if not ok:failures.append('explicit_lowercase')
    if re.search(r'\ball (?:uppercase|capital) letters\b|\ball uppercase\b',prompt,re.I):
        ok=not any(c.islower() for c in answer);checks.append({'kind':'uppercase','pass':ok})
        if not ok:failures.append('explicit_uppercase')
    if 'double angular brackets' in prompt.lower():
        ok=bool(re.search(r'<<[^<>\n]+>>',answer));checks.append({'kind':'bracket_title','pass':ok})
        if not ok:failures.append('explicit_bracket_title')
    if re.search(r'\bpostscript\b',prompt,re.I):
        ok=bool(re.search(r'P\.S\.',answer));checks.append({'kind':'postscript','pass':ok})
        if not ok:failures.append('explicit_postscript')
    if re.search(r'highlighted section.*markdown',prompt,re.I):
        ok=bool(re.search(r'\*[^*\n]+\*',answer));checks.append({'kind':'markdown_highlight','pass':ok})
        if not ok:failures.append('explicit_markdown_highlight')
    if re.search(r'finish your response with this exact phrase',prompt,re.I):
        # This is the existing source's fixed, visibly defined ending template.
        ending='Is there anything else I can help with?'
        if ending not in prompt:failures.append('unsupported_exact_ending_template')
        else:
            ok=answer.rstrip().endswith(ending);checks.append({'kind':'exact_ending','expected':ending,'pass':ok})
            if not ok:failures.append('explicit_exact_ending')
    if re.search(r'(?:output|return|respond|answer|format).{0,45}\b(?:only )?(?:a |an )?(?:valid )?json(?: object| array)?\b',prompt,re.I):
        try:json.loads(answer);ok=True
        except ValueError:ok=False
        checks.append({'kind':'parsable_json_only','pass':ok})
        if not ok:failures.append('explicit_json_parse')
    if strict:
        if re.search(r'\[(?:concept|phenomenon|topic|question|keyword|relation|num_words|frequency)\]',prompt,re.I):failures.append('unresolved_substantive_input_placeholder')
        if re.search(r'\b(?:nth|every (?:word|sentence|paragraph)|each (?:word|sentence|paragraph)|alternat(?:e|ing)|palindrome|rhyme|acrostic|syllable|alliteration)\b',prompt,re.I):failures.append('constraint_outside_finite_check_scope')
        if not checks:failures.append('no_supported_explicit_constraint')
        # Defer answer copied wholesale into a question instead of an explicit
        # transformation/copying instruction. No answer templates are synthesized.
        if len(words(answer))>12 and old.norm(answer) in old.norm(prompt) and not re.search(r'\b(?:repeat|copy|reproduce|rewrite)\b',prompt,re.I):failures.append('answer_embedded_in_question_ambiguous_task')
    return checks,failures


def classify(row):
    source=row['source'];ms=row['messages'];u='\n'.join(m['content'] for m in ms if m['role']=='user');a='\n'.join(m['content'] for m in ms if m['role']=='assistant')
    if source not in EXTRA:
        reason,t=old.scope_decision(row,{})
        if reason:return reason,None,[]
        task={'FOUNDATION':'GENERAL_QA','TEXT_TRANSFORM':'TEXT_TRANSFORM','PYTHON_BASIC':'PYTHON_BASIC','PRACTICAL_CHAT':'PRACTICAL_CHAT'}[t]
        if task=='GENERAL_QA' and (old.CHECKABLE.search(u) or row.get('task_type')=='INSTRUCTION'):task='INSTRUCTION'
    else:
        text=u+'\n'+a
        if old.TRANSLATION.search(u):return 'nonenglish_target',None,[]
        letters=[c for c in a if c.isalpha()]
        if letters and sum(ord(c)>127 for c in letters)/len(letters)>.08:return 'nonenglish_answer',None,[]
        if old.ADVANCED.search(text) or old.NICHE.search(u):return 'advanced_or_specialized_claims',None,[]
        if old.LIBRARY.search(text):return 'dependency_or_specialized_scope',None,[]
        if old.ROLEPLAY.search(u):return 'elaborate_roleplay',None,[]
        if old.PROGRAM.search(text) or old.PY.search(text):return 'programming_in_unjudged_extra_source',None,[]
        task='INSTRUCTION' if source=='smol-contraints' else 'PRACTICAL_CHAT' if source=='everyday-conversations' else 'TEXT_TRANSFORM'
    checks=[]
    for i,m in enumerate(ms):
        if m['role']!='assistant':continue
        context='\n'.join(v['content'] for v in ms[:i] if v['role'] in ('user','system'))
        # D1 transforms have an explicit original system, which is retained.
        if source in ('smol-summarize-20k','smollm-rewrite-30k'):
            user=next(v['content'] for v in reversed(ms[:i]) if v['role']=='user')
            answer=m['content'];system=ms[0]['content']
            if re.search(r'\d+(?:[.,:]\d+)*',answer):
                unknown=set(re.findall(r'\d+(?:[.,:]\d+)*',answer))-set(re.findall(r'\d+(?:[.,:]\d+)*',user))
                if unknown:return 'transform_added_numeric_detail',None,checks
            if 'one very short sentence' in system and len(sentences(answer))!=1:return 'summary_sentence_count',None,checks
            if 'up to three sentences' in system:
                if len(sentences(answer))>3:return 'summary_sentence_count',None,checks
                if re.search(r'\b(?:you|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves)\b',answer,re.I):return 'inherited_summary_pronoun_requirement',None,checks
            if 'more concise' in system and len(words(answer))>=len(words(user)):return 'rewrite_not_shorter',None,checks
            checks.append({'kind':'grounded_transform_numeric_and_known_system_constraints','pass':True,'semantic_preservation':'NOT_UNIVERSALLY_VERIFIED'})
        elif task!='PYTHON_BASIC':
            cs,fs=constraint_checks(context,m['content'],source=='smol-contraints');checks.extend(cs)
            if fs:return fs[0],None,checks
    return None,task,checks


def metadata(r,tok):
    from src.chat_template import encode_chat
    ids,labels=encode_chat(tok,r['messages'],default_system=None)
    require(len(ids)==len(labels),'formatter ids/labels length mismatch')
    if len(ids)>2048:raise ValueError('formatter_over2048_no_truncation')
    turns=[];tasktokens=Counter();user='';previous=None
    for i,m in enumerate(r['messages']):
        if m['role']=='user':user=m['content']
        if m['role']=='assistant':
            n=len(tok.encode(m['content']).ids)
            if n>256:raise ValueError('assistant_over256_selection_limit')
            if n==0:raise ValueError('empty_assistant_content')
            task=r['p2_task']
            if task not in ('PYTHON_BASIC','PRACTICAL_CHAT','TEXT_TRANSFORM'):
                if old.TRANSFORM.search(user):task='TEXT_TRANSFORM'
                elif old.CHECKABLE.search(user):task='INSTRUCTION'
                else:task='GENERAL_QA' if task=='GENERAL_QA' else task
            tasktokens[task]+=n+1
            turns.append({'message_index':i,'content_tokens':n,'supervised_tokens_with_eos':n+1,'task':task})
    supervised=sum(v!=-100 for v in labels[1:])
    require(sum(tasktokens.values())==supervised,'actual shifted target attribution mismatch')
    require(sum(v==3 for v in labels[1:])==len(turns),'assistant EOS target mismatch')
    return {'sequence_tokens':len(ids),'shifted_supervised_tokens':supervised,'assistant_turns':len(turns),
      'assistant_content_tokens_total':supervised-len(turns),'per_turn':turns,'task_supervised_tokens':dict(tasktokens),
      'max_assistant_content_tokens':max(t['content_tokens'] for t in turns),'messages_sha256':msgsha(r)}


def summarize(rs):
    st=Counter();sr=Counter();tt=Counter();tr=Counter();lens=[];conv=[];seq=[]
    for r in rs:
        st[r['source']]+=r['shifted_supervised_tokens'];sr[r['source']]+=1;tt.update(r['task_supervised_tokens']);tr[r['p2_task']]+=1
        lens.extend(t['content_tokens'] for t in r['per_turn']);conv.append(r['assistant_content_tokens_total']);seq.append(r['sequence_tokens'])
    def dist(v):
        v=sorted(v)
        return {**{f'p{p}':v[min(len(v)-1,math.ceil(len(v)*p/100)-1)] if v else None for p in (10,50,90,95,99)},'max':max(v,default=None)}
    total=sum(st.values())
    return {'rows':len(rs),'assistant_turns':len(lens),'shifted_supervised_tokens_including_eos':total,
      'source_rows':dict(sr),'source_supervised_tokens':dict(st),'source_supervised_fraction':{k:v/total for k,v in st.items()} if total else {},
      'task_rows':dict(tr),'task_supervised_tokens':dict(tt),'task_supervised_fraction':{k:tt[k]/total if total else 0 for k in TASKS},
      'assistant_content_length':dist(lens),'conversation_assistant_total_length':dist(conv),'sequence_length':dist(seq),
      'assistant_messages_le128':sum(v<=128 for v in lens),'assistant_messages_le256':sum(v<=256 for v in lens),
      'assistant_messages_le128_fraction':sum(v<=128 for v in lens)/len(lens) if lens else 0,
      'assistant_messages_le256_fraction':sum(v<=256 for v in lens)/len(lens) if lens else 0}


def choose(pool,target):
    buckets={t:sorted([r for r in pool if r['p2_task']==t],key=lambda r:(r['max_assistant_content_tokens']>128,rank(r['audit_id'],'selection'))) for t in TASKS}
    means={t:sum(r['shifted_supervised_tokens'] for r in b[:target])/min(target,len(b)) if b else 100 for t,b in buckets.items()}
    selected=[]
    # At most three small arithmetic quota calibrations, never another screening.
    for _ in range(3):
        units={t:WEIGHTS[t]/means[t] for t in TASKS};den=sum(units.values())
        quotas={t:min(len(buckets[t]),int(target*units[t]/den)) for t in TASKS}
        chosen={t:buckets[t][:quotas[t]] for t in TASKS}
        for t,b in chosen.items():
            if b:means[t]=sum(r['shifted_supervised_tokens'] for r in b)/len(b)
        selected=[r for t in TASKS for r in chosen[t]]
    used={r['audit_id'] for r in selected}
    remainder=sorted([r for r in pool if r['audit_id'] not in used and r['p2_task']!='PYTHON_BASIC'],key=lambda r:(r['source']=='smol-magpie-ultra-short',r['max_assistant_content_tokens']>128,rank(r['audit_id'],'backfill')))
    selected.extend(remainder[:max(0,target-len(selected))]);used={r['audit_id'] for r in selected}
    # Remove excess source/Python weight and replace with already eligible rows.
    spare=[r for r in remainder if r['audit_id'] not in used]
    def sums():
        total=sum(r['shifted_supervised_tokens'] for r in selected)
        py=sum(r['task_supervised_tokens'].get('PYTHON_BASIC',0) for r in selected)
        mag=sum(r['shifted_supervised_tokens'] for r in selected if r['source']=='smol-magpie-ultra-short')
        return total,py,mag
    total,py,mag=sums()
    for r in sorted([r for r in selected if r['source']=='smol-magpie-ultra-short'],key=lambda r:rank(r['audit_id'],'source-cap'),reverse=True):
        if mag/total<=.30:break
        selected.remove(r);total-=r['shifted_supervised_tokens'];mag-=r['shifted_supervised_tokens'];py-=r['task_supervised_tokens'].get('PYTHON_BASIC',0)
        replacement=next((v for v in spare if v['source']!='smol-magpie-ultra-short'),None)
        if replacement:spare.remove(replacement);selected.append(replacement);total+=replacement['shifted_supervised_tokens']
    for r in sorted([r for r in selected if r['p2_task']=='PYTHON_BASIC'],key=lambda r:rank(r['audit_id'],'python-cap'),reverse=True):
        if py/total<=.10:break
        selected.remove(r);total-=r['shifted_supervised_tokens'];py-=r['task_supervised_tokens'].get('PYTHON_BASIC',0)
        replacement=next((v for v in spare if v['source']!='smol-magpie-ultra-short'),None)
        if replacement:spare.remove(replacement);selected.append(replacement);total+=replacement['shifted_supervised_tokens']
    used={r['audit_id'] for r in selected}
    for r in buckets['PYTHON_BASIC']:
        if py/total>=.065:break
        if r['audit_id'] in used:continue
        replace=next((v for v in reversed(selected) if v['p2_task']!='PYTHON_BASIC'),None)
        if not replace:break
        selected.remove(replace);total-=replace['shifted_supervised_tokens']
        selected.append(r);used.add(r['audit_id']);total+=r['shifted_supervised_tokens'];py+=r['shifted_supervised_tokens']
    return sorted(selected,key=lambda r:rank(r['audit_id'],'dataset-order'))


def main():
    start=time.perf_counter();require(not (OUT/'FROZEN_MANIFEST.json').exists(),'P2 data already frozen')
    require(sha(D1/'D1_2_SFT_CANDIDATE.jsonl')=='5779989520176a29cb43799963dd4e12078d2cf186e20bef23df65f6c95e9855','D1.2 input SHA differs')
    status=read(D4/'RUN_STATUS.json');require(status['terminal'] and status['stop_reason']=='complete' and status['route_counts']['KEEP_PROVISIONAL']==36471,'D4 final completed pool absent')
    prior=read(D1/'D1_2_SUMMARY.json');all_d4=rows(D4/'MAIN_RESULTS.jsonl');by_aid={r['audit_id']:r for r in all_d4}
    keeps=[r for r in all_d4 if r['route']=='KEEP_PROVISIONAL'];require(len(keeps)==36471,'D4 final KEEP count differs')
    cleanfields=['audit_id','source','original_row_index','messages','task_type','quality','judgment']
    candidates=[{**{k:r[k] for k in cleanfields},'origin_pool':'FINAL_D4_KEEP_PROVISIONAL','original_d4_route':r['route']} for r in keeps]
    current={identity(r):r for r in candidates}
    for r in rows(D1/'D1_2_SFT_CANDIDATE.jsonl'):
        if identity(r) in current:require(r['messages']==current[identity(r)]['messages'],'D4/D1 messages mismatch')
        if r['source'] in EXTRA:
            candidates.append({**r,'audit_id':'P2-D1-'+rank(str(identity(r)),'source-identity')[:24],'origin_pool':'EXISTING_D1_2_ADDITIONAL_SOURCE','task_type':'INSTRUCTION' if r['source']=='smol-contraints' else 'CONVERSATION' if r['source']=='everyday-conversations' else 'TEXT_TRANSFORM'})
    require(len({identity(r) for r in candidates})==len(candidates),'duplicate source identity in combined pool')
    exclusions={};inherited=[]
    for r in rows(PREV/'selection/SELECTION_DECISIONS.jsonl'):
        if r['selection_decision'].startswith('DEFER'):
            original=by_aid[r['audit_id']]
            require(identity(original)==identity(r) and msgsha(original)==r['messages_sha256'],'prior deferral identity changed')
            exclusions[r['audit_id']]='inherited_'+r['selection_reason'];inherited.append(r)
    hold_paths=[ROOT/'runs/D4_completion_and_SFT_pilot_handoff/CONFIRMED_CONTENT_HOLDS.jsonl',PREV/'selection/ROOT_BOUNDED_REVIEW_DEFERRALS.jsonl']
    explicit=[]
    for p in hold_paths:
        for h in rows(p):
            r=by_aid[h['audit_id']];require(identity(r)==identity(h) and msgsha(r)==h['messages_sha256'],'confirmed hold identity differs')
            exclusions[h['audit_id']]='inherited_confirmed_hold_or_deferral';explicit.append(h)
    h=read(SFT99/'dataset/DERIVATIVE_MANIFEST.json')['deferral'];r=by_aid[h['audit_id']]
    require(identity(r)==identity(h) and msgsha(r)==h['messages_sha256'],'water-footprint deferral identity differs')
    exclusions[h['audit_id']]='inherited_water_footprint_deferral';explicit.append(h)
    benchmark=Path(prior['benchmark_filter']['d0_output_dir'])/'BENCHMARK_OVERLAP_HITS.jsonl'
    require(sha(benchmark)==prior['benchmark_filter']['d0_benchmark_hits_sha256'],'benchmark protection evidence differs')
    protected_ids=set();exempt_ids=set()
    for h in rows(benchmark):
        ident=(h['sft_source'],h['sft_row_index'])
        exemption=h['benchmark']=='IFEval' and h['sft_source']=='smol-contraints' and h['match_type']=='13gram_overlap'
        (exempt_ids if exemption else protected_ids).add(ident)
    protected=[];protected_paths=[]
    def add(path,flag):
        protected_paths.append(str(path))
        for r in rows(path):
            if 'messages' in r:protected.append({**r,'protection_flag':flag})
    add(D2/'D2_JUDGE_INPUT.jsonl','never_train_or_fresh_val');add(D2/'D2_CALIBRATION_CASES.jsonl','never_train_or_fresh_val')
    add(D4/'inputs/FIRST200_JUDGE_INPUT.jsonl','never_train_or_fresh_val')
    add(PREV/'evaluation/DEVELOPMENT_SUITE.jsonl','never_train_or_fresh_val')
    add(SFT99/'dataset/PILOT_V2_99STEP_VALIDATION.jsonl','never_train_or_fresh_val')
    add(SFT99/'dataset/PILOT_V2_99STEP_TRAIN.jsonl','not_fresh_val')
    for p in [PREV/'preview/PILOT_V2_FRESH200_REVIEW.jsonl',PREV/'preview/QUALITATIVE_REVIEW_12.jsonl']:
        if p.exists():add(p,'not_fresh_val')
    uf=old.UnionFind(len(candidates)+len(protected));owner={};links=Counter()
    combined=candidates+protected
    for i,r in enumerate(combined):
        for key in old.grouping_keys(r):
            key=hashlib.sha256(key.encode()).hexdigest()
            if key in owner:uf.union(i,owner[key]);links['normalized_context_or_conversation']+=1
            else:owner[key]=i
        if i and i%20000==0:print('grouped',i,flush=True)
    by_orig=defaultdict(list)
    for i,r in enumerate(combined):
        if 'original_row_index' in r:by_orig[r['original_row_index']].append(i)
    families=old.family_indices(prior)
    for fam in families:
        present=[i for ind in fam for i in by_orig[ind]]
        for i in present[1:]:uf.union(present[0],i);links['known_D1_family']+=1
    members=defaultdict(list)
    for i in range(len(combined)):members[uf.find(i)].append(i)
    group_flags=defaultdict(set)
    for group,indices in members.items():
        gid=rank('\n'.join(sorted(str(identity(combined[i]))+':'+msgsha(combined[i]) for i in indices)),'group-identity')
        split='validation' if int(rank(gid,'split')[:16],16)/2**64<.045 else 'train'
        for i in indices:
            r=combined[i]
            if r.get('protection_flag'):group_flags[group].add(r['protection_flag'])
            if r.get('audit_id') in exclusions or identity(r) in protected_ids:group_flags[group].add('inherited_hold_or_benchmark')
            if i<len(candidates):r['group_id']=gid;r['assigned_split']=split;r['_group']=group
    print('groups_complete',len(members),'candidate_rows',len(candidates),flush=True)
    suite_index=old.frozen_suite_index(rows(PREV/'evaluation/DEVELOPMENT_SUITE.jsonl'))
    from src.chat_template import load_chat_tokenizer
    import torch
    torch.set_num_threads(4)
    require(sha(old.TOKENIZER)==old.TOKENIZER_SHA,'tokenizer SHA differs')
    tok=load_chat_tokenizer(str(old.TOKENIZER))
    eligible=[];decisions=[];counts=Counter()
    for i,r in enumerate(candidates):
        flags=group_flags[r['_group']];reason=None
        if 'inherited_hold_or_benchmark' in flags:reason=exclusions.get(r['audit_id'],'inherited_hold_or_benchmark_prompt_group')
        elif 'never_train_or_fresh_val' in flags:reason='protected_evaluation_or_old_val_prompt_group'
        elif r['assigned_split']=='validation' and 'not_fresh_val' in flags:reason='old_training_or_inspected_prompt_group_not_fresh_val'
        reason=reason or old.suite_overlap(r,suite_index)
        task=None;checks=[]
        if reason is None:reason,task,checks=classify(r)
        if reason is None:
            r['p2_task']=task
            try:r.update(metadata(r,tok))
            except (ValueError,AssertionError) as exc:reason=str(exc)
        if reason is None:
            r['deterministic_checks']=checks;r.pop('_group',None);eligible.append(r)
        counts[reason or 'ELIGIBLE']+=1
        decisions.append({'audit_id':r['audit_id'],'source':r['source'],'original_row_index':r['original_row_index'],'group_id':r['group_id'],'assigned_split':r['assigned_split'],'decision':reason or 'ELIGIBLE','p2_task':task,'messages_sha256':msgsha(r)})
        if i and i%15000==0:print('screened',i,'eligible',len(eligible),flush=True)
    # One eligible representative per existing exact prompt/conversation group.
    representatives={}
    for r in eligible:
        k=r['group_id'];choice=(r['max_assistant_content_tokens']>128,rank(r['audit_id'],'group-representative'))
        if k not in representatives or choice<representatives[k][0]:representatives[k]=(choice,r)
    pool=[v[1] for v in representatives.values()]
    train=choose([r for r in pool if r['assigned_split']=='train'],12000)
    val=choose([r for r in pool if r['assigned_split']=='validation'],500)
    selected=train+val
    from sft.train_sft import build_example
    for r in selected:
        x,y,w=build_example(r,tok,2048,None,1.0,[],'none')
        require(int((y[1:]!=-100).sum())==r['shifted_supervised_tokens'],'real trainer shifted denominator differs')
        require(float(w)==1,'unexpected row weight')
    require(not ({r['group_id'] for r in train}&{r['group_id'] for r in val}),'train/val group leakage')
    preview_buckets=defaultdict(list)
    for r in selected:preview_buckets[(r['source'],r['p2_task'],'le128' if r['max_assistant_content_tokens']<=128 else '129to256')].append(r)
    for b in preview_buckets.values():b.sort(key=lambda r:rank(r['audit_id'],'preview60'))
    preview=[];strata=Counter()
    while len(preview)<min(60,len(selected)):
        for k in sorted(preview_buckets):
            if preview_buckets[k] and len(preview)<60:
                r=preview_buckets[k].pop(0);preview.append({**r,'preview_index':len(preview)+1});strata[' / '.join(k)]+=1
    for name,rs in [('DRAFT_TRAIN.jsonl',train),('DRAFT_VALIDATION.jsonl',val),('DRAFT_CANDIDATE.jsonl',selected),('PREVIEW60_FULL_MESSAGES.jsonl',preview),('ELIGIBLE_POOL.jsonl',pool),('SELECTION_DECISIONS.jsonl',decisions)]:save_rows(name,rs)
    save_rows('INHERITED_EXPLICIT_HOLDS.jsonl',explicit)
    sources=[D1/'D1_2_SFT_CANDIDATE.jsonl',D1/'D1_2_SUMMARY.json',D4/'MAIN_RESULTS.jsonl',D4/'RUN_STATUS.json',benchmark,OLD_SCRIPT,PREV/'selection/SELECTION_DECISIONS.jsonl',SFT99/'dataset/DERIVATIVE_MANIFEST.json']+[Path(p) for p in protected_paths]
    manifest={'status':'DRAFT_CONTENT_REVIEW_PENDING','CONTENT_REVIEW_PENDING':True,'TRAINING_STARTED':False,'seed':SEED,'rank_domain':DOMAIN,
      'input_rows':len(candidates),'completed_d4_keep_rows':36471,'additional_D1_2_rows':len(candidates)-36471,
      'input_identities':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sources},
      'inherited_deferrals_count':len(inherited),'explicit_hold_records':len(explicit),
      'benchmark_protection':{'policy':prior['benchmark_filter']['policy'],'nonexempt_source_identities':len(protected_ids),'inherited_ifeval_constraint_template_exemptions':len(exempt_ids),'limitations':'Existing lexical evidence/policy retained; template overlap exemption is not benchmark-clean certification; no new semantic benchmark detector.'},
      'grouping':{'groups_including_protections':len(members),'links':dict(links),'known_family_records':len(families),'family_limit':'Existing top-family summary has truncated memberships; no complete semantic family registry','representatives':len(pool),'eval_protected_rows':len(protected),'split_before_scope_quota':True,'cross_split_groups':0},
      'scope_counts':dict(counts),'eligible':summarize(pool),'train':summarize(train),'validation':summarize(val),'candidate':summarize(selected),
      'preview':{'rows':len(preview),'strata':dict(strata),'file':'PREVIEW60_FULL_MESSAGES.jsonl','sha256':sha(OUT/'PREVIEW60_FULL_MESSAGES.jsonl'),'full_original_messages_preserved':True,'quality_population_accuracy_estimate':False},
      'guard_status':{'sufficient_train_rows':8000<=len(train)<=16000,'sufficient_validation_rows':300<=len(val)<=800,
         'train_python_5_to_10_percent':.05<=summarize(train)['task_supervised_fraction']['PYTHON_BASIC']<=.10,
         'magpie_at_most30_percent':summarize(train)['source_supervised_fraction'].get('smol-magpie-ultra-short',0)<=.30,
         'assistant_at_least60percent_le128':summarize(train)['assistant_messages_le128_fraction']>=.60,
         'assistant_at_least90percent_le256':summarize(train)['assistant_messages_le256_fraction']>=.90},
      'files':{n:{'sha256':sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in ['DRAFT_TRAIN.jsonl','DRAFT_VALIDATION.jsonl','DRAFT_CANDIDATE.jsonl']},
      'formatter':{'path':str(ROOT/'src/chat_template.py'),'sha256':sha(ROOT/'src/chat_template.py'),'default_system':None,'truncate':False,'max_sequence_tokens':2048,'actual_build_example_labels_shift_verified':len(selected)},
      'builder_sha256':sha(__file__),'wall_seconds':time.perf_counter()-start,
      'limitations':['D4 KEEP_PROVISIONAL and D1.2 membership are eligibility, not correctness certification.','Finite content/constraint filters can miss errors or defer suitable rows; pending full-message preview is bounded and not an accuracy estimator.','Python uses inherited AST/scope checks only; no candidate program executed.','All original messages preserved, no response truncation or system removal.','Soft token anchors are approximate; no corpus shrinking to smallest task supply.']}
    save('DRAFT_MANIFEST.json',manifest)
    print(json.dumps({'status':manifest['status'],'train':manifest['train'],'validation':manifest['validation'],'guards':manifest['guard_status'],'wall_seconds':manifest['wall_seconds']}),flush=True)

if __name__=='__main__':main()
