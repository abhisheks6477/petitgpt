"""CPU-only bounded P3 preparation. No target model load/optimizer/network."""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,'.')
from src.chat_template import load_chat_tokenizer, encode_chat
from curriculum import SEED, WORDS, build_procedural, build_plan, canonical, digest, inventory, training_values
ROOT=Path('.');RUN=Path(__file__).resolve().parents[1];OUT=RUN/'preparation'
P2=ROOT/'runs/sft_p2_concise_instruction_20260906';H=ROOT/'runs/P2_RESULT_REVIEW_AND_LEARNABILITY_HANDOFF'
TOK=ROOT/'runs/g_production_2026-08-21/release/tokenizer.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def jsonl(p,rs):Path(p).write_text(''.join(canonical(r)+'\n' for r in rs))
def identity(r):return (r.get('source'),r.get('original_row_index'))
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def context(r):return norm('\n'.join(m['content'] for m in r.get('messages',[]) if m['role']!='assistant'))

def select_replay(tok):
    manifest=json.loads((P2/'data/FROZEN_MANIFEST.json').read_text())
    for name in ['P2_TRAIN.jsonl','P2_VALIDATION.jsonl','P2_FINAL_DEFERRALS.jsonl']:
        assert sha(P2/'data'/name)==manifest['files'][name]['sha256']
    train=rows(P2/'data/P2_TRAIN.jsonl')
    # Frozen P2 train is already subject to original benchmark/group exclusions.
    # Recheck identities and prompt groups against small local protected files;
    # do not re-screen D4 or run another corpus/benchmark acquisition.
    protected_paths=[P2/'data/P2_VALIDATION.jsonl',P2/'data/P2_FINAL_DEFERRALS.jsonl',
                     P2/'data/INHERITED_EXPLICIT_HOLDS.jsonl',H/'IN_DISTRIBUTION_PROBES_WITH_REFERENCES.jsonl']
    for p in manifest['input_identities']:
        if any(v in p for v in ['D2_JUDGE_INPUT','D2_CALIBRATION_CASES','FIRST200_JUDGE_INPUT','DEVELOPMENT_SUITE','PILOT_V2_99STEP_VALIDATION']):protected_paths.append(Path(p))
    protected=[]
    for p in protected_paths:protected+=rows(p)
    # tiny input files use no source IDs: context matching still excludes them.
    for p in H.glob('*SFT.jsonl'):protected+=rows(p);protected_paths.append(p)
    for p in H.glob('*WITH_REFERENCES.jsonl'):
        if p not in protected_paths:protected+=rows(p);protected_paths.append(p)
    bad_ids={identity(r) for r in protected if r.get('source') is not None and r.get('original_row_index') is not None}
    bad_aids={r.get('audit_id',r.get('original_audit_id')) for r in protected}
    bad_groups={r.get('group_id') for r in protected if r.get('group_id')}
    bad_contexts={context(r) for r in protected if r.get('messages')}
    bad_messages={digest(r['messages']) for r in protected if r.get('messages')}
    local_deferrals=rows(OUT/'REVIEW_DEFERRALS.jsonl') if (OUT/'REVIEW_DEFERRALS.jsonl').exists() else []
    bad_aids.update(r['audit_id'] for r in local_deferrals)
    pools=collections.defaultdict(list);rejected=collections.Counter()
    for r in train:
        if identity(r) in bad_ids or r['audit_id'] in bad_aids or r['group_id'] in bad_groups or context(r) in bad_contexts or digest(r['messages']) in bad_messages:
            rejected['identity_or_known_hold_or_protected_context']+=1;continue
        if r['p2_task'] not in ['GENERAL_QA','TEXT_TRANSFORM','PRACTICAL_CHAT','PYTHON_BASIC']:
            rejected['outside_replay_task_selection']+=1;continue
        ids,labels=encode_chat(tok,r['messages'],default_system=None)
        if len(ids)>512:rejected['over512']+=1;continue
        # Concise existing replay, no changed messages or teacher judgments.
        if sum(v!=-100 for v in labels[1:])>220:rejected['over220_targets']+=1;continue
        r={**r,'p3_formatted_tokens':len(ids),'p3_shifted_targets':sum(v!=-100 for v in labels[1:])}
        pools[r['p2_task']].append(r)
    desired={'GENERAL_QA':1408,'TEXT_TRANSFORM':1120,'PRACTICAL_CHAT':288,'PYTHON_BASIC':256}
    selected=[]
    for task,n in desired.items():
        pool=sorted(pools[task],key=lambda r:digest([SEED,'replay',r['audit_id']]))
        selected+=pool[:n]
    if len(selected)<3072:
        used={r['audit_id'] for r in selected}
        rest=sorted([r for p in pools.values() for r in p if r['audit_id'] not in used],key=lambda r:digest([SEED,'fallback',r['audit_id']]))
        selected+=rest[:3072-len(selected)]
    assert len(selected)==3072 and any(r['p2_task']=='PYTHON_BASIC' for r in selected)
    selected.sort(key=lambda r:digest([SEED,'final-replay-order',r['audit_id']]))
    audit={'source_train_sha256':sha(P2/'data/P2_TRAIN.jsonl'),'original_frozen_manifest_sha256':sha(P2/'data/FROZEN_MANIFEST.json'),
           'original_benchmark_policy':manifest['benchmark_protection'],'inherited_p2_group_exclusions':manifest['grouping'],
           'protected_files':{str(p):sha(p) for p in protected_paths},'excluded':dict(rejected),
           'available_by_task':{k:len(v) for k,v in pools.items()},'task_rows':dict(collections.Counter(r['p2_task'] for r in selected)),
           'source_rows':dict(collections.Counter(r['source'] for r in selected)),
           'all_original_messages_preserved':True,'new_teacher_judgments':0,'python_label_is_not_token_percentage':True}
    return selected,audit

def annotate(tok,r):
    ids,labels=encode_chat(tok,r['messages'],default_system=None)
    assert len(ids)<=512 and any(v!=-100 for v in labels[1:])
    # Independent role-by-role target reconstruction including EOS boundaries.
    expected=[]
    for m in r['messages']:
        if m['role']=='assistant':expected+=tok.encode(m['content']).ids+[3]
    assert [v for v in labels[1:] if v!=-100]==expected
    assert all(label==-100 or label==tid for label,tid in zip(labels,ids))
    r['formatted_tokens']=len(ids);r['shifted_targets']=len(expected)
    r['input_unsupervised_tokens']=len(ids)-len(expected)
    return r

def main():
    assert not (OUT/'FROZEN.json').exists(),'immutable freeze exists'
    tok=load_chat_tokenizer(str(TOK));assert sha(TOK)=='d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce'
    replay,audit=select_replay(tok)
    full_replay='\n'.join(m['content'] for r in replay for m in r['messages'])
    pools={'train':training_values()}
    # Unseen values are lexical identifiers and natural phrases. Exact case-folded
    # substring absence is stricter than whole-entry absence; no pretraining claim.
    candidates=[]
    for n,w in enumerate(WORDS):
        candidates += [f'{w.title()}{31+n}',f'{w} {WORDS[(n+17)%len(WORDS)]}',f'{w.title()}{WORDS[(n+7)%len(WORDS)].title()}']
    candidates=[v for v in candidates if v.casefold() not in full_replay.casefold() and v not in pools['train']]
    candidates=sorted(set(candidates),key=lambda v:digest([SEED,'unseen-pool',v]))
    assert len(candidates)>=128
    pools['development']=candidates[:64];pools['final']=candidates[64:128]
    signatures={v:[len(tok.encode(prefix+v).ids) for prefix in ['', ' ', '- ', '[', ': ', ', ']] for pool in pools.values() for v in pool}
    write(OUT/'VALUE_BOUNDARY_SIGNATURES.json',signatures)
    sets=build_procedural(pools,signatures)
    for split,rs in sets.items():
        for r in rs:annotate(tok,r)
        jsonl(OUT/(split+'.jsonl'),rs)
    for r in replay:annotate(tok,r)
    jsonl(OUT/'replay.jsonl',replay);write(OUT/'REPLAY_SELECTION.json',audit)
    write(OUT/'TEMPLATE_INVENTORY.json',inventory());write(OUT/'VALUE_POOLS.json',pools)
    value_stats=[]
    for split,pool in pools.items():
        for v in pool:
            value_stats.append({'split':split,'value':v,'tokens_standalone':tok.encode(v).ids,
                                'tokens_leading_space':tok.encode(' '+v).ids,'replay_casefold_substring_occurrences':full_replay.casefold().count(v.casefold())})
    jsonl(OUT/'VALUE_TOKEN_LENGTHS.jsonl',value_stats)
    plan=build_plan(sets['train'],replay);jsonl(OUT/'UPDATE_PLAN.jsonl',plan)
    preview=[]
    for family in ['COPY','FIELD','MEMBERSHIP','JSON']:
        rs=[r for r in sets['train'] if r['family']==family]
        preview += [rs[i] for i in [0,1,112,113,448,449,896,897]]
    # 16 source/task/length-stratified fresh records: 4 per actual task.
    for task in ['GENERAL_QA','TEXT_TRANSFORM','PRACTICAL_CHAT','PYTHON_BASIC']:
        rs=sorted([r for r in replay if r['p2_task']==task],key=lambda r:(r['source'],r['formatted_tokens']))
        if (OUT/'PREVIEW48_INITIAL.jsonl').exists():
            original=[r for r in rows(OUT/'PREVIEW48_INITIAL.jsonl') if r.get('p2_task')==task]
            available={r['audit_id']:r for r in rs};chosen=[]
            for prior in original:
                used={r['audit_id'] for r in chosen}
                replacement=available.get(prior['audit_id'])
                if replacement is None:
                    replacement=min((r for r in rs if r['audit_id'] not in used and r['audit_id'] not in {a['audit_id'] for a in original}),key=lambda r:(r['source']!=prior['source'],abs(r['formatted_tokens']-prior['formatted_tokens']),r['audit_id']))
                chosen.append(replacement)
            preview+=chosen
        else:
            preview += [rs[int((i+.5)*len(rs)/4)] for i in range(4)]
    jsonl(OUT/'PREVIEW48.jsonl',preview)
    text=[]
    for i,r in enumerate(preview,1):
        text.append(f'## {i}. {r["audit_id"]} / {r.get("family",r.get("p2_task"))} / {r.get("source","procedural")} / tokens={r["formatted_tokens"]}\n')
        for m in r['messages']:text.append(m['role'].upper()+':\n'+m['content']+'\n')
    (OUT/'PREVIEW48.md').write_text('\n'.join(text))
    (OUT/'ALL_TEMPLATES.md').write_text('\n\n'.join(f'### {r["template_id"]} ({r["split"]})\n{r["text"]}' for r in inventory()))
    sample=[]
    for family in ['COPY','FIELD','MEMBERSHIP','JSON']:
        rs=[r for r in sets['train'] if r['family']==family]
        for t in range(16):sample+=rs[t*112:t*112+2]
    jsonl(OUT/'TRAIN_SAMPLE128.jsonl',sample)
    jsonl(OUT/'COUNTERFACTUAL_PAIRS.jsonl',[r for r in preview if r.get('family') in ('FIELD','MEMBERSHIP')])
    stats={}
    for split,rs in {**sets,'replay':replay}.items():
        lens=sorted(r['formatted_tokens'] for r in rs)
        stats[split]={'rows':len(rs),'formatted_tokens':sum(lens),'shifted_targets':sum(r['shifted_targets'] for r in rs),
                      'length_min':lens[0],'length_median':lens[len(lens)//2],'length_max':lens[-1],
                      'all_fit512':True,'all_assistant_content_and_eos_only':True}
    write(OUT/'DATA_STATS.json',stats)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
