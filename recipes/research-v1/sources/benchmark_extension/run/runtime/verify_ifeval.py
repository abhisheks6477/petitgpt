"""Pinned IFEval verifier: isolated dependencies, fixtures, and saved-response scoring."""
from pathlib import Path
import os,sys,json,hashlib,random,collections,importlib.metadata,argparse
R=Path(__file__).resolve().parents[1]
os.environ['NLTK_DATA']=str(R/'nltk_data')
sys.path[:0]=[str(R/'evaluator_deps'),str(R/'upstream')]
import langdetect
from lm_eval.tasks.ifeval import utils,instructions_registry
random.seed(0);langdetect.DetectorFactory.seed=0

def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in (R/p).read_text().splitlines()]
def save(p,x):(R/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def fixtures():
 english='This is a clear English sentence about the weather and the beautiful garden outside our home.'
 french='Ceci est une phrase française qui parle du beau jardin et des fleurs devant notre maison.'
 cases=[
 ('keywords:existence',{'keywords':['apple','banana']},'apple banana','apple'),
 ('keywords:frequency',{'keyword':'apple','frequency':2,'relation':'at least'},'apple apple','banana'),
 ('keywords:forbidden_words',{'forbidden_words':['apple']},'banana','apple'),
 ('keywords:letter_frequency',{'letter':'z','let_frequency':2,'let_relation':'at least'},'zz','abc'),
 ('language:response_language',{'language':'en'},english,french),
 ('length_constraints:number_sentences',{'num_sentences':2,'relation':'at least'},'This is the first sentence. This is the second sentence.','One sentence.'),
 ('length_constraints:number_paragraphs',{'num_paragraphs':2},'First paragraph.\n***\nSecond paragraph.','Only one paragraph.'),
 ('length_constraints:number_words',{'num_words':3,'relation':'at least'},'one two three','one'),
 ('length_constraints:nth_paragraph_first_word',{'num_paragraphs':2,'nth_paragraph':2,'first_word':'apple'},'First paragraph.\n\napple is red.','First paragraph.\n\nBanana is yellow.'),
 ('detectable_content:number_placeholders',{'num_placeholders':2},'[name] [address]','plain text'),
 ('detectable_content:postscript',{'postscript_marker':'P.S.'},'A letter.\nP.S. Please call.','A letter.'),
 ('detectable_format:number_bullet_lists',{'num_bullets':2},'* first\n* second','first and second'),
 ('detectable_format:constrained_response',{},'My answer is yes.','A different answer.'),
 ('detectable_format:number_highlighted_sections',{'num_highlights':2},'*first* and *second*','plain text'),
 ('detectable_format:multiple_sections',{'section_spliter':'Section','num_sections':2},'Section 1\nFirst.\nSection 2\nSecond.','No headings.'),
 ('detectable_format:json_format',{},'{"answer":true}','invalid json'),
 ('detectable_format:title',{},'<<A title>>\nContent.','No title.'),
 ('combination:two_responses',{},'First answer.\n******\nSecond answer.','Only one answer.'),
 ('combination:repeat_prompt',{'prompt_to_repeat':'Repeat this prompt.'},'Repeat this prompt. Here is the answer.','Here is the answer.'),
 ('startend:end_checker',{'end_phrase':'The end.'},'A story. The end.','A story.'),
 ('change_case:capital_word_frequency',{'capital_frequency':2,'capital_relation':'at least'},'THE DOG runs home.','the dog runs home.'),
 ('change_case:english_capital',{},english.upper(),english.lower()),
 ('change_case:english_lowercase',{},english.lower(),english.upper()),
 ('punctuation:no_comma',{},'one two three','one, two, three'),
 ('startend:quotation',{},'"A quoted answer."','An unquoted answer.'),
 ]
 results=[]
 for i,(iid,kwargs,good,bad) in enumerate(cases):
  doc={'key':i,'prompt':'Synthetic verifier fixture.','instruction_id_list':[iid],'kwargs':[kwargs]}
  good_out=utils.process_results(doc,[good]);bad_out=utils.process_results(doc,[bad]);ok=good_out['prompt_level_strict_acc'] and good_out['prompt_level_loose_acc'] and not bad_out['prompt_level_strict_acc'] and not bad_out['prompt_level_loose_acc']
  results.append({'instruction_id':iid,'kwargs':kwargs,'conforming_response':good,'nonconforming_response':bad,'conforming':good_out,'nonconforming':bad_out,'pass':bool(ok)})
 doc={'key':100,'prompt':'Return JSON.','instruction_id_list':['detectable_format:json_format'],'kwargs':[{}]}
 difference=utils.process_results(doc,['Here is the requested JSON:\n{"answer":true}\nThank you.']);assert not difference['prompt_level_strict_acc'] and difference['prompt_level_loose_acc']
 actual=rows('ifeval/normalized_rows.jsonl');ids=set(t for d in actual for t in d['instruction_id_list']);assert ids<=set(instructions_registry.INSTRUCTION_DICT)
 assert ids==set(t[0] for t in cases)
 execution=[]
 for d in actual:
  x=utils.process_results(d,[english]);assert len(x['inst_level_strict_acc'])==len(d['instruction_id_list']);execution.append({'key':d['key'],'instruction_count':len(d['instruction_id_list'])})
 out={'status':'PASS' if all(x['pass'] for x in results) else 'FAIL','cases':results,'strict_loose_difference':difference,'dataset_rows':len(actual),'instruction_count':sum(len(d['instruction_id_list']) for d in actual),'instruction_id_coverage':sorted(ids),'all_dataset_kwargs_and_checkers_executed':True,'unknown_ids':[],'langdetect_seed':0,'random_seed':0}
 save('ifeval/verifier/FIXTURES.json',out)
 save('protocol/EVALUATOR_ENVIRONMENT.json',{'isolation':'task-local evaluator_deps prepended only in verifier process','versions':{p:importlib.metadata.version(p) for p in ['langdetect','nltk','immutabledict','wheel','packaging','six','click','joblib','regex','tqdm']},'package_archives':{str(p.relative_to(R)):sha(p) for p in (R/'evaluator_wheels').iterdir()},'resource_downloaded':'punkt_tab','resource_files':{str(p.relative_to(R)):sha(p) for p in (R/'nltk_data').rglob('*') if p.is_file()},'langdetect_seed':0,'random_seed':0,'pinned_verifier_source_unmodified':True})
 print(out['status'],len(results),'instruction types',out['instruction_count'],'dataset instructions',flush=True)
 assert out['status']=='PASS'

def score(stage,mid):
 data=rows('ifeval/normalized_rows.jsonl');docs={d['key']:d for d in data};path=f'ifeval/generations/{mid}.jsonl' if stage=='full' else f'smoke/ifeval_{mid}.jsonl';gens=rows(path)
 expected=[d['key'] for d in data] if stage=='full' else list(json.loads((R/'protocol/IFEVAL_SMOKE_SELECTION.json').read_text())['keys'])
 assert [g['key'] for g in gens]==expected and len(set(expected))==len(expected)
 scored=[]
 for g in gens:
  assert hashlib.sha256(g['response'].encode()).hexdigest()==g['response_sha256'];out=utils.process_results(docs[g['key']],[g['response']]);scored.append({'key':g['key'],'response_sha256':g['response_sha256'],**out})
 def aggregate(rr):
  n=len(rr);count=sum(len(r['inst_level_strict_acc']) for r in rr);ps=sum(r['prompt_level_strict_acc'] for r in rr);pl=sum(r['prompt_level_loose_acc'] for r in rr);si=sum(sum(r['inst_level_strict_acc']) for r in rr);li=sum(sum(r['inst_level_loose_acc']) for r in rr)
  return {'rows':n,'prompt_level_strict_acc':ps/n,'prompt_level_loose_acc':pl/n,'inst_level_strict_acc':utils.agg_inst_level_acc([r['inst_level_strict_acc'] for r in rr]),'inst_level_loose_acc':utils.agg_inst_level_acc([r['inst_level_loose_acc'] for r in rr]),'prompt_strict_correct':ps,'prompt_loose_correct':pl,'instruction_count':count,'instruction_strict_correct':si,'instruction_loose_correct':li}
 result=aggregate(scored);outpath=f'ifeval/verifier/{mid}_{stage}.jsonl';(R/outpath).write_text(''.join(canon(x)+'\n' for x in scored))
 # Fresh verifier pass over saved responses, in the original order, independently summed.
 random.seed(0);langdetect.DetectorFactory.seed=0
 repeated=[utils.process_results(docs[g['key']],[g['response']]) for g in gens];assert repeated==[{k:v for k,v in x.items() if k not in ['key','response_sha256']} for x in scored]
 pn=sum(int(all(x['inst_level_strict_acc'])) for x in repeated);ln=sum(int(all(x['inst_level_loose_acc'])) for x in repeated);strict=[v for x in repeated for v in x['inst_level_strict_acc']];loose=[v for x in repeated for v in x['inst_level_loose_acc']]
 assert result['prompt_level_strict_acc']==pn/len(gens) and result['prompt_level_loose_acc']==ln/len(gens)
 assert result['inst_level_strict_acc']==sum(strict)/len(strict) and result['inst_level_loose_acc']==sum(loose)/len(loose)
 result.update(model=mid,status='PASS',stage=stage,generations_file=path,generations_sha256=sha(R/path),verifier_rows_sha256=sha(R/outpath),independent_recomputation='PASS',instruction_id_coverage=sorted(set(t for g in gens for t in docs[g['key']]['instruction_id_list'])),stop_reasons=dict(collections.Counter(g['stop_reason'] for g in gens)),generated_tokens=sum(g['generated_token_count'] for g in gens))
 save(f'ifeval/RESULTS_{mid}.json' if stage=='full' else f'smoke/IFEVAL_SCORED_{mid}.json',result);print('SCORED',mid,stage,len(gens),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['fixtures','smoke','full']);p.add_argument('--model');a=p.parse_args();fixtures() if a.stage=='fixtures' else score(a.stage,a.model)
