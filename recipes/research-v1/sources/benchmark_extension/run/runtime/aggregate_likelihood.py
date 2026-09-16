"""Independent likelihood audit: no imports from execution/scoring driver."""
from pathlib import Path
import json,hashlib,math,csv
R=Path(__file__).resolve().parents[1];S=R.parent/'source'
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 models=json.loads((R/'protocol/MODEL_IDENTITIES.json').read_text());data=json.loads((R/'protocol/DATA_IDENTITIES.json').read_text());all_results={};audit=[]
 for task,expected in [('arc_challenge',1172),('hellaswag',10042)]:
  official=rows(R/task/'normalized_rows.jsonl');assert len(official)==expected and sha(R/task/'normalized_rows.jsonl')==data[task]['normalized_rows_sha256'];all_results[task]={}
  same_ids=[]
  for m in models:
   mid=m['id'];p=R/task/f'rows_{mid}.jsonl';rr=rows(p);assert len(rr)==expected;assert len(set(r['id'] for r in rr))==expected
   correct=0;norm_correct=0;candidates=0
   for i,(r,d) in enumerate(zip(rr,official)):
    assert r['row_index']==i and all(r[k]==d[k] for k in d)
    assert r['row_sha256']==hashlib.sha256(canon(d).encode()).hexdigest()
    cs=r['choices'];sc=r['scores'];den=r['candidate_chars'];assert len(sc)==len(cs)==len(den) and all(math.isfinite(s) for s in sc)
    assert den==[len(c) for c in cs] and all(n>0 for n in den)
    # Sort by score descending and index ascending; explicit first tie semantics.
    pred=sorted(range(len(sc)),key=lambda j:(-sc[j],j))[0];norm=sorted(range(len(sc)),key=lambda j:(-sc[j]/den[j],j))[0]
    assert r['prediction_acc']==pred and r['prediction_acc_norm']==norm and r['acc']==int(pred==d['gold']) and r['acc_norm']==int(norm==d['gold'])
    correct+=pred==d['gold'];norm_correct+=norm==d['gold'];candidates+=len(cs)
   same_ids.append([r['id'] for r in rr]);stored=json.loads((R/task/f'RESULTS_{mid}.json').read_text())
   v={'model':mid,'task':task,'rows':expected,'acc_correct':correct,'acc_norm_correct':norm_correct,'acc':correct/expected,'acc_norm':norm_correct/expected,'candidate_sequences':candidates,'rows_file':str(p.relative_to(R)),'rows_sha256':sha(p)}
   assert all(stored[k]==v[k] for k in v);all_results[task][mid]=v
   audit.append({'model':mid,'task':task,'status':'PASS','rows':expected,'finite_scores':True,'unique_complete_rows':True,'official_order_and_choices_equal':True,'first_argmax_recomputed':True,'aggregates_equal':True,'rows_sha256':sha(p)})
  assert all(x==same_ids[0] for x in same_ids)
  save(R/task/'RESULTS.json',{'status':'PASS','data_identity':data[task],'results':list(all_results[task].values())})
 save(R/'protocol/INDEPENDENT_LIKELIHOOD_AUDIT.json',{'status':'PASS','implementation':'independent stdlib audit, sorting tie resolution, no scorer imports','checks':audit})
 def pc(x):return f'{x*100:.2f}%'
 text='| Model | ARC-C acc | ARC-C acc_norm | HellaSwag acc | HellaSwag acc_norm |\n|---|---:|---:|---:|---:|\n'
 for m in models:
  mid=m['id'];a=all_results['arc_challenge'][mid];h=all_results['hellaswag'][mid];text+=f"| {mid} | {pc(a['acc'])} | {pc(a['acc_norm'])} | {pc(h['acc'])} | {pc(h['acc_norm'])} |\n"
 text+='\nARC-Challenge: 1,172 rows/model. HellaSwag: 10,042 rows/model.\n\nacc_norm uses Python len(candidate text before the leading delimiter), with HellaSwag choices preprocessed by the pinned task. It is the PetitGPT project protocol, not a claim of leaderboard protocol equivalence.\n'
 (R/'tables/LIKELIHOOD_NEW_TASKS.md').write_text(text)
 p=S/'docs/petitgpt-v1/tables/PUBLIC_BENCHMARK_RESULTS.csv';old=list(csv.DictReader(p.open()));frozen={(d['model'],d['task']):d for d in old}
 proposal='已有 ARC-Easy/PIQA 直接引用公开 CSV，没有重新运行或重新计算。每格为 acc / acc_norm；不计算平均分。\n\n| Model | ARC-Easy | ARC-Challenge | PIQA | HellaSwag |\n|---|---:|---:|---:|---:|\n'
 for m in models:
  mid=m['id'];ae=frozen[(mid,'arc_easy')];pi=frozen[(mid,'piqa')];a=all_results['arc_challenge'][mid];h=all_results['hellaswag'][mid]
  proposal+=f"| {mid} | {ae['acc_pct']}% / {ae['acc_norm_pct']}% | {pc(a['acc'])} / {pc(a['acc_norm'])} | {pi['acc_pct']}% / {pi['acc_norm_pct']}% | {pc(h['acc'])} / {pc(h['acc_norm'])} |\n"
 (R/'tables/FOUR_TASK_PROPOSAL.md').write_text(proposal)
 save(R/'protocol/PUBLISHED_RESULTS_REFERENCE.json',{'file':str(p),'sha256':sha(p),'rows_copied_without_recomputation':old})
 print('INDEPENDENT LIKELIHOOD AUDIT PASS')
if __name__=='__main__':main()
