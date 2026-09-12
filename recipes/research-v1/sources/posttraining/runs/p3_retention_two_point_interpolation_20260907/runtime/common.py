"""Paths and compact I/O for the two fixed inference candidates."""
import hashlib
import json
from pathlib import Path
ROOT=Path('.')
RUN=Path(__file__).resolve().parents[1]
P2=ROOT/'runs/sft_p2_concise_instruction_20260906'
P3=ROOT/'runs/p3_basic_instruction_generalization_20260907'
RET=ROOT/'runs/p3_step320_retention_comparison_20260907'
TASK=ROOT/'runs/p3_retention_review_and_two_point_interpolation/CODEX_TWO_POINT_WEIGHT_INTERPOLATION.md'
TOK=ROOT/'runs/g_production_2026-08-21/release/tokenizer.json'
TOK_SHA='d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce'
PARENTS=[(P2/'train/step_000750.pt','20afc40096568343c59d8a55e9606667c279f5f1c976908c57de18a0107f4e4e'),
         (P3/'train/step_000320.pt','b11cb018d9f7382e46088f587dc62db9bc25c7d696d399e6b8a001f7ca9cb4e2')]
CANDIDATES={'alpha050':0.50,'alpha075':0.75}
FLAGS={'TASK':'P3_RETENTION_TWO_POINT_INTERPOLATION_V1','NEW_ALPHA_VALUES':[.50,.75],
       'PARENT_WEIGHTS_MODIFIED':False,'OPTIMIZER_CONSTRUCTED':False,'OPTIMIZER_UPDATES':0,
       'TRAINING_STARTED':False,'PAID_API_REQUESTS':0,'P3_FINAL_BATTERY_RERUN':False,
       'EXTRA_ALPHA_SEARCH':False,'RELEASE_MODEL_SELECTED':False,
       'NEXT_ACTION':'RETURN_FOR_RETENTION_TRADEOFF_REVIEW'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def jsonl(p,xs):Path(p).write_text(''.join(canonical(x)+'\n' for x in xs))
