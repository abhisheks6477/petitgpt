"""Finite, materialized P2 row stream over at most two existing epoch shuffles."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from torch.utils.data import Sampler
from src.posttrain_resume import DeterministicEpochBatchSampler

SCHEMA='p2_two_pass_materialized_v1'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def derive_plan(rows, *, train_path, seed=20260906, micro_bsz=2, grad_accum=16):
    n=len(rows);effective=micro_bsz*grad_accum
    if len({str(r['audit_id']) for r in rows})!=n:raise ValueError('duplicate training row IDs')
    if n<1 or effective<1:raise ValueError('nonempty rows and positive batch geometry required')
    sampler=DeterministicEpochBatchSampler(n,micro_bsz,seed=seed,drop_last=False)
    epochs=[[i for batch in sampler for i in batch] for _ in range(2)]
    assert all(sorted(order)==list(range(n)) for order in epochs)
    stream=epochs[0]+epochs[1]
    updates=min(1000,len(stream)//effective)
    consumed=stream[:updates*effective]
    ids=[str(rows[i]['audit_id']) for i in consumed]
    counts=Counter(consumed)
    assert max(counts.values())<=2
    first_boundary=min(updates,(n+effective-1)//effective)
    per_update=[]
    for step in range(updates):
        indices=consumed[step*effective:(step+1)*effective]
        row_ids=[str(rows[i]['audit_id']) for i in indices]
        per_update.append({'step':step+1,'row_ids':row_ids,'row_indices':indices,
            'row_ids_sha256':hashlib.sha256('\n'.join(row_ids).encode()).hexdigest(),
            'supervised_targets':sum(int(rows[i]['shifted_supervised_tokens']) for i in indices)})
    return {'schema':SCHEMA,'train_path':str(Path(train_path).resolve()),'train_sha256':file_sha(train_path),
        'dataset_rows':n,'seed':seed,'micro_bsz':micro_bsz,'grad_accum':grad_accum,
        'effective_conversations_per_update':effective,'max_passes':2,'max_updates_cap':1000,
        'epoch_seed_rule':'(seed + epoch * 0x4F1BBCDCBFA54001) & 0x7fffffffffffffff',
        'epochs':epochs,'max_steps':updates,'warmup_steps':max(1,round(updates*.05)),
        'evaluation_steps':sorted({first_boundary,updates}),'checkpoint_steps':sorted({first_boundary,updates}),
        'consumed_row_indices':consumed,'consumed_row_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'consumed_rows':len(consumed),'unique_rows_consumed':len(counts),
        'exposure_histogram':{str(k):sum(counts.get(i,0)==k for i in range(n)) for k in [0,1,2]},
        'unused_tail_indices':stream[len(consumed):],'unused_tail_rows':len(stream)-len(consumed),
        'supervised_targets':sum(u['supervised_targets'] for u in per_update),'updates':per_update,
        'boundary_note':'Complete epoch permutations are concatenated before grouping. A 32-row update may straddle epoch 0/1; no update extends beyond the two-pass stream. Remaining terminal tail is unused.'}

def load_checked_plan(path, *, train_path, dataset_rows, seed, micro_bsz, grad_accum, max_steps):
    plan=json.loads(Path(path).read_text())
    required={'schema':SCHEMA,'dataset_rows':dataset_rows,'seed':seed,'micro_bsz':micro_bsz,
              'grad_accum':grad_accum,'max_steps':max_steps,'max_passes':2}
    if any(plan.get(k)!=v for k,v in required.items()):raise ValueError('P2 plan geometry differs from trainer')
    if file_sha(train_path)!=plan['train_sha256']:raise ValueError('P2 train file changed after plan freeze')
    rows=[json.loads(line) for line in Path(train_path).read_text().splitlines()]
    expected=derive_plan(rows,train_path=train_path,seed=seed,micro_bsz=micro_bsz,grad_accum=grad_accum)
    if plan!=expected:raise ValueError('P2 materialized plan differs from deterministic two-pass derivation')
    if not 1<=max_steps<=1000:raise ValueError('P2 update cap exceeded')
    return plan

class MaterializedP2BatchSampler(Sampler):
    def __init__(self,plan):
        self.indices=plan['consumed_row_indices'];self.batch_size=plan['micro_bsz']
    def __iter__(self):
        for start in range(0,len(self.indices),self.batch_size):
            yield self.indices[start:start+self.batch_size]
    def __len__(self):return len(self.indices)//self.batch_size
