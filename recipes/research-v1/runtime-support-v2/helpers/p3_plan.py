"""Public training plan subset. No procedural generation or held-out templates."""
import hashlib
import json
import random

SEED = 20260907

FAMILIES = ('COPY', 'FIELD', 'MEMBERSHIP', 'JSON')

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def build_plan(proc,replay):
    rng=random.Random(SEED);plan=[]
    for epoch in range(2):
        streams={f:[r['audit_id'] for r in proc if r['family']==f] for f in FAMILIES}
        streams['replay']=[r['audit_id'] for r in replay]
        for rows in streams.values():rng.shuffle(rows)
        offsets={k:0 for k in streams}
        for block in range(32):
            kinds=['procedural']*7+['replay']*3;rng.shuffle(kinds)
            for kind in kinds:
                ids=[]
                for key in FAMILIES if kind=='procedural' else ['replay']:
                    n=8 if kind=='procedural' else 32
                    ids+=streams[key][offsets[key]:offsets[key]+n];offsets[key]+=n
                rng.shuffle(ids)
                plan.append({'update':len(plan)+1,'epoch':epoch+1,'block':epoch*32+block+1,'stream':kind,'row_ids':ids})
    return plan
