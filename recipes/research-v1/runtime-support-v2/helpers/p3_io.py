"""Public supplied-input I/O and tokenizer annotation subset; no data acquisition."""
import hashlib
import json
from pathlib import Path
from src.chat_template import encode_chat

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n')

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
