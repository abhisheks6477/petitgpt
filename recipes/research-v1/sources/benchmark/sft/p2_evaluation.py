"""FP32 selected-target P2 validation with disjoint teacher-forced position NLL."""
import hashlib
import torch
from sft.p2_loss import assistant_token_nll_fp32

BUCKETS=('first_content','next_up_to_7_content','remaining_content','assistant_eos')

def answer_position_buckets(labels):
    """Return row-major bucket IDs for exactly labels[:,1:] != -100 targets."""
    shifted=labels[:,1:];mask=shifted!=-100
    cats=torch.full_like(shifted,-1)
    for row in range(shifted.shape[0]):
        offset=0;inside=False
        for pos,value in enumerate(shifted[row].tolist()):
            if value==-100:
                inside=False;offset=0;continue
            if value==3:
                if not inside:raise ValueError('EOS must follow nonempty assistant content')
                cats[row,pos]=3;inside=False;offset=0;continue
            if not inside:offset=0;inside=True
            cats[row,pos]=0 if offset==0 else (1 if offset<8 else 2)
            offset+=1
        if inside:raise ValueError('assistant reference must terminate in EOS')
    result=cats[mask]
    if not bool((result>=0).all()):raise ValueError('unbucketed supervised target')
    return result

@torch.no_grad()
def evaluate_position_nll(model,loader,*,device,autocast_dtype=None,expected_rows=None,should_stop=None):
    was_training=model.training;model.eval();total=0.;targets=0;row_ids=[];batches=0;rows=0
    stats={name:{'targets':0,'nll_sum':0.,'teacher_forced_top1_correct':0} for name in BUCKETS}
    try:
        for batch in loader:
            if should_stop is not None and should_stop():raise InterruptedError('P2 validation stop requested')
            x=batch['input_ids'].to(device,non_blocking=True);y=batch['labels'].to(device,non_blocking=True)
            if bool(((y[:,1:]!=-100).sum(1)==0).any()):raise ValueError('zero supervised targets in validation row')
            with torch.autocast(torch.device(device).type,dtype=autocast_dtype,enabled=autocast_dtype is not None):logits=model(x)
            nll=assistant_token_nll_fp32(logits,y)
            if not bool(torch.isfinite(nll).all()):raise FloatingPointError('nonfinite validation NLL')
            cat=answer_position_buckets(batch['labels']).to(device)
            mask=y[:,1:]!=-100;correct=logits[:,:-1][mask].argmax(-1)==y[:,1:][mask]
            value=float(nll.double().sum());total+=value;targets+=len(nll);rows+=x.shape[0];batches+=1
            row_ids.extend(batch['row_ids'])
            for i,name in enumerate(BUCKETS):
                take=cat==i;stats[name]['targets']+=int(take.sum())
                stats[name]['nll_sum']+=float(nll[take].double().sum())
                stats[name]['teacher_forced_top1_correct']+=int(correct[take].sum())
            del logits,nll,x,y
    finally:model.train(was_training)
    if not targets:raise ValueError('zero supervised validation targets')
    if expected_rows is not None and rows!=expected_rows:raise ValueError('incomplete validation row coverage')
    if len(set(row_ids))!=rows or len(row_ids)!=rows:raise ValueError('duplicate or missing validation row IDs')
    assert sum(s['targets'] for s in stats.values())==targets
    assert abs(sum(s['nll_sum'] for s in stats.values())-total)<1e-6
    for s in stats.values():
        s['token_nll']=s['nll_sum']/s['targets'] if s['targets'] else None
        s['teacher_forced_top1_accuracy']=s['teacher_forced_top1_correct']/s['targets'] if s['targets'] else None
    return {'val_loss':total/targets,'nll_sum':total,'supervised_targets':targets,'rows':rows,'batches':batches,
            'row_ids_sha256':hashlib.sha256('\n'.join(row_ids).encode()).hexdigest(),'position_buckets':stats,
            'ce_precision':'selected logits cast FP32 before CE inside disabled autocast; FP64 scalar aggregation',
            'interpretation':'Teacher-forced next reference token likelihood and top1; not free-generation accuracy'}
