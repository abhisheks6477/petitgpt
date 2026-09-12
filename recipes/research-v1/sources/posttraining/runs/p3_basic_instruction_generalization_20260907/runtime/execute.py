"""P3: separate baseline, one guarded bounded trainer, and post-run inference.
No production helper is modified. Every evaluation output is write-once.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,'.')
import torch
from torch.utils.data import DataLoader
from src.model import GPT,gpt_config_from_checkpoint_dict
from src.chat_template import load_chat_tokenizer,encode_prompt,encode_chat,encode_completion
from src.canonical_schedule import lr_schedule
from sft.train_sft import load_ckpt,effective_batch_backward,checked_optimizer_step,collate_fn_builder,save_checkpoint_atomic
from sft.p2_loss import assistant_nll_sum_fp32
from sft.p2_evaluation import evaluate_position_nll
from checkers import check
from prepare import RUN,OUT as DATA,ROOT,P2,H,TOK,sha,rows,write,canonical

CKPT=P2/'train/step_000750.pt'
EXPECTED='20afc40096568343c59d8a55e9606667c279f5f1c976908c57de18a0107f4e4e'
SEED=20260907
EVAL=RUN/'evaluation'
TRAIN=RUN/'train'
CONFIG={'source_checkpoint':str(CKPT),'source_sha256':EXPECTED,'optimizer':'fresh AdamW','model_weights_only':True,
        'lr_peak':5e-5,'warmup_updates':32,'planned_updates':640,'lr_schedule':'canonical cosine','min_lr_ratio':0.1,
        'betas':[0.9,0.95],'eps':1e-8,'weight_decay':0.0,'grad_clip':1.0,
        'microbatch':2,'grad_accum':16,'records_per_update':32,'seq_len':512,'architecture_context':2048,
        'seed':SEED,'passes':2,'training_process_wall_cap_seconds':7200,'no_automatic_restart':True,
        'precision':'BF16 forward; accepted selected-logits FP32 assistant CE','loss':'effective-update target-token mean including EOS',
        'procedural_updates_per_block':7,'replay_updates_per_block':3,'generation_cap':64,
        'greedy':True,'temperature':0,'top_k':0,'top_p':1,'eos_id':3,
        'formatter':'preserve_messages/default_system=None/full_context','cpu_threads':4,'loader_workers':0,
        'no_checkpoint_selection':True}

def verify_freeze():
    freeze=json.loads((DATA/'FROZEN.json').read_text())
    for p,h in freeze['files'].items():assert sha(RUN/p)==h,'Frozen file changed: '+p
    assert sha(CKPT)==EXPECTED
    assert sha(TOK)=='d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce'
    for p,h in freeze['accepted_source_files'].items():assert sha(ROOT/p)==h
    assert CONFIG==json.loads((DATA/'RUN_CONFIG.json').read_text())
    return freeze

def load_model(path,tag):
    start=time.monotonic();ck=load_ckpt(str(path));cfg=ck.get('config') or ck.get('cfg')
    model=GPT(gpt_config_from_checkpoint_dict(cfg));sd=ck['model']
    prefix=any(k.startswith('_orig_mod.') for k in sd)
    if prefix:sd={k.removeprefix('_orig_mod.'):v for k,v in sd.items()}
    loaded=model.load_state_dict(sd,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert model.config.max_seq_len==2048 if hasattr(model,'config') else cfg['max_seq_len']==2048
    assert sum(p.numel() for p in model.parameters())==124635456
    del sd,ck;model.to('cuda').eval()
    write(EVAL/(tag+'_LOAD.json'),{'checkpoint':str(path),'sha256':sha(path),'strict_load':True,'config':cfg,
                                 'optimizer_restored':False,'scheduler_restored':False,'rng_restored':False,
                                 'stripped_compiled_prefix':prefix,'load_seconds':time.monotonic()-start})
    return model,cfg

@torch.inference_mode()
def generate(model,tok,messages,cap):
    ids=encode_prompt(tok,messages,default_system=None,mode='full_context')
    assert len(ids)+cap<=2048
    with torch.autocast('cuda',dtype=torch.bfloat16):
        gen=model.generate(torch.tensor([ids],device='cuda'),max_new_tokens=cap,temperature=0,top_k=0,top_p=1,eos_id=3)
    new=gen[0,len(ids):].tolist();eos=bool(new and new[-1]==3);before=new[:-1] if eos else new
    text=tok.decode(before,skip_special_tokens=False)
    grams=[tuple(before[i:i+4]) for i in range(max(0,len(before)-3))]
    return {'prompt_token_ids':ids,'generated_token_ids':new,'output_text_for_scoring':text,
            'output_text_raw_including_terminal_eos':tok.decode(new,skip_special_tokens=False),
            'stop_reason':'eos' if eos else 'max_new_tokens','generated_tokens_including_eos':len(new),
            'empty_output':not text,'repeated_4gram_fraction':1-len(set(grams))/len(grams) if grams else 0.,
            'unexpected_control_token_count':sum(v in {0,2,4,5,6} for v in before),
            'answer_cleanup':False,'constrained_decoding':False,'max_new_tokens':cap}

def agg(rs):
    out={'n':len(rs),'pass':sum(r['check']['pass'] for r in rs),
         'accuracy':sum(r['check']['pass'] for r in rs)/len(rs),
         'eos_stop':sum(r['stop_reason']=='eos' for r in rs),'cap_stop':sum(r['stop_reason']=='max_new_tokens' for r in rs),
         'empty_outputs':sum(r['empty_output'] for r in rs),
         'mean_repeated_4gram_fraction':sum(r['repeated_4gram_fraction'] for r in rs)/len(rs)}
    family=rs[0]['family']
    if all(r['family']==family for r in rs):
        if family=='MEMBERSHIP':
            cm={a:{p:sum(r['check']['actual_class']==a and r['check']['predicted_class']==p for r in rs)
                   for p in ['positive','negative','invalid']} for a in ['positive','negative']}
            out['confusion_matrix']=cm
            out['positive_recall']=cm['positive']['positive']/sum(cm['positive'].values())
            out['negative_recall']=cm['negative']['negative']/sum(cm['negative'].values())
            out['constant_positive_class_baseline']=sum(cm['positive'].values())/len(rs)
            out['best_constant_raw_label_baseline']=max(collections.Counter(r['reference'] for r in rs).values())/len(rs)
        elif family=='JSON':
            for k in ['valid_json','exact_key_set','correct_values','duplicate_keys']:out[k]=sum(r['check'][k] for r in rs)
        else:
            for k in ['correct_value_exact','extra_content_with_expected_value','wrong_or_missing_value']:out[k]=sum(r['check'][k] for r in rs)
    return out

@torch.inference_mode()
def evaluate(model,tok,rs,tag):
    model.eval();start=time.monotonic();results=[]
    with (EVAL/(tag+'_OUTPUTS.jsonl')).open('x') as f:
        for i,r in enumerate(rs,1):
            gen=generate(model,tok,r['messages'][:-1],64)
            rec={**{k:r[k] for k in ['audit_id','family','condition','template_id','group_id','matched_payload_id','reference']},
                 'model_tag':tag,**gen,'check':check(r,gen['output_text_for_scoring'])}
            f.write(canonical(rec)+'\n');f.flush();results.append(rec)
            if i%64==0:print(json.dumps({'stage':'generation','tag':tag,'completed':i,'total':len(rs),'seconds':time.monotonic()-start}),flush=True)
    summary={'overall':agg(results),'by_family':{},'by_family_condition':{},'elapsed_seconds':time.monotonic()-start,
             'full_raw_outputs_sha256':sha(EVAL/(tag+'_OUTPUTS.jsonl'))}
    for family in sorted({r['family'] for r in results}):
        fr=[r for r in results if r['family']==family];summary['by_family'][family]=agg(fr)
        for c in sorted({r['condition'] for r in fr}):summary['by_family_condition'][family+'/'+c]=agg([r for r in fr if r['condition']==c])
    paired={}
    for family in sorted({r['family'] for r in results}):
        for value in ['seen','unseen']:
            a={r['matched_payload_id']:r for r in results if r['family']==family and r['condition']=='train_template_'+value}
            b={r['matched_payload_id']:r for r in results if r['family']==family and r['condition']=='heldout_template_'+value}
            if a:
                assert a.keys()==b.keys()
                paired[family+'/'+value]={'n':len(a),'both_pass':sum(a[k]['check']['pass'] and b[k]['check']['pass'] for k in a),
                                         'train_only_pass':sum(a[k]['check']['pass'] and not b[k]['check']['pass'] for k in a),
                                         'heldout_only_pass':sum(not a[k]['check']['pass'] and b[k]['check']['pass'] for k in a),
                                         'neither_pass':sum(not a[k]['check']['pass'] and not b[k]['check']['pass'] for k in a)}
    summary['matched_rendering_pairs']=paired;write(EVAL/(tag+'_SUMMARY.json'),summary)
    print(json.dumps({'stage':'evaluation_complete','tag':tag,**summary['overall']}),flush=True)
    return summary

@torch.inference_mode()
def val500(model,tok,tag):
    start=time.monotonic();val=rows(P2/'data/P2_VALIDATION.jsonl');assert len(val)==500
    collate=collate_fn_builder(tok,2048,None,False,1.0,[],'contains_any')
    loader=DataLoader(val,batch_size=2,shuffle=False,num_workers=0,collate_fn=collate)
    result=evaluate_position_nll(model,loader,device='cuda',autocast_dtype=torch.bfloat16,expected_rows=500)
    by={}
    # Reuse the same NLL evaluator for each actual task; no fresh generation/selection.
    # Per-task retention is also measured by the original 32 Part A validation probes.
    result.update({'validation_sha256':sha(P2/'data/P2_VALIDATION.jsonl'),'elapsed_seconds':time.monotonic()-start,
                   'sequence_length':2048,'full_messages_preserved':True,'no_truncation':True})
    write(EVAL/(tag+'_VAL500.json'),result)
    print(json.dumps({'stage':'val500','tag':tag,'nll':result['val_loss'],'seconds':result['elapsed_seconds']}),flush=True)
    return result

def baseline(tok):
    assert not (EVAL/'BASELINE_COMPLETE.json').exists()
    model,cfg=load_model(CKPT,'p2_baseline')
    evaluate(model,tok,rows(DATA/'development.jsonl'),'p2_development')
    val500(model,tok,'p2')
    evaluate(model,tok,rows(DATA/'TRAIN_SAMPLE128.jsonl'),'p2_train128')
    write(EVAL/'BASELINE_COMPLETE.json',{'status':'COMPLETE','optimizer_updates':0,'source_sha256':EXPECTED})

def encoded_rows(tok):
    lookup={}
    for r in rows(DATA/'train.jsonl')+rows(DATA/'replay.jsonl'):
        ids,labels=encode_chat(tok,r['messages'],default_system=None);assert len(ids)<=512
        lookup[r['audit_id']]={'input_ids':torch.tensor(ids+[0]*(512-len(ids)),dtype=torch.long),
                               'labels':torch.tensor(labels+[-100]*(512-len(labels)),dtype=torch.long),
                               'targets':sum(v!=-100 for v in labels[1:]),'tokens':len(ids),
                               'source':r.get('source','procedural'),'task':r.get('family',r.get('p2_task'))}
    return lookup

def checkpoint(model,opt,cfg,step,status):
    path=TRAIN/f'step_{step:06d}.pt'
    assert not path.exists()
    obj={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'step':step,
         'scheduler':{'canonical':True,'completed_updates':step,'planned_updates':640,'warmup_steps':32,'base_lr':5e-5,'min_lr_ratio':0.1},
         'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate()},
         'scaler':None,'run_config':CONFIG,'plan_sha256':sha(DATA/'UPDATE_PLAN.jsonl'),
         'data_freeze_sha256':sha(DATA/'FROZEN.json'),'status':status}
    save_checkpoint_atomic(str(path),obj)
    write(TRAIN/f'step_{step:06d}_IDENTITY.json',{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'complete_optimizer_scheduler_rng':True})
    return path

def train(tok,process_started):
    assert (EVAL/'BASELINE_COMPLETE.json').exists()
    # Exclusive launch marker survives failure and prevents a second trainer/run.
    with (TRAIN/'SINGLE_TRAINING_LAUNCH.json').open('x') as f:
        json.dump({'pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'python':sys.executable,'argv':sys.argv,'configuration':CONFIG},f,indent=2)
    status={'status':'RUNNING','last_completed_update':0,'pid':os.getpid(),'exit_reason':None}
    write(TRAIN/'STATUS.json',status)
    def alarm(signum,frame):raise TimeoutError('training process reached 7200-second wall cap; no restart')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1,7200-(time.monotonic()-process_started)))
    model=None;opt=None
    try:
        model,cfg=load_model(CKPT,'p3_train_init')
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);random.seed(SEED)
        model.train();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
        assert not opt.state
        enc=encoded_rows(tok);plan=rows(DATA/'UPDATE_PLAN.jsonl');torch.cuda.reset_peak_memory_stats()
        started=time.monotonic();training_compute=0.;peak_allocated=0;peak_reserved=0
        with (TRAIN/'UPDATE_METRICS.jsonl').open('x') as log:
            for u in plan:
                step=u['update'];model.train();batches=[]
                for i in range(0,32,2):
                    ids=u['row_ids'][i:i+2]
                    batches.append({'input_ids':torch.stack([enc[k]['input_ids'] for k in ids]),
                                    'labels':torch.stack([enc[k]['labels'] for k in ids]),'row_ids':ids})
                lr=lr_schedule(step-1,32,5e-5,schedule='cosine',schedule_total_steps=640)
                for g in opt.param_groups:g['lr']=lr
                torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
                result=effective_batch_backward(model,batches,device='cuda',autocast_dtype=torch.bfloat16,nll_sum_fn=assistant_nll_sum_fp32)
                assert result['row_ids']==u['row_ids'] and result['supervised_targets']==sum(enc[k]['targets'] for k in u['row_ids'])
                norm=checked_optimizer_step(model,opt,loss=result['loss'],grad_clip=1.)
                torch.cuda.synchronize();seconds=time.monotonic()-t;training_compute+=seconds
                peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated());peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
                record={**u,**result,'lr':lr,'preclip_grad_norm':norm,'update_seconds':seconds,
                        'formatted_tokens':sum(enc[k]['tokens'] for k in u['row_ids']),
                        'source_target_exposure':dict(collections.Counter({src:sum(enc[k]['targets'] for k in u['row_ids'] if enc[k]['source']==src) for src in {enc[k]['source'] for k in u['row_ids']}})),
                        'task_target_exposure':{task:sum(enc[k]['targets'] for k in u['row_ids'] if enc[k]['task']==task) for task in {enc[k]['task'] for k in u['row_ids']}},
                        'process_elapsed_seconds':time.monotonic()-process_started,
                        'peak_training_cuda_allocated_bytes':peak_allocated}
                log.write(canonical(record)+'\n');log.flush()
                status.update(last_completed_update=step,training_compute_seconds=training_compute,
                              process_elapsed_seconds=time.monotonic()-process_started,
                              peak_training_cuda_allocated_bytes=peak_allocated,
                              peak_training_cuda_reserved_bytes=peak_reserved)
                if step%10==0 or step==1:
                    write(TRAIN/'STATUS.json',status)
                    print(json.dumps({'stage':'train','step':step,'stream':u['stream'],'loss':result['loss'],'lr':lr,'grad_norm':norm,'seconds':seconds,'process_seconds':status['process_elapsed_seconds']}),flush=True)
                if step in (320,640):
                    checkpoint(model,opt,cfg,step,status)
                    evaluate(model,tok,rows(DATA/'development.jsonl'),f'p3_step{step}_development')
                    val500(model,tok,f'p3_step{step}')
                    torch.cuda.reset_peak_memory_stats()
        status.update(status='COMPLETED',exit_reason='max_updates_reached',process_elapsed_seconds=time.monotonic()-process_started,
                      training_loop_wall_seconds=time.monotonic()-started)
        write(TRAIN/'STATUS.json',status);signal.setitimer(signal.ITIMER_REAL,0)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        status.update(status='STOPPED',exit_reason=type(exc).__name__,error=str(exc),process_elapsed_seconds=time.monotonic()-process_started,
                      no_automatic_restart=True)
        write(TRAIN/'STATUS.json',status)
        # Only already completed checkpoint files are safe after an asynchronous failure.
        raise

@torch.inference_mode()
def part_a(model,tok):
    ps=[r for r in rows(H/'IN_DISTRIBUTION_PROBES_WITH_REFERENCES.jsonl') if r['origin_split']=='validation'];assert len(ps)==32
    saved={r['probe_id']:r for r in rows(ROOT/'runs/p2_learnability_diagnostic_20260906/part_a/PROBE_OUTPUTS_p2_step750.jsonl')}
    results=[]
    with (EVAL/'P3_PART_A_VALIDATION32.jsonl').open('x') as f:
        for r in ps:
            gen=generate(model,tok,r['messages'],384)
            ids,labels=encode_completion(tok,r['messages'],r['reference_completion'],default_system=None)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(torch.tensor([ids],device='cuda'))
                nll=float(assistant_nll_sum_fp32(logits,torch.tensor([labels],device='cuda')))
            nt=sum(v!=-100 for v in labels[1:]);del logits
            rec={**r,**gen,'model':'P3','reference_nll_sum':nll,'reference_target_tokens':nt,'reference_nll_per_token':nll/nt,
                 'p2_cached_output':saved[r['probe_id']]['output_text_for_scoring'],'content_status':'PENDING_MANUAL_REVIEW',
                 'exact_reference_accuracy_not_used':True}
            f.write(canonical(rec)+'\n');f.flush();results.append(rec)
            if len(results)%8==0:print(json.dumps({'stage':'part_a','completed':len(results)}),flush=True)
    def summary(rs):return {'n':len(rs),'eos_stop':sum(r['stop_reason']=='eos' for r in rs),'empty':sum(not r['output_text_for_scoring'] for r in rs),
                           'cap_stop':sum(r['stop_reason']=='max_new_tokens' for r in rs),
                           'mean_repeated_4gram_fraction':sum(r['repeated_4gram_fraction'] for r in rs)/len(rs),
                           'reference_nll':sum(r['reference_nll_sum'] for r in rs)/sum(r['reference_target_tokens'] for r in rs)}
    report={}
    for tag,rs in [('P3',results),('P2',[saved[r['probe_id']] for r in ps])]:
        report[tag]={'overall':summary(rs),'by_task':{t:summary([r for r in rs if r['primary_task']==t]) for t in {r['primary_task'] for r in rs}}}
    write(EVAL/'PART_A_SUMMARY.json',report)

def final_eval(tok):
    status=json.loads((TRAIN/'STATUS.json').read_text());assert status['last_completed_update']>0
    step=status['last_completed_update'];path=TRAIN/f'step_{step:06d}.pt';assert path.exists()
    # Final tests become inspected only AFTER the one training process has ended.
    write(EVAL/'FINAL_EVALUATION_OPENED.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_status':status,
                                             'data_sha256':sha(DATA/'final.jsonl'),'settings':CONFIG})
    model,cfg=load_model(CKPT,'p2_final');evaluate(model,tok,rows(DATA/'final.jsonl'),'p2_final');del model;torch.cuda.empty_cache()
    model,cfg=load_model(path,'p3_final');evaluate(model,tok,rows(DATA/'final.jsonl'),'p3_final')
    evaluate(model,tok,rows(DATA/'TRAIN_SAMPLE128.jsonl'),'p3_train128')
    part_a(model,tok);del model;torch.cuda.empty_cache()
    write(EVAL/'FINAL_NATIVE_COMPLETE.json',{'status':'COMPLETED','final_checkpoint':str(path)})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['baseline','train','final']);args=parser.parse_args()
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(SEED);random.seed(SEED)
    verify_freeze();assert torch.cuda.is_available();tok=load_chat_tokenizer(str(TOK))
    if args.mode=='baseline':baseline(tok)
    elif args.mode=='train':train(tok,started)
    else:final_eval(tok)
    print(json.dumps({'stage':'process_complete','mode':args.mode,'elapsed_seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':main()
