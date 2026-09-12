"""Exactly two CPU FP32 parameter blends. No optimization or inference search."""
import collections
import datetime
import hashlib
import sys
import time
sys.path.insert(0,'.')
import torch
from src.model import GPT,gpt_config_from_checkpoint_dict
from src.chat_template import load_chat_tokenizer
from sft.train_sft import load_ckpt,save_checkpoint_atomic
from common import *

def interpolate(a,b,alpha):
    # Special-case mathematical endpoints: subtraction/addition need not return
    # bitwise b in floating arithmetic. Endpoints are checked, never published.
    if alpha==0:return a.float().clone()
    if alpha==1:return b.float().clone()
    return a.float()+alpha*(b.float()-a.float())

def state_digest(sd):
    h=hashlib.sha256()
    for k,v in sorted(sd.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.set_grad_enabled(False)
    prep=RUN/'preparation'
    with (prep/'MATERIALIZATION_LAUNCH.json').open('x') as f:f.write(canonical({'python':sys.executable,'argv':sys.argv,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    assert sha(TOK)==TOK_SHA;load_chat_tokenizer(str(TOK))
    # Only model/config are retained from the scoped safe loader's dict payload.
    states=[];configs=[];identities=[]
    for path,expected in PARENTS:
        actual=sha(path);assert actual==expected
        ck=load_ckpt(str(path));sd=ck['model'];cfg=ck.get('config') or ck.get('cfg')
        if any(k.startswith('_orig_mod.') for k in sd):sd={k.removeprefix('_orig_mod.'):v for k,v in sd.items()}
        states.append(sd);configs.append(cfg);del ck
        identities.append({'path':str(path),'sha256':actual,'bytes':path.stat().st_size,'state_digest_before':state_digest(sd)})
    a,b=states;assert configs[0]==configs[1];cfg=configs[0]
    model=GPT(gpt_config_from_checkpoint_dict(cfg)).eval().requires_grad_(False)
    param_map=dict(model.named_parameters(remove_duplicate=False));model_keys=set(model.state_dict())
    assert set(a)==set(b)==model_keys
    aliases=collections.defaultdict(list)
    for k,p in param_map.items():aliases[id(p)].append(k)
    groups=list(aliases.values());assert sum(p.numel() for p in model.parameters())==124635456
    fixed_keys=sorted(model_keys-set(param_map))
    assert model.tok_emb.weight is model.lm_head.weight
    assert torch.equal(a['tok_emb.weight'],a['lm_head.weight']) and torch.equal(b['tok_emb.weight'],b['lm_head.weight'])
    for k in a:
        assert a[k].shape==b[k].shape==model.state_dict()[k].shape and a[k].dtype==b[k].dtype
        if a[k].is_floating_point():assert bool(torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all())
        if k in fixed_keys or not a[k].is_floating_point():assert torch.equal(a[k],b[k])
    # Nonpersistent RoPE caches are derived from identical config, not blendable
    # checkpoint state. Compare fresh architecture buffers under the same config.
    twin=GPT(gpt_config_from_checkpoint_dict(configs[1])).eval().requires_grad_(False)
    mb=dict(model.named_buffers());tb=dict(twin.named_buffers());assert mb.keys()==tb.keys()
    assert all(torch.equal(mb[k],tb[k]) for k in mb);del twin,tb
    x=torch.tensor([-.75,2.,8.,-16.]);y=torch.tensor([.25,6.,4.,16.]);xs=x.clone();ys=y.clone()
    assert torch.equal(interpolate(x,y,0),x) and torch.equal(interpolate(x,y,1),y)
    assert torch.equal(interpolate(x,y,.5),torch.tensor([-.25,4.,6.,0.]))
    assert torch.equal(interpolate(x,y,.75),torch.tensor([0.,5.,5.,8.]))
    assert torch.equal(x,xs) and torch.equal(y,ys)
    # Exact endpoint parameter checks on actual parents at their source precision.
    assert all(a[k].dtype==torch.float32 for k in param_map)
    for keys in groups:
        k=keys[0];assert torch.equal(interpolate(a[k],b[k],0),a[k]);assert torch.equal(interpolate(a[k],b[k],1),b[k])
    artifacts=[];per_parameter=[]
    for tag,alpha in CANDIDATES.items():
        out={}
        for keys in groups:
            k=keys[0]
            if a[k].is_floating_point():v=interpolate(a[k],b[k],alpha)
            else:assert torch.equal(a[k],b[k]);v=a[k].clone()
            assert v.data_ptr()!=a[k].data_ptr() and v.data_ptr()!=b[k].data_ptr()
            assert v.dtype==torch.float32 and bool(torch.isfinite(v).all())
            # Independent convex-combination computation bounds FP32 roundoff;
            # exact serialized state is checked separately below.
            oracle=(1-alpha)*a[k].double()+alpha*b[k].double()
            error=float((v.double()-oracle).abs().max());scale=max(1.,float(oracle.abs().max()))
            assert error<=4*torch.finfo(torch.float32).eps*scale
            per_parameter.append({'candidate':tag,'parameter':k,'aliases':keys,'shape':list(v.shape),'dtype':str(v.dtype),'fp64_oracle_max_abs_error':error})
            for name in keys:
                assert torch.equal(a[name],a[k]) and torch.equal(b[name],b[k]);out[name]=v
            del oracle
        for k in fixed_keys:out[k]=a[k].clone()
        assert set(out)==model_keys and out['tok_emb.weight'] is out['lm_head.weight']
        model.load_state_dict(out,strict=True);assert model.tok_emb.weight is model.lm_head.weight
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in out.items())
        metadata={'artifact_kind':'inference_only_weight_interpolation','alpha':alpha,
                  'formula':'theta_P2 + alpha * (theta_P3_step320 - theta_P2)',
                  'parents':[{k:v for k,v in p.items() if k in ('path','sha256')} for p in identities],
                  'tokenizer_sha256':TOK_SHA,'optimizer_updates':0,'source_training_step':None}
        path=RUN/'weights'/(tag+'.pt');assert not path.exists()
        save_checkpoint_atomic(str(path),{'model':out,'config':cfg,'derivation':metadata})
        reload=load_ckpt(str(path));assert set(reload)=={'model','config','derivation'}
        assert reload['config']==cfg and reload['derivation']==metadata
        assert all(torch.equal(v,reload['model'][k]) for k,v in out.items())
        model.load_state_dict(reload['model'],strict=True)
        assert model.tok_emb.weight is model.lm_head.weight
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in out.items())
        artifacts.append({'tag':tag,'alpha':alpha,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
                          'model_state_digest':state_digest(out),'strict_load_save_reload_exact':True,'tied_parameters_shared_after_load':True})
        del reload,out
    for sd,ident,(path,expected) in zip(states,identities,PARENTS):
        ident['state_digest_after']=state_digest(sd);ident['sha256_after']=sha(path)
        assert ident['state_digest_before']==ident['state_digest_after'] and ident['sha256_after']==expected
    jsonl(prep/'PARAMETER_CHECKS.jsonl',per_parameter)
    result={**FLAGS,'STATUS':'MATERIALIZATION_CHECKS_PASSED','NEW_INFERENCE_WEIGHT_ARTIFACTS':len(artifacts),
            'parents':identities,'config':cfg,'config_sha256':hashlib.sha256(canonical(cfg).encode()).hexdigest(),
            'tokenizer_path':str(TOK),'tokenizer_sha256':TOK_SHA,'artifacts':artifacts,
            'unique_parameter_tensors':len(groups),'state_keys':len(model_keys),'fixed_state_keys':fixed_keys,
            'nonpersistent_architecture_buffers_equal':list(mb),'toy_arithmetic_passed':True,
            'actual_endpoint_parameters_exact':True,'real_endpoint_forward_rerun':False,
            'all_parameters_finite':True,'no_parent_storage_mutation':True,'elapsed_seconds':time.perf_counter()-started}
    write(prep/'MATERIALIZATION_CHECKS.json',result)
    print(json.dumps({'status':result['STATUS'],'artifacts':artifacts,'seconds':result['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
