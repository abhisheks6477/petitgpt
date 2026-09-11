"""Local native alpha075 loading and precision profiles."""
import argparse,json
from pathlib import Path
from contextlib import contextmanager
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_model
from src.model import GPT,gpt_config_from_checkpoint_dict,audit_gpt_parameter_count
from src.chat_template import load_chat_tokenizer,encode_prompt
from src.accepted_generate import generate as accepted_generate

@contextmanager
def precision(profile):
    if profile not in ('bf16_native','fp32_math'): raise ValueError('Unsupported precision profile')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=profile=='bf16_native'
    with sdpa_kernel([SDPBackend.MATH] if profile=='fp32_math' else [SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH,SDPBackend.CUDNN_ATTENTION]):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=profile=='bf16_native'):
            yield

def load_bundle(model_dir):
    root=Path(model_dir).resolve()
    cfg=gpt_config_from_checkpoint_dict(json.loads((root/'config.json').read_text()))
    model=GPT(cfg).eval()
    load_model(model,str(root/'model.safetensors'),strict=True,device='cpu')
    audit=audit_gpt_parameter_count(model,cfg)
    if audit['actual_total']!=124635456: raise ValueError('Parameter identity guard failed')
    if model.tok_emb.weight is not model.lm_head.weight: raise ValueError('Embedding tie lost')
    if model.tok_emb.weight.data_ptr()!=model.lm_head.weight.data_ptr(): raise ValueError('Storage tie lost')
    if any(p.dtype!=torch.float32 for p in model.parameters()): raise ValueError('Expected FP32 weights')
    return model,load_chat_tokenizer(str(root/'tokenizer.json'))

def validate_input(tok,messages,cap):
    if not isinstance(messages,list) or not messages: raise ValueError('Messages must be a non-empty JSON array')
    if any(not isinstance(m,dict) or set(m)!={'role','content'} for m in messages): raise ValueError('Each message must contain only role and content')
    if isinstance(cap,bool) or not isinstance(cap,int) or not 1<=cap<=384: raise ValueError('max_new_tokens must be 1..384')
    ids=encode_prompt(tok,messages,default_system=None,mode='full_context')
    if len(ids)+cap>2048: raise ValueError('Context overflow: prompt plus token budget exceeds 2048')
    return ids

@torch.inference_mode()
def generate(model,tok,messages,cap=384,profile='bf16_native'):
    validate_input(tok,messages,cap)
    model.eval()
    with precision(profile):
        return accepted_generate(model,tok,messages,cap,autocast_enabled=profile=='bf16_native')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory',required=True,type=Path)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--messages-json',type=Path);g.add_argument('--prompt')
    p.add_argument('--profile',choices=['bf16_native','fp32_math'],default='bf16_native')
    p.add_argument('--max-new-tokens',type=int,default=384)
    a=p.parse_args()
    messages=json.loads(a.messages_json.read_text()) if a.messages_json else [{'role':'user','content':a.prompt}]
    tok=load_chat_tokenizer(str(a.model_directory/'tokenizer.json'))
    validate_input(tok,messages,a.max_new_tokens)
    model,tok=load_bundle(a.model_directory)
    model=model.to('cuda').eval()
    print(json.dumps(generate(model,tok,messages,a.max_new_tokens,a.profile),ensure_ascii=False))
if __name__=='__main__': main()
