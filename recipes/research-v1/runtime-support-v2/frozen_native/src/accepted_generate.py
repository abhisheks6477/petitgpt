import torch
from src.chat_template import encode_prompt

@torch.inference_mode()
def generate(model,tok,messages,cap,autocast_enabled=True):
    ids=encode_prompt(tok,messages,default_system=None,mode='full_context')
    assert len(ids)+cap<=2048
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast_enabled):
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
