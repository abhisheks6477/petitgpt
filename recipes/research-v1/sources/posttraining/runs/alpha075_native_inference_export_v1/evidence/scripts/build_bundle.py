# Historical generation-code adaptation excerpt only.
# Original builder also copied private review artifacts; that orchestration is omitted.
# R, B and E are logical output paths supplied by the future integration adapter.
from pathlib import Path
import difflib
src=Path('./runs/r1_behavior_onepass_mix_v1/runtime/accepted_generate.py').read_text()
dst=src.replace('def generate(model,tok,messages,cap):','def generate(model,tok,messages,cap,autocast_enabled=True):').replace("with torch.autocast('cuda',dtype=torch.bfloat16):","with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast_enabled):")
(B/'src/accepted_generate.py').write_text(dst)
(E/'SOURCE_CODE.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),dst.splitlines(True),fromfile='bound/accepted_generate.py',tofile='bundle/src/accepted_generate.py')))
