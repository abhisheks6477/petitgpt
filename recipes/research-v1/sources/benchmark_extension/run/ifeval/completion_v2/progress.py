"""Read-only progress view; does not run or restart evaluation."""
from pathlib import Path
import json,collections
v=Path(__file__).resolve().parent
for p in sorted((v/'generations').glob('*.jsonl')):
 raw=p.read_bytes();ls=raw.splitlines() if raw.endswith(b'\n') else raw.splitlines()[:-1];complete=[json.loads(x) for x in ls if x]
 print(json.dumps({'model':p.stem,'completed':len(complete),'cap_hits':sum(g['stop_reason']=='max_new_tokens' for g in complete),'generated_tokens':sum(g['generated_token_count'] for g in complete),'generation_seconds':round(sum(g['seconds'] for g in complete),1),'scored':(v/f'RESULTS_{p.stem}.json').exists()},ensure_ascii=False))
print('STOP_RUNTIME', (v/'STOP_RUNTIME.json').exists())
print('LOG_TAIL', (v/'full.log').read_text()[-400:])
