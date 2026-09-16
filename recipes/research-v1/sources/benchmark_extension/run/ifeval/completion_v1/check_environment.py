from pathlib import Path
import sys,os,json,hashlib,base64,importlib.metadata as md,datetime
R=Path(__file__).resolve().parents[2]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',NLTK_DATA=str(R/'nltk_data'))
sys.path[:0]=[str(R/'evaluator_deps'),str(R/'runtime')]
prior=json.loads((R/'protocol/EVALUATOR_ENVIRONMENT.json').read_text());checked={}
for name,v in prior['versions'].items():
 d=md.distribution(name);assert d.version==v,(name,d.version,v);entries=[]
 for f in d.files or []:
  if f.hash:
   p=Path(d.locate_file(f))
   if '..' in f.parts:
    entries.append({'file':str(p),'external_console_entry':True,'exists':p.exists(),'note':'pip --target RECORD console entry; not an imported verifier dependency'});continue
   b=p.read_bytes();h=hashlib.new(f.hash.mode,b).digest();assert base64.urlsafe_b64encode(h).decode().rstrip('=')==f.hash.value,str(p);entries.append({'file':str(p),'sha256':hashlib.sha256(b).hexdigest()})
 checked[name]={'version':v,'record_verified_files':entries}
import nltk
assert nltk.data.find('tokenizers/punkt_tab')
import verify_ifeval
assert json.loads((R/'ifeval/verifier/FIXTURES.json').read_text())['status']=='PASS'
base=json.loads((R/'protocol/ENVIRONMENT.json').read_text())
for name,v in base['versions'].items():assert md.version(name)==v,(name,md.version(name),v)
result={'status':'PASS','fixtures_reused_without_rerun':True,'packages':checked,'source_data_resource_package_archive_identities':'PASS in PRIOR_EVIDENCE_IDENTITIES checks','all_verifier_imports':'PASS','network_downloads':0,'dependency_installations':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'ifeval/completion_v1/ENVIRONMENT_RECHECK.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS: prior verifier versions, installed RECORD file hashes, imports and punkt_tab availability; fixtures reused without rerun')
