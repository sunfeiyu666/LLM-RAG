"""Package the reproducible T7 handoff, excluding pilots and unrelated caches."""
import json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/t7'

def request_hashes(value):
    found=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='request_hash' and isinstance(item,str):found.add(item)
            else:found.update(request_hashes(item))
    elif isinstance(value,list):
        for item in value:found.update(request_hashes(item))
    return found

def main():
    assert json.loads((OUT/'verification.json').read_text(encoding='utf-8'))['passed']
    files=[]
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.md','.txt','.csv'})
    for folder in ['src','tools','data']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(p for p in OUT.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl'} and p.name!='payload_file_hashes.json')
    files.extend(OUT/name for name in ['run.log','unit_tests.log','reproduction.log','correctness_new.log','evidence_recheck.log'])
    files.extend((OUT/'figures').glob('*.png'))
    hashes=set()
    for name in ['faithfulness.jsonl','evidence_link_repairs.jsonl','correctness_new.jsonl']:
        for line in (OUT/name).read_text(encoding='utf-8').splitlines():hashes.update(request_hashes(json.loads(line)))
    for key in hashes:
        cache=OUT/'cache'/f'{key}.json';assert cache.exists(),cache
        files.append(cache)
    files=sorted(set(files))
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path=OUT/'payload_file_hashes.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    files.append(manifest_path)
    target=ROOT.parent/'T7_系统评估代码与报告.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        for p in files:archive.write(p,'LLM-RAG-T7/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for relative,expected in manifest.items():
            assert hashlib.sha256(archive.read('LLM-RAG-T7/'+relative)).hexdigest()==expected
    print(json.dumps({'package':str(target),'files':len(files),'official_call_caches':len(hashes),
        'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},ensure_ascii=False))

if __name__=='__main__':main()
