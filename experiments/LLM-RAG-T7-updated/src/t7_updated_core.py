"""Offline T7 evaluation of the selected, frozen upstream T5 system."""
import json,csv,re,hashlib,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'upstream/t4_t5'
OUT=ROOT/'results/t7'
KS=(1,3,5,10)

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalized(s):return re.sub(r'\s+|[*`_#]','',s).lower()

def answer_units(text):
    parts=[s.strip() for s in re.split(r'\n+|(?<=[。！？；])',text) if s.strip()]
    return [{'id':f's{i:03}','text':s} for i,s in enumerate(parts,1)]

def rank_metrics(ids,gold):
    if not gold or len(ids)!=len(set(ids)):raise ValueError('Empty gold or duplicated ranking')
    relevant=set(gold);result={}
    for k in KS:
        count=len(set(ids[:k])&relevant)
        result.update({f'recall_at_{k}':int(count>0),f'precision_at_{k}':count/k,
                       f'gold_coverage_at_{k}':count/len(relevant)})
    for cap,label in [(10,'at_10'),(len(ids),'full')]:
        positions=[i for i,cid in enumerate(ids[:cap],1) if cid in relevant]
        result['mrr_'+label]=1/positions[0] if positions else 0
        result['map_'+label]=sum(j/i for j,i in enumerate(positions,1))/len(relevant)
    return result

def rouge_l(a,b):
    a=re.findall(r'[\u4e00-\u9fffa-z0-9]',a.lower());b=re.findall(r'[\u4e00-\u9fffa-z0-9]',b.lower())
    if not a or not b:return 0.0
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b,1):current.append(previous[j-1]+1 if x==y else max(previous[j],current[-1]))
        previous=current
    return 2*previous[-1]/(len(a)+len(b))

def mean(values):
    values=[v for v in values if v is not None]
    return statistics.fmean(values) if values else None

def aggregate(rows):
    keys=[f'{metric}_at_{k}' for metric in ['recall','precision','gold_coverage'] for k in KS]
    keys+=['mrr_full','map_full','mrr_at_10','map_at_10','upstream_accuracy_0_2','upstream_citation_0_2','rouge_l_char','exact_match',
           'faithfulness_1_5','context_tokens','generation_output_tokens','total_seconds']
    result={k:mean(r[k] for r in rows) for k in keys if rows and k in rows[0]}
    result['n']=len(rows)
    if rows and 'fact_count' in rows[0]:
        n=sum(r['fact_count'] for r in rows);u=sum(r['unsupported_fact_count'] for r in rows)
        result.update(fact_count=n,unsupported_fact_count=u,hallucination_micro=u/n if n else None,
                      hallucination_macro=mean(r['hallucination_rate'] for r in rows),no_fact_answers=sum(r['fact_count']==0 for r in rows))
    return result

def validate_review(review,packet):
    assert review['packet_sha256']==hashlib.sha256(json.dumps(packet,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    units={u['id']:u['text'] for u in packet['answer_units']}
    used={c['unit_id'] for c in review['claims']};excluded={u['id'] for u in review['excluded_units']}
    assert used|excluded==set(units) and not used&excluded
    blocks={d['chunk_id']:d['formatted_text'] for d in packet['actual_documents']}
    for c in review['claims']:
        assert 1<=c['score']<=5 and c['supported']==(c['score']>=4)
        assert c['unit_text']==units[c['unit_id']]
        assert not c['supported'] or c['evidence']
        for e in c['evidence']:
            assert e['chunk_id'] in blocks and e['quote'] and e['quote'] in blocks[e['chunk_id']]
    n=len(review['claims']);u=sum(not c['supported'] for c in review['claims'])
    assert n==review['fact_count'] and u==review['unsupported_fact_count']
    assert review['faithfulness_1_5']==mean(c['score'] for c in review['claims'])
    assert review['hallucination_rate']==(u/n if n else None)
