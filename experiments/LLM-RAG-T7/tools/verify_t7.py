"""Offline integrity checks against frozen inputs, rankings and literal evidence."""
import json,hashlib,sys,math,csv
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from t3_core import BM25
from t7_core import read_jsonl,rank_metrics,format_document,answer_units,normalize

def main():
    out=ROOT/'results/t7';binding=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))['binding']
    for key,relative in [('qa_sha256','data/eval/qa_set.jsonl'),('answers_sha256','data/system_answers.jsonl'),
                         ('chunks_sha256','data/frozen_chunks.json'),('correctness_scores_sha256','data/t3_judgments.jsonl')]:
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==binding[key],key
    qs={r['id']:r for r in read_jsonl(ROOT/'data/eval/qa_set.jsonl')}
    answers={r['id']:r for r in read_jsonl(ROOT/'data/system_answers.jsonl')}
    frozen=json.loads((ROOT/'data/frozen_chunks.json').read_text(encoding='utf-8'))
    chunks={d['metadata']['chunk_id']:d for d in frozen}
    bm25=BM25([SimpleNamespace(**d) for d in frozen])
    metrics={r['id']:r for r in read_jsonl(out/'per_question_metrics.jsonl')}
    reviewed=read_jsonl(out/'faithfulness_reviewed.jsonl')
    assert len(qs)==len(answers)==len(metrics)==len(reviewed)==20 and len(chunks)==202
    facts=0;unsupported=0
    for r in reviewed:
        qid=r['id'];q=qs[qid];a=answers[qid]
        ranking=[x['chunk_id'] for x in bm25.rank(q['question'],limit=len(chunks))]
        assert ranking[:5]==a['top_chunk_ids']==metrics[qid]['actual_top5']
        assert '\n\n'.join(f'[{i}] {format_document(chunks[cid])}' for i,cid in enumerate(ranking[:5],1))==a['context']
        for key,value in rank_metrics(ranking,q['gold_chunk_ids']).items():
            assert math.isclose(value,metrics[qid][key],abs_tol=1e-12), (qid,key)
        units=answer_units(a['answer']);by_unit={u['id']:u['text'] for u in units}
        assert r['units']==units
        assert set(by_unit)==({c['unit_id'] for c in r['claims']}|{u['id'] for u in r['excluded_units']})
        sentences={s['id']:s for s in r['evidence_sentences']}
        for c in r['claims']:
            assert c['unit_text']==by_unit[c['unit_id']]
            assert c['supported']==(c['score']>=4)
            assert c['evidence']
            for e in c['evidence']:
                assert e['chunk_id'] in ranking[:5]
                assert e['quote']==sentences[e['sentence_id']]['text']
                assert normalize(e['quote']) in normalize(format_document(chunks[e['chunk_id']]))
        facts+=r['fact_count'];unsupported+=r['unsupported_fact_count']
        assert r['fact_count']==len(r['claims'])
        assert r['unsupported_fact_count']==sum(not c['supported'] for c in r['claims'])
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert facts==s['overall']['fact_count'] and unsupported==s['overall']['unsupported_fact_count']
    assert math.isclose(unsupported/facts,s['overall']['hallucination_micro'])
    reviews=json.loads((out/'failure_reviews.json').read_text(encoding='utf-8'))
    worst=json.loads((out/'worst_five.json').read_text(encoding='utf-8'))
    assert len(reviews)==5 and {r['id'] for r in reviews}=={r['id'] for r in worst}
    for r in reviews:
        assert r['answer_excerpt'] in answers[r['id']]['answer']
        for e in r['evidence']:assert e['quote'] in chunks[e['chunk_id']]['page_content']
    rows=json.loads((out/'per_question.json').read_text(encoding='utf-8'))
    keys=['id','level','recall_at_1','recall_at_3','recall_at_5','recall_at_10','precision_at_1','precision_at_3','precision_at_5','precision_at_10',
          'gold_coverage_at_5','mrr_full','map_full','mrr_at_10','map_at_10','correctness_0_5','new_model_correctness_0_5','t3_correctness_0_5',
          'faithfulness_1_5','fact_count','unsupported_fact_count','hallucination_rate','generation_truncated']
    with (ROOT/'T7完整评估表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    report={'passed':True,'questions':20,'chunks':202,'verified_real_contexts':20,'reviewed_facts':facts,'unsupported':unsupported,
            'worst_cases':5,'figures':4,'checks':['input SHA256 binding','full BM25 ranking and metrics','actual Context exact equality',
            'every answer unit classified','all evidence drawn from actual top5','fact denominator and micro average','five literal failure cases'],
            'limitation':'Integrity checks do not independently validate semantic entailment or subjective rubric judgments.'}
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
