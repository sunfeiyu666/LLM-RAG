"""Freeze the chosen T5 answers and reconstruct their actual uncompressed Context."""
import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from t7_updated_core import UP,OUT,KS,read_jsonl,write_jsonl,csv_rows,sha,rank_metrics,rouge_l,normalized,answer_units

def main():
    OUT.mkdir(parents=True,exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    snapshots=json.loads((ROOT/'upstream_manifest.json').read_text(encoding='utf-8'))
    for record in snapshots:assert sha(ROOT/record['path'])==record['sha256'],record['path']
    qa_path=UP/'data/eval/qa_set.jsonl';qs=read_jsonl(qa_path);byq={q['id']:q for q in qs}
    t5meta=json.loads((UP/'results/t5_final_20261026/run_metadata.json').read_text(encoding='utf-8'))
    assert len(qs)==20 and sha(qa_path)==t5meta['qa_sha256']
    allrows=read_jsonl(UP/'results/t5_final_20261026/t5_full_rows.jsonl')
    assert len(allrows)==120
    selected=[r for r in allrows if r['strategy']=='direct' and r['compressed'] is False]
    assert len(selected)==20 and {r['id'] for r in selected}==set(byq)
    target={'created_at':datetime.now(timezone.utc).isoformat(),
        'target':'Upstream T5 BM25 top-5 candidates -> direct top-3 -> uncompressed -> Qwen real saved answers',
        'selected_before_new_faithfulness_scoring':True,
        'reason':'Upstream T4 chose BM25; T5 direct/no-compression had the highest upstream accuracy and citation scores among its six configurations.',
        'scope':'Offline evaluation of the actual T4/T5 202-chunk system, not an unrun integration of T2 best chunks or T3 optimizers.',
        't4_t5_commit':next(r['commit'] for r in snapshots if r['ref']=='origin/t4-t5-zhuxiangduo'),
        'main_t2_commit':next(r['commit'] for r in snapshots if r['ref']=='origin/main'),
        'qa_sha256':sha(qa_path),'upstream_answers_sha256':sha(UP/'results/t5_final_20261026/t5_full_rows.jsonl'),
        'generation_parameters':t5meta['generation_parameters'],'model_digest':t5meta['model_digest']}
    manifest=ROOT/'target_manifest.json'
    if not manifest.exists():manifest.write_text(json.dumps(target,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        old=json.loads(manifest.read_text(encoding='utf-8'))
        for key in ['qa_sha256','upstream_answers_sha256','t4_t5_commit','main_t2_commit']:assert old[key]==target[key]
    # Imported upstream code is preserved byte-for-byte in the provenance snapshot.
    sys.path.insert(0,str(UP/'src'))
    from kb import load_pages,split_documents,format_document
    from rag_chain import format_context
    from retrieval_compare import BM25
    documents=split_documents(load_pages());byid={d.metadata['chunk_id']:d for d in documents}
    assert len(documents)==202
    bm25=BM25(documents)
    frozen=[{'page_content':d.page_content,'metadata':d.metadata} for d in documents]
    (ROOT/'data/frozen_chunks.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'data/qa_set.jsonl').write_bytes(qa_path.read_bytes())
    upstream_review=csv_rows(UP/'results/t5_final_20261026/t5_answer_review.csv')
    scores={r['id']:r for r in upstream_review if r['strategy']=='direct' and r['compressed']=='False'}
    assert len(scores)==20
    saved_t4={r['id']:r for r in read_jsonl(UP/'results/t4_bge_20261025/t4_rankings.jsonl') if r['method']=='bm25'}
    metrics=[];answers=[];packets=[]
    for row in selected:
        q=byq[row['id']];r=scores[q['id']]
        assert row['question']==q['question'] and row['gold_answer']==q['gold_answer']
        assert row['gold_chunk_ids']==q['gold_chunk_ids'] and row['answer']==r['answer']
        ranking=[d.metadata['chunk_id'] for d in bm25.search(q['question'],len(documents))]
        assert ranking[:10]==saved_t4[q['id']]['retrieved_chunk_ids']
        assert ranking[:3]==row['selected_chunk_ids']
        selected_docs=[byid[cid] for cid in row['selected_chunk_ids']]
        assert sum(len(d.page_content) for d in selected_docs)==row['context_chars']
        context=format_context(selected_docs)
        a={**row,'context':context,'context_sha256':hashlib.sha256(context.encode()).hexdigest(),
            'context_status':'Reconstructed from exact upstream formatter, matching selected IDs and stored raw context character count; original full Context string and finish_reason not saved upstream.'}
        answers.append(a)
        m={'id':q['id'],'level':q['level'],'question':q['question'],**rank_metrics(ranking,q['gold_chunk_ids']),
            'gold_ranks':{g:ranking.index(g)+1 for g in q['gold_chunk_ids']},'ranking':ranking,
            'candidate_top5':ranking[:5],'actual_top3':ranking[:3],
            'upstream_accuracy_0_2':int(r['accuracy_0_2']),'upstream_citation_0_2':int(r['citation_support_0_2']),
            'upstream_review_note':r['notes'],'rouge_l_char':rouge_l(row['answer'],q['gold_answer']),
            'exact_match':int(normalized(row['answer'])==normalized(q['gold_answer'])),
            'context_tokens':row['context_tokens'],'generation_output_tokens':row['generation_output_tokens'],
            'total_seconds':row['total_seconds'],'reported_output_at_limit':row['generation_output_tokens']>=t5meta['generation_parameters']['num_predict']}
        metrics.append(m)
        packets.append({'id':q['id'],'level':q['level'],'question':q['question'],'answer':row['answer'],
            'answer_units':answer_units(row['answer']),
            'actual_documents':[{'block':i,'chunk_id':d.metadata['chunk_id'],'formatted_text':format_document(d)} for i,d in enumerate(selected_docs,1)],
            'note':'Source-only faithfulness review packet; deliberately excludes gold_answer and upstream scores.'})
    write_jsonl(ROOT/'data/system_answers.jsonl',answers);write_jsonl(OUT/'retrieval_metrics.jsonl',metrics);write_jsonl(OUT/'faithfulness_packets.jsonl',packets)
    worst=sorted(metrics,key=lambda r:(r['upstream_accuracy_0_2'],r['upstream_citation_0_2'],r['rouge_l_char'],r['id']))[:5]
    (OUT/'worst_five_preselected.json').write_text(json.dumps(worst,ensure_ascii=False,indent=2),encoding='utf-8')
    units=['# 新版T7实际回答单元（用于证据复核）']
    for p in packets:
        units += [f"## {p['id']} {p['level']}",p['question']]
        units += [f"{u['id']} {u['text']}" for u in p['answer_units']]
    (OUT/'answer_units.txt').write_text('\n\n'.join(units),encoding='utf-8')
    print(json.dumps({'prepared':True,'questions':20,'chunks':202,'T4_rankings_match':20,'T5_selected_context_chars_match':20,
        'worst_ids':[r['id'] for r in worst],'upstream_scores':[(m['id'],m['upstream_accuracy_0_2']) for m in metrics]},ensure_ascii=False))

if __name__=='__main__':main()
