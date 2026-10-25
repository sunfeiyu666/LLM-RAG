"""Evaluate the explicitly selected frozen T3 BM25 system, no re-generation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

from t3_core import BM25
from t7_core import (MODEL, ENDPOINT, read_jsonl, append_jsonl, normalize,
                     format_document, rank_metrics, EvidenceJudge, judge_faithfulness)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/t7'


def log(message):
    print(datetime.now().strftime('%H:%M:%S'), message, flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','metrics','faithfulness'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    qs = read_jsonl(ROOT/'data/eval/qa_set.jsonl')
    assert len(qs) == 20 and Counter(q['level'] for q in qs) == Counter({'L1':5,'L2':5,'L3':5,'L4':5})
    answers = {r['id']: r for r in read_jsonl(ROOT/'data/system_answers.jsonl')}
    t3_scores = {r['id']: r for r in read_jsonl(ROOT/'data/t3_judgments.jsonl')}
    chunks = json.loads((ROOT/'data/frozen_chunks.json').read_text(encoding='utf-8'))
    by_id = {d['metadata']['chunk_id']: d for d in chunks}
    docs = [SimpleNamespace(page_content=d['page_content'], metadata=d['metadata']) for d in chunks]
    bm25 = BM25(docs)
    with urlopen(ENDPOINT+'/api/tags', timeout=10) as r:
        model = next(m for m in json.load(r)['models'] if m['name'] == MODEL)
    binding = {'target': 'T3 BM25 endpoint (rrf_0, alpha=0)',
               'qa_sha256': sha(ROOT/'data/eval/qa_set.jsonl'),
               'answers_sha256': sha(ROOT/'data/system_answers.jsonl'),
               'chunks_sha256': sha(ROOT/'data/frozen_chunks.json'),
               'correctness_scores_sha256': sha(ROOT/'data/t3_judgments.jsonl'),
               'judge_model_digest': model['digest'], 'faithfulness_protocol': 't7-claims-v3-compact-sentence-ids',
               'judge_context_tokens': {'default':12288,'long_output_retry':16384},
               'ks': [1,3,5,10], 'candidate_pool_for_diagnosis': 20}
    manifest = OUT/'run_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))['binding'] == binding, 'Frozen T7 inputs changed'
    else:
        original = json.loads((ROOT/'data/t3_run_manifest.json').read_text(encoding='utf-8'))
        manifest.write_text(json.dumps({'created_at': datetime.now(timezone.utc).isoformat(), 'binding': binding,
            'evaluated_generation_binding': original['binding'],
            'target_selection': 'User explicitly selected the local T3 BM25 scheme.',
            'notes': ['20 saved real answers and their actual contexts are evaluated offline; no generation changes.',
                      'Correctness reuses frozen T3 blinded 0-5 judgments; exact identical texts share scores.',
                      'Faithfulness judge sees actual retrieved blocks, question and answer only, never gold_answer.',
                      'Hypothetical givens and pure advice/refusal/headings are excluded from factual denominator.',
                      'Supported requires score>=4 and a real retrieved evidence-sentence ID; quotes are restored by code.',
                      'Partial support counts unsupported under the strict course hallucination definition.',
                      'No factual claims => faithfulness and hallucination rate are null, not perfect scores.',
                      'MRR/MAP full use the full BM25 ranking; @10 variants are also reported.',
                      'MAP denominator is total marked gold chunks, including those missed by the cutoff.',
                      'Gold IDs do not exhaust equivalent overlapping chunks; failures need text-level review.',
                      'Same-model, assistant-assisted evaluation is not independent human validation.']},
            ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = {r['id']:r for r in read_jsonl(OUT/'per_question_metrics.jsonl')}
    faith = {r['id']:r for r in read_jsonl(OUT/'faithfulness.jsonl')}
    llm = EvidenceJudge(OUT/'cache', model['digest'])
    for q in qs:
        qid = q['id']
        answer = answers[qid]
        assert answer['method'] == 'rrf_0'
        ranking = bm25.rank(q['question'], limit=len(chunks))
        ids = [r['chunk_id'] for r in ranking]
        assert ids[:5] == answer['top_chunk_ids'], f'BM25 ranking drift for {qid}'
        retrieved = [by_id[cid] for cid in ids[:5]]
        expected_context = '\n\n'.join(f'[{i}] {format_document(d)}' for i,d in enumerate(retrieved,1))
        assert expected_context == answer['context'], f'Actual generation context drift for {qid}'
        assert set(q['gold_chunk_ids']).issubset(by_id)
        if args.stage in ['all','metrics'] and qid not in metrics:
            record = {'id': qid, 'level': q['level'], 'question': q['question'], 'gold_chunk_ids': q['gold_chunk_ids'],
                      'ranking': ranking, 'actual_top5': ids[:5], 'gold_ranks': {g: ids.index(g)+1 if g in ids else None for g in q['gold_chunk_ids']},
                      **rank_metrics(ids,q['gold_chunk_ids']),
                      'correctness_0_5': t3_scores[qid]['scores']['rrf_0']['score'],
                      'correctness_reason': t3_scores[qid]['scores']['rrf_0']['reason'],
                      'rouge_l_char': answer['rouge_l_char'],
                      'exact_match': int(normalize(answer['answer']) == normalize(q['gold_answer'])),
                      'generation_truncated': answer['llm']['truncated'],
                      'generation_prompt_tokens': answer['llm']['prompt_tokens'],
                      'generation_output_tokens': answer['llm']['output_tokens'],
                      'generation_recorded_seconds': answer['llm']['elapsed_seconds'],
                      'generation_was_cached': answer['llm']['cache_hit'],
                      'citation_numbers_valid': answer['citation_numbers_valid']}
            append_jsonl(OUT/'per_question_metrics.jsonl', record)
            metrics[qid] = record
            log(f'METRICS {qid}')
        if args.stage in ['all','faithfulness'] and qid not in faith:
            log(f'FACT_CHECK {qid} sentences={len(__import__("t7_core").answer_units(answer["answer"]))}')
            result = judge_faithfulness(llm, q, answer['answer'], retrieved)
            append_jsonl(OUT/'faithfulness.jsonl', result)
            faith[qid] = result
            log(f'FACT_CHECK_DONE {qid} facts={result["fact_count"]} unsupported={result["unsupported_fact_count"]} flags={result["verification_flags"]}')
    log(f'COMPLETE metrics={len(metrics)} faithfulness={len(faith)}')


if __name__ == '__main__':
    main()
