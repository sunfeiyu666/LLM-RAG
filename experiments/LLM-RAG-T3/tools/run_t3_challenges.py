"""Predeclared colloquial paraphrases, separate from the frozen 20 questions."""
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kb import load_index
from t3_core import LocalLLM, RetrievalEngine, plan_query, retrieval_metrics, read_jsonl, append_jsonl, ENDPOINT, MODEL

PARAPHRASES = [
    ('c01', 'q014', '2026秋新开的微专，最后就选中了十四个人，这班还办不办？这回的选拔还算不算数，下回要不要重报？'),
    ('c02', 'q006', '我是2025级学生，按2026年的规定，换专业前学过的其他课，和后来读辅修、微专业的课，各能往现在主专业的选修里抵多少？后者多几个学分？'),
    ('c03', 'q007', '云大2026秋那次，读辅修和读微专分别啥时候把申请交上去？是不是一拨时间办？'),
    ('c04', 'q011', '按云大2027学校那套推免办法，四年制软工应届生前三年绩点二点九，别的都够，只看绩点够线没？说下怎么算的。'),
]


def main():
    out = ROOT / 'results/t3'
    original = {q['id']: q for q in read_jsonl(ROOT / 'data/eval/qa_set.jsonl')}
    declared = []
    for cid, source, text in PARAPHRASES:
        q = original[source]
        declared.append({**q, 'id': cid, 'question': text, 'source_question_id': source,
                         'supplemental': True, 'purpose': 'Colloquial diagnostic paraphrase, not part of the primary n=20 benchmark.'})
    frozen = ROOT / 'data/eval/t3_colloquial_challenges.jsonl'
    encoded = ''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in declared)
    if frozen.exists() and frozen.read_text(encoding='utf-8') != encoded:
        raise ValueError('Declared challenge set changed')
    frozen.write_text(encoded, encoding='utf-8')
    with urlopen(ENDPOINT + '/api/tags', timeout=10) as r:
        info = next(m for m in json.load(r)['models'] if m['name'] == MODEL)
    llm = LocalLLM(out / 'cache', info['digest'])
    engine = RetrievalEngine(load_index())
    done = {q['id'] for q in read_jsonl(out / 'challenge_retrieval.jsonl')}
    for q in declared:
        if q['id'] in done:
            continue
        print('CHALLENGE', q['id'], flush=True)
        plan = plan_query(llm, q['question'])
        methods = engine.retrieve(q['question'], plan)
        for method, data in methods.items():
            data['metrics'] = retrieval_metrics([x['chunk_id'] for x in data['top']], q['gold_chunk_ids'])
        record = {'id': q['id'], 'source_question_id': q['source_question_id'], 'level': q['level'],
                  'question': q['question'], 'plan': plan, 'methods': methods}
        append_jsonl(out / 'challenge_retrieval.jsonl', record)
        print('CHALLENGE_DONE', q['id'], {m: methods[m]['metrics'] for m in ['baseline', 'rewrite', 'multi', 'decompose']}, flush=True)


if __name__ == '__main__':
    main()
