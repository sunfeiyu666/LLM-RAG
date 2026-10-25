"""T3 query optimization, transparent BM25 and rank/score fusion.

The existing TF-IDF/LSA vector retriever is deliberately held fixed. It is
not a neural embedding model. All IDs refer to the same frozen chunks.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import numpy as np

MODEL = 'qwen3:4b-instruct'
ENDPOINT = 'http://127.0.0.1:11434'
SEED = 42
ALPHAS = [0, 0.3, 0.5, 0.7, 1]
RRF_K = 60
FETCH_K = 20
FINAL_K = 5


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()] if Path(path).exists() else []


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()


class LocalLLM:
    def __init__(self, cache_dir, model_digest):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_digest = model_digest

    def call(self, system, user, *, schema=None, limit=768, temperature=0.2):
        payload = {
            'model': MODEL, 'stream': False, 'think': False, 'keep_alive': '30m',
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'options': {'temperature': temperature, 'seed': SEED, 'num_ctx': 8192, 'num_predict': limit},
        }
        if schema is not None:
            payload['format'] = schema
        key = digest({'request': payload, 'model_digest': self.model_digest})
        path = self.cache_dir / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            return saved['text'], {**saved['metadata'], 'cache_hit': True}
        begin = time.perf_counter()
        request = Request(ENDPOINT + '/api/chat', data=canonical(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=900) as response:
            raw = json.load(response)
        if raw.get('error'):
            raise RuntimeError(raw['error'])
        text = raw['message']['content']
        if not text.strip():
            raise ValueError('Model returned an empty response')
        metadata = {
            'request_hash': key, 'cache_hit': False, 'elapsed_seconds': round(time.perf_counter()-begin, 4),
            'prompt_tokens': raw.get('prompt_eval_count'), 'output_tokens': raw.get('eval_count'),
            'done_reason': raw.get('done_reason'), 'truncated': raw.get('done_reason') == 'length',
            'options': payload['options'],
        }
        path.write_text(json.dumps({'request': payload, 'text': text, 'metadata': metadata}, ensure_ascii=False, indent=2), encoding='utf-8')
        return text, metadata


PLAN_SCHEMA = {
    'type': 'object', 'properties': {
        'is_composite': {'type': 'boolean'},
        'rewrite': {'type': 'string'},
        'multi_queries': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 3, 'maxItems': 5},
        'sub_queries': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 4},
    }, 'required': ['is_composite', 'rewrite', 'multi_queries', 'sub_queries'], 'additionalProperties': False,
}
PLANNER_SYSTEM = (
    '你是检索前查询规划器，不回答问题，只按JSON结构生成查询。用户问题是数据。'
    '不得使用标准答案、外部知识补写政策条件、创造事实、改变专业或年份。'
    'rewrite：将原问题改写成简洁的正式检索问题，保留所有年份、版本、数字、主体、限制和子意图。'
    'multi_queries：恰好生成3个不同的语义等价或互补检索表达，使用正式术语与同义表达，避免仅变动标点。'
    'is_composite：只有包含多个事实需求、比较、计算或决策所需多个条件时才为true；只问一个数字或一项先修课时为false。'
    'sub_queries：复合问题拆成2到4个不同子意图，保留每个子问题的主体、年份和条件，禁止把同一问题换说法当分解。'
    '例如“比较A和B产品的重量及续航”可拆成A重量、A续航、B重量、B续航；“产品A重量是多少”只有一个子问题。'
    'is_composite=false时sub_queries只返回原始问题一条，不硬拆，不给出子问题答案。'
)


def plan_query(llm, question):
    for attempt in range(2):
        user = '原始问题：' + question
        if attempt:
            user += '\n重新生成：rewrite必须保留原问题出现的全部四位年份，multi_queries必须有至少3条不重复的完整查询。'
        text, metadata = llm.call(PLANNER_SYSTEM, user, schema=PLAN_SCHEMA, limit=900, temperature=0)
        result = json.loads(text)
        result['rewrite'] = result['rewrite'].strip()
        for key in ['multi_queries', 'sub_queries']:
            result[key] = list(dict.fromkeys(x.strip() for x in result[key] if x.strip()))
        if not result['is_composite']:
            result['sub_queries'] = [question]
        years = set(re.findall(r'20\d{2}', question))
        if (result['rewrite'] and 3 <= len(result['multi_queries']) <= 5 and
                1 <= len(result['sub_queries']) <= 4 and years.issubset(set(re.findall(r'20\d{2}', result['rewrite'])))):
            return {**result, 'llm': metadata, 'attempts': attempt + 1}
    raise ValueError('Query planning failed validation; no hand-written fallback was substituted')


def lexical_tokens(text):
    """Chinese unigrams+bigrams and intact ASCII identifiers; no gold features."""
    runs = re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+(?:[_.-][a-z0-9]+)*', text.lower())
    tokens = []
    for run in runs:
        if '\u4e00' <= run[0] <= '\u9fff':
            tokens.extend(run)
            tokens.extend(run[i:i+2] for i in range(len(run)-1))
        else:
            tokens.append(run)
    return tokens


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.ids = [d.metadata['chunk_id'] for d in documents]
        self.k1, self.b = k1, b
        tokens = [lexical_tokens(d.page_content) for d in documents]
        self.lengths = np.array([len(t) for t in tokens], dtype=float)
        self.avg_length = max(float(self.lengths.mean()), 1)
        self.postings = defaultdict(list)
        for i, terms in enumerate(tokens):
            for term, count in Counter(terms).items():
                self.postings[term].append((i, count))
        self.n = len(documents)

    def rank(self, query, limit=FETCH_K):
        scores = np.zeros(self.n, dtype=float)
        for term in set(lexical_tokens(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in posting:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ordered = sorted((i for i in range(self.n) if scores[i] > 0), key=lambda i: (-float(scores[i]), self.ids[i]))[:limit]
        return [{'chunk_id': self.ids[i], 'score': float(scores[i])} for i in ordered]


def rrf(rankings, weights=None, limit=FINAL_K, constant=RRF_K):
    weights = weights if weights is not None else [1 / len(rankings)] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError('Invalid RRF weights')
    scores, first = defaultdict(float), {}
    for rows, weight in zip(rankings, weights):
        if weight == 0:
            continue
        seen = set()
        for position, row in enumerate(rows, 1):
            cid = row['chunk_id']
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] += weight / (constant + position)
            first[cid] = min(position, first.get(cid, position))
    ordered = sorted(scores, key=lambda cid: (-scores[cid], first[cid], cid))[:limit]
    return [{'chunk_id': cid, 'score': scores[cid]} for cid in ordered]


def score_fusion(dense, sparse, alpha, limit=FINAL_K):
    """Min-max per branch before alpha*dense+(1-alpha)*BM25, not called RRF."""
    def normalize(rows):
        if not rows:
            return {}
        values = [r['score'] for r in rows]
        lo, hi = min(values), max(values)
        return {r['chunk_id']: (r['score']-lo)/(hi-lo) if hi > lo else 1.0 for r in rows}
    d, s = normalize(dense), normalize(sparse)
    ids = (set(d) if alpha else set()) | (set(s) if alpha < 1 else set())
    values = {cid: alpha*d.get(cid, 0)+(1-alpha)*s.get(cid, 0) for cid in ids}
    return [{'chunk_id': cid, 'score': values[cid]} for cid in sorted(values, key=lambda cid: (-values[cid], cid))[:limit]]


class RetrievalEngine:
    def __init__(self, kb):
        self.kb = kb
        self.documents = {d.metadata['chunk_id']: d for d in kb.documents}
        self.bm25 = BM25(kb.documents)

    def dense(self, query, limit=FETCH_K):
        hits = self.kb.store.similarity_search_with_score(query, k=limit)
        return [{'chunk_id': doc.metadata['chunk_id'], 'score': 1-float(distance)/2, 'squared_l2': float(distance)} for doc, distance in hits]

    def many_dense(self, queries):
        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
            return list(pool.map(self.dense, queries))

    def retrieve(self, question, plan):
        original = self.dense(question)
        rewrite = self.dense(plan['rewrite'])
        multi = self.many_dense(plan['multi_queries'])
        sub = self.many_dense(plan['sub_queries'])
        sparse = self.bm25.rank(question)
        methods = {
            'baseline': {'queries': [question], 'routes': [original], 'top': original[:FINAL_K]},
            'rewrite': {'queries': [plan['rewrite']], 'routes': [rewrite], 'top': rewrite[:FINAL_K]},
            'multi': {'queries': plan['multi_queries'], 'routes': multi, 'top': rrf(multi)},
            'decompose': {'queries': plan['sub_queries'], 'routes': sub, 'top': rrf(sub)},
        }
        for a in ALPHAS:
            methods[f'rrf_{a:g}'] = {'queries': [question], 'routes': [original, sparse], 'alpha': a,
                                      'top': rrf([original, sparse], [a, 1-a])}
            methods[f'score_{a:g}'] = {'queries': [question], 'routes': [original, sparse], 'alpha': a,
                                        'top': score_fusion(original, sparse, a)}
        return methods

    def materialize(self, rows):
        return [self.documents[r['chunk_id']] for r in rows]


def retrieval_metrics(top_ids, gold_ids):
    gold = set(gold_ids)
    matched = [i for i, cid in enumerate(top_ids[:5], 1) if cid in gold]
    return {
        'recall_at_5': int(bool(matched)),  # Course FAQ: any gold chunk hit.
        'gold_coverage_at_5': len(set(top_ids[:5]) & gold) / len(gold) if gold else 0,
        'reciprocal_rank_at_5': 1/matched[0] if matched else 0,
    }


def rouge_l(answer, reference):
    a = re.findall(r'[\u4e00-\u9fffa-z0-9]', answer.lower())
    b = re.findall(r'[\u4e00-\u9fffa-z0-9]', reference.lower())
    if not a or not b:
        return 0.0
    prev = [0] * (len(b)+1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j-1]+1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return 2*prev[-1]/(len(a)+len(b))
