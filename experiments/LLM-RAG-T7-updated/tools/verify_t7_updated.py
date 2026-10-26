"""Read-only audit of frozen T7 inputs and results; writes only verification outputs.

No answer generation, rescoring, chart creation, network access, or data repair.
Run: python tools/verify_t7_updated.py
"""
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import traceback
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from t7_updated_core import aggregate, rank_metrics, rouge_l, validate_review

OUT = ROOT / 'results/t7'
UP = ROOT / 'upstream/t4_t5'
EXPECTED_ANSWERS_SHA = '33b470750909d827302edb0ad8f3a24994017fdb9ebe640296ec23c15b36d156'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows):
    result = {r['id']: r for r in rows}
    require(len(rows) == len(result), 'Duplicate question IDs')
    return result


def near(a, b):
    if a is None or b is None:
        require(a is b, f'None mismatch: {a} != {b}')
    else:
        require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), f'Numeric mismatch: {a} != {b}')


def reference_metrics(ranking, gold):
    """Separate cumulative-hit implementation; AP denominator includes missed gold."""
    relevant = set(gold)
    hits = [i + 1 for i, chunk in enumerate(ranking) if chunk in relevant]
    result = {}
    for k in (1, 3, 5, 10):
        count = sum(p <= k for p in hits)
        result[f'recall_at_{k}'] = float(count > 0)
        result[f'precision_at_{k}'] = count / k
        result[f'gold_coverage_at_{k}'] = count / len(relevant)
    for cutoff, name in ((10, 'at_10'), (len(ranking), 'full')):
        count = 0
        precisions = []
        first = None
        for position, chunk in enumerate(ranking[:cutoff], 1):
            if chunk in relevant:
                count += 1
                first = first or position
                precisions.append(count / position)
        result[f'mrr_{name}'] = 1 / first if first else 0
        result[f'map_{name}'] = sum(precisions) / len(relevant)
    return result


def reference_rouge(answer, gold):
    # Bitset LCS independently checks the production dynamic-programming routine.
    a = ''.join(re.findall(r'[\u4e00-\u9fffa-z0-9]', answer.lower()))
    b = ''.join(re.findall(r'[\u4e00-\u9fffa-z0-9]', gold.lower()))
    masks = {}
    for i, char in enumerate(b):
        masks[char] = masks.get(char, 0) | (1 << i)
    state = 0
    for char in a:
        x = state | masks.get(char, 0)
        state = x & ~(x - ((state << 1) | 1))
    return 2 * state.bit_count() / (len(a) + len(b)) if a and b else 0.0


def check_review(review, packet):
    """Structural/evidence audit, not an independent semantic assessment."""
    require(review['id'] == packet['id'] and review['level'] == packet['level'], 'Review identity mismatch')
    digest = hashlib.sha256(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    require(digest == review['packet_sha256'], 'Review packet hash mismatch')
    units = {u['id']: u['text'] for u in packet['answer_units']}
    require(len(units) == len(packet['answer_units']), 'Duplicate answer units')
    used = {c['unit_id'] for c in review['claims']}
    excluded = {u['id'] for u in review['excluded_units']}
    require(len(excluded) == len(review['excluded_units']), 'Duplicate excluded units')
    require(used | excluded == set(units) and not used & excluded, 'Unclassified or overlapping units')
    require(review['classified_unit_ids'] == list(units), 'Classified unit index mismatch')
    docs = {d['chunk_id']: d for d in packet['actual_documents']}
    claim_ids = [c['id'] for c in review['claims']]
    require(len(claim_ids) == len(set(claim_ids)), 'Duplicate claims')
    for c in review['claims']:
        require(c['unit_text'] == units[c['unit_id']], 'Claim answer text changed')
        require(type(c['score']) is int and 1 <= c['score'] <= 5, 'Invalid score')
        require(c['supported'] is (c['score'] >= 4), 'Supported flag does not match score')
        require(bool(c['claim']) and bool(c['reason']), 'Missing claim/reason')
        require(c['verdict'] == {1:'contradicted', 2:'unsupported', 3:'partial', 4:'supported_inference', 5:'directly_supported'}[c['score']], 'Verdict mismatch')
        require(not c['supported'] or bool(c['evidence']), 'Supported claim lacks evidence')
        for e in c['evidence']:
            require(e['chunk_id'] in docs, 'Evidence outside actual Context')
            d = docs[e['chunk_id']]
            require(e['block'] == d['block'], 'Evidence block number mismatch')
            require(bool(e['quote']) and e['quote'] in d['formatted_text'], 'Quote not literal in actual Context')
            require(e['quote_verified'] is True, 'Unverified evidence flag')
    for e in review['excluded_units']:
        require(e['text'] == units[e['id']] and bool(e['reason']), 'Excluded unit text/reason mismatch')
    n = len(review['claims'])
    unsupported = sum(c['score'] < 4 for c in review['claims'])
    require((n, unsupported) == (review['fact_count'], review['unsupported_fact_count']), 'Fact count mismatch')
    near(review['faithfulness_1_5'], statistics.fmean(c['score'] for c in review['claims']) if n else None)
    near(review['hallucination_rate'], unsupported / n if n else None)
    validate_review(review, packet)


def main():
    checks = []
    inputs = {}
    def run(name, action):
        try:
            detail = action()
            checks.append({'check': name, 'passed': True, 'detail': detail})
        except Exception as exc:
            checks.append({'check': name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}',
                           'traceback': traceback.format_exc()})

    def sources():
        manifest = read(ROOT / 'upstream_manifest.json')
        require(len(manifest) == 88, 'Expected 88 pinned upstream files')
        require(len({r['path'] for r in manifest}) == len(manifest), 'Duplicate snapshot paths')
        target = read(ROOT / 'target_manifest.json')
        for r in manifest:
            p = (ROOT / r['path']).resolve()
            require(p.is_relative_to((ROOT / 'upstream').resolve()), 'Snapshot path escapes upstream')
            require(sha(p) == r['sha256'], f'Changed snapshot: {r["path"]}')
            expected = target['t4_t5_commit'] if r['ref'] == 'origin/t4-t5-zhuxiangduo' else target['main_t2_commit']
            require(r['commit'] == expected, 'Snapshot commit mismatch')
        require(sha(ROOT / 'data/qa_set.jsonl') == target['qa_sha256'] == sha(UP / 'data/eval/qa_set.jsonl'), 'QA version mismatch')
        require(sha(UP / 'results/t5_final_20261026/t5_full_rows.jsonl') == target['upstream_answers_sha256'], 'Source answers hash mismatch')
        require(sha(ROOT / 'data/system_answers.jsonl') == EXPECTED_ANSWERS_SHA, 'Answers changed since source-only review')
        meta = read(UP / 'results/t5_final_20261026/run_metadata.json')
        require(meta['model_digest'] == target['model_digest'] and meta['generation_parameters'] == target['generation_parameters'], 'Model/configuration mismatch')
        return {'snapshot_files': len(manifest), 'qa_sha256': target['qa_sha256'], 'answers_sha256': EXPECTED_ANSWERS_SHA}
    run('版本与88份源文件完整性', sources)

    def identity():
        for key, path in [('qa','data/qa_set.jsonl'), ('answers','data/system_answers.jsonl'),
                          ('metrics','results/t7/retrieval_metrics.jsonl'), ('packets','results/t7/faithfulness_packets.jsonl'),
                          ('reviews','results/t7/faithfulness_reviewed.jsonl')]:
            inputs[key] = indexed(lines(ROOT / path))
        q = inputs['qa']; ids = set(q)
        require(len(q) == 20 and Counter(r['level'] for r in q.values()) == Counter({'L1':5,'L2':5,'L3':5,'L4':5}), '20 questions / level balance mismatch')
        for key, rows in inputs.items():
            require(set(rows) == ids, f'Question set mismatch: {key}')
            require(all(rows[i]['level'] == q[i]['level'] for i in ids), f'Level mismatch: {key}')
        raw = lines(UP / 'results/t5_final_20261026/t5_full_rows.jsonl')
        selected = indexed([r for r in raw if r['strategy'] == 'direct' and r['compressed'] is False])
        require(len(raw) == 120 and set(selected) == ids, 'T5 selected configuration mismatch')
        with (UP / 'results/t5_final_20261026/t5_answer_review.csv').open(encoding='utf-8-sig', newline='') as f:
            review = indexed([r for r in csv.DictReader(f) if r['strategy'] == 'direct' and r['compressed'] == 'False'])
        for i in q:
            a = inputs['answers'][i]; p = inputs['packets'][i]; m = inputs['metrics'][i]
            require(all(a[k] == v for k, v in selected[i].items()), f'Saved answer/source row mismatch {i}')
            require(a['question'] == q[i]['question'] == p['question'] == m['question'], f'Question mismatch {i}')
            require(a['gold_answer'] == q[i]['gold_answer'] and a['gold_chunk_ids'] == q[i]['gold_chunk_ids'], f'Gold mismatch {i}')
            require(a['answer'] == p['answer'] == review[i]['answer'], f'Answer mismatch {i}')
            require('gold_answer' not in p and 'gold_chunk_ids' not in p, 'Gold leaked into source-only review packet')
            from t7_updated_core import answer_units
            require(p['answer_units'] == answer_units(a['answer']), f'Answer units mismatch {i}')
            near(m['upstream_accuracy_0_2'], int(review[i]['accuracy_0_2']))
            near(m['upstream_citation_0_2'], int(review[i]['citation_support_0_2']))
            require(m['upstream_review_note'] == review[i]['notes'], 'Upstream review note mismatch')
        return {'questions':20,'source_rows':120,'selected_configuration':'direct / compressed=False'}
    run('20题、等级、原始回答及评分对应', identity)

    def retrieval_context():
        sys.path.insert(0, str(UP / 'src'))
        from kb import load_pages, split_documents, format_document
        from rag_chain import format_context
        from retrieval_compare import BM25
        docs = split_documents(load_pages())
        frozen = read(ROOT / 'data/frozen_chunks.json')
        require(len(docs) == len(frozen) == 202, 'Chunk count mismatch')
        for d, old in zip(docs, frozen):
            require(d.page_content == old['page_content'], 'Frozen chunk text mismatch')
            require({k:v for k,v in d.metadata.items() if k != 'source'} == {k:v for k,v in old['metadata'].items() if k != 'source'}, 'Chunk metadata mismatch')
        corpus = {d.metadata['chunk_id']: d for d in docs}
        require(len(corpus) == 202, 'Duplicate chunk IDs')
        inputs['corpus'] = corpus
        bm25 = BM25(docs)
        rankings = lines(UP / 'results/t4_bge_20261025/t4_rankings.jsonl')
        require(len(rankings) == 60 and len({(r['method'],r['id']) for r in rankings}) == 60, 'T4 ranking count/duplicates')
        t4bm = {r['id']:r for r in rankings if r['method'] == 'bm25'}
        for i,q in inputs['qa'].items():
            m = inputs['metrics'][i]; a = inputs['answers'][i]; p = inputs['packets'][i]
            order = [d.metadata['chunk_id'] for d in bm25.search(q['question'],202)]
            require(set(order) == set(corpus) and len(order) == 202, f'Incomplete ranking {i}')
            require(m['ranking'] == order and order[:10] == t4bm[i]['retrieved_chunk_ids'], f'BM25/source ranking mismatch {i}')
            require(m['candidate_top5'] == order[:5] and m['actual_top3'] == a['selected_chunk_ids'] == order[:3], f'Actual retrieval selection mismatch {i}')
            require(set(q['gold_chunk_ids']) <= set(corpus), f'Unknown gold chunks {i}')
            require(m['gold_ranks'] == {g:order.index(g)+1 for g in q['gold_chunk_ids']}, f'Gold ranks mismatch {i}')
            selected = [corpus[c] for c in order[:3]]
            require(a['context'] == format_context(selected), f'Reconstructed Context mismatch {i}')
            require(a['context_sha256'] == hashlib.sha256(a['context'].encode()).hexdigest(), f'Context SHA mismatch {i}')
            require(a['context_chars'] == sum(len(d.page_content) for d in selected), f'Upstream body character count mismatch {i}')
            expected = [{'block':j,'chunk_id':d.metadata['chunk_id'],'formatted_text':format_document(d)} for j,d in enumerate(selected,1)]
            require(p['actual_documents'] == expected, f'Review Context mismatch {i}')
        with (UP / 'results/t4_bge_20261025/t4_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            for saved in csv.DictReader(f):
                group = [r for r in rankings if r['method'] == saved['method']]
                require(len(group) == 20 and {r['id'] for r in group} == set(inputs['qa']), 'T4 method question set mismatch')
                values = [reference_metrics(r['retrieved_chunk_ids'],inputs['qa'][r['id']]['gold_chunk_ids']) for r in group]
                for key, field in [('recall_at_1','recall@1'),('recall_at_3','recall@3'),('recall_at_5','recall@5'),('recall_at_10','recall@10'),('mrr_at_10','mrr@10')]:
                    near(statistics.fmean(v[key] for v in values),float(saved[field]))
        return {'reconstructed_chunks':202,'BM25_full_rankings':20,'Context_checks':20,'T4_saved_rankings_checked':60}
    run('重新分块、BM25排名及实际Context', retrieval_context)

    def metrics():
        for i,q in inputs['qa'].items():
            m = inputs['metrics'][i]; a = inputs['answers'][i]
            reference = reference_metrics(m['ranking'],q['gold_chunk_ids'])
            production = rank_metrics(m['ranking'],q['gold_chunk_ids'])
            for key,value in reference.items():
                near(m[key],value); near(production[key],value)
            expected = reference_rouge(a['answer'],q['gold_answer'])
            near(expected,m['rouge_l_char']); near(expected,rouge_l(a['answer'],q['gold_answer']))
            norm = lambda s: ''.join(c for c in s.lower() if not c.isspace() and c not in '*`_#')
            near(m['exact_match'],int(norm(a['answer']) == norm(q['gold_answer'])))
            for key in ('context_tokens','generation_output_tokens','total_seconds'):
                near(m[key],a[key])
            require(m['reported_output_at_limit'] == (a['generation_output_tokens'] >= 2048), 'Output-limit flag mismatch')
        return {'questions':20,'independent_checks':'Cumulative ranking metrics, bitset LCS ROUGE-L, normalized EM; timing/token values only checked against source'}
    run('独立复算检索指标、ROUGE-L和EM', metrics)

    def faithfulness():
        flattened = []
        for i,r in inputs['reviews'].items():
            check_review(r,inputs['packets'][i])
            flattened += [{'question_id':i,'level':r['level'],**c} for c in r['claims']]
        require(flattened == lines(OUT / 'claim_evaluations.jsonl'), 'Flattened claims differ')
        n = sum(r['fact_count'] for r in inputs['reviews'].values())
        u = sum(r['unsupported_fact_count'] for r in inputs['reviews'].values())
        require((n,u) == (213,50), 'Published review totals changed')
        return {'reviews':20,'facts':n,'unsupported':u,'boundary':'Literal evidence/structure verified; semantic scores remain assistant judgments'}
    run('逐事实评分、引文与实际前三片段边界', faithfulness)

    def summaries():
        expected = []
        for i,m in inputs['metrics'].items():
            f = inputs['reviews'][i]
            expected.append({**m,**{k:f[k] for k in ('fact_count','unsupported_fact_count','faithfulness_1_5','hallucination_rate')}})
        require(expected == read(OUT / 'per_question.json'), 'Per-question table differs from underlying results')
        s = read(OUT / 'summary.json')
        for group, saved in [(expected,s['overall'])] + [([r for r in expected if r['level']==l],s['by_level'][l]) for l in ('L1','L2','L3','L4')]:
            require(aggregate(group) == saved, 'Production aggregate mismatch')
            for key,value in saved.items():
                if key in ('n','fact_count','unsupported_fact_count','hallucination_micro','hallucination_macro','no_fact_answers'):
                    continue
                vals = [r[key] for r in group if r[key] is not None]
                near(value,sum(vals)/len(vals) if vals else None)
            n = sum(r['fact_count'] for r in group); u = sum(r['unsupported_fact_count'] for r in group)
            near(saved['hallucination_micro'],u/n if n else None)
            require(saved['fact_count'] == n and saved['unsupported_fact_count'] == u and saved['n'] == len(group), 'Aggregate counts mismatch')
        with (ROOT / 'T7完整评估表_更新版.csv').open(encoding='utf-8-sig',newline='') as f:
            rows = indexed(list(csv.DictReader(f)))
        require(set(rows) == set(inputs['qa']), 'CSV question set mismatch')
        for r in expected:
            for k,value in rows[r['id']].items():
                if k in ('id','level'):
                    require(value == r[k], 'CSV identity mismatch')
                else:
                    near(float(value),r[k])
        inputs['merged'] = expected
        return {'levels':4,'CSV_rows':20,'all_summary_metrics':'match per-question source and independent averages'}
    run('完整评估表、总体与L1–L4汇总', summaries)

    def failures():
        worst = sorted(inputs['metrics'].values(),key=lambda r:(r['upstream_accuracy_0_2'],r['upstream_citation_0_2'],r['rouge_l_char'],r['id']))[:5]
        require(worst == read(OUT / 'worst_five_preselected.json'), 'Preselected worst-five data mismatch')
        cases = read(OUT / 'failure_reviews.json')
        require([r['id'] for r in worst] == [r['id'] for r in cases], 'Wrong worst-five selection')
        for c in cases:
            i=c['id']; m=inputs['metrics'][i]
            require(bool(c['answer_excerpt']) and c['answer_excerpt'] in inputs['answers'][i]['answer'], 'Failure excerpt not literal')
            require(c['primary_cause'] in ('检索失败','排序失败','生成失败','评测集本身有歧义'), 'Unknown failure type')
            require(all(c[k] for k in ('analysis','improvement','verification_plan')), 'Missing failure analysis')
            for e in c['evidence']:
                require(e['quote'] in inputs['corpus'][e['chunk_id']].page_content, 'Failure evidence not literal corpus text')
                require(e['rank'] == m['ranking'].index(e['chunk_id'])+1, 'Failure rank wrong')
                require(e['in_candidate_top5'] == (e['chunk_id'] in m['candidate_top5']), 'Candidate boundary wrong')
                require(e['in_actual_top3'] == (e['chunk_id'] in m['actual_top3']), 'Actual Context boundary wrong')
        return {'worst_ids':[r['id'] for r in cases],'case_evidence':'literal excerpts, ranks, candidate/Context boundaries verified'}
    run('最差五题选择及案例证据', failures)

    def reports():
        # Verify existing drafts only. Do not rewrite reports or generate/review charts.
        report = (ROOT / 'T7系统评估报告_更新版.md').read_text(encoding='utf-8')
        table = (ROOT / 'T7完整评估表_更新版.md').read_text(encoding='utf-8')
        cases = (ROOT / 'T7失败案例分析_更新版.md').read_text(encoding='utf-8')
        s = read(OUT / 'summary.json'); o=s['overall']
        tokens = [f"MRR={o['mrr_full']:.3f}",f"MAP={o['map_full']:.3f}",f"忠实度{o['faithfulness_1_5']:.3f}/5",
                  f"{o['unsupported_fact_count']}/{o['fact_count']}",f"{o['hallucination_micro']:.2%}"]
        for token in tokens:
            require(token in report, f'Report headline disagrees with results: {token}')
        def row(values): return '| '+' | '.join(str(v) for v in values)+' |'
        for r in inputs['merged']:
            expected=row([r['id'],r['level'],f"{r['recall_at_3']:.2%}",f"{r['precision_at_3']:.3f}",f"{r['map_at_10']:.3f}",f"{r['rouge_l_char']:.3f}",r['upstream_accuracy_0_2'],f"{r['faithfulness_1_5']:.3f}",f"{r['unsupported_fact_count']}/{r['fact_count']}",f"{r['hallucination_rate']:.2%}"])
            require(expected in table, f'Markdown question table mismatch {r["id"]}')
        for level,v in s['by_level'].items():
            require(row([level,v['n'],f"{v['upstream_accuracy_0_2']:.3f}",f"{v['rouge_l_char']:.3f}",f"{v['faithfulness_1_5']:.3f}",f"{v['unsupported_fact_count']}/{v['fact_count']}",f"{v['hallucination_micro']:.2%}"]) in report, f'Report level table mismatch {level}')
        for c in read(OUT / 'failure_reviews.json'):
            require(c['id'] in cases and inputs['answers'][c['id']]['answer'] in cases and c['analysis'] in cases, 'Case report missing original answer/analysis')
        target = read(ROOT / 'target_manifest.json')
        for key in ('t4_t5_commit','main_t2_commit','qa_sha256','model_digest'):
            require(target[key] in report, f'Report version mismatch {key}')
        for token in ('generated_answers=false','retrieval_evidence_proxy','不冒称盲评','未直接保存完整Context','未额外调用本地Qwen评分器'):
            require(token in report, f'Missing interpretation boundary {token}')
        return {'existing_draft_tables':'20 question rows, four levels, headline metrics, versions and five case answers match', 'charts':'not audited; unchanged'}
    run('现有报告与Markdown表的数字和版本一致性', reports)

    def negative_tests():
        suite = unittest.defaultTestLoader.discover(str(ROOT/'src'), pattern='test_t7_validation.py')
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        require(result.testsRun == 10 and result.wasSuccessful(), stream.getvalue())
        return {'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
                'log':stream.getvalue(),'checks':'Reject changed packets, outside-Context evidence, fabricated quotes, wrong block numbers, score/flag mismatches and omitted units; verify metric boundaries and no-fact N/A.'}
    run('校验程序拦错与指标边界测试', negative_tests)

    failures_count = sum(not c['passed'] for c in checks)
    result = {'checked_at':datetime.now(timezone.utc).isoformat(),'passed':failures_count==0,
              'checks_passed':len(checks)-failures_count,'checks_failed':failures_count,'checks':checks,
              'scope':'Data/version, retrieval/context reconstruction, metric calculations, claim evidence boundaries and existing report consistency. No new inference or scoring.',
              'not_validated':['Semantic correctness/completeness of assistant claim judgments and gold annotations',
                               'Original full Context string/finish_reason not saved upstream',
                               'Tokenization or runtime speed measured independently',
                               'Chart visual layout, one-click reproduction, packaging or GitHub upload']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# T7更新版：完整校验记录','',f"结果：{'通过' if result['passed'] else '未通过'}；{result['checks_passed']}/{len(checks)}组检查通过。",'']
    for c in checks:
        text.append(f"- {'通过' if c['passed'] else '失败'}：{c['check']}。"+('' if c['passed'] else c['error']))
    text += ['', '本次只读取既有输入和结果，新增校验记录；没有重新生成回答、修改评分或重做图表。',
             '另运行10项自动化测试，确认伪造引文、引用片段越界、资料被改动、评分标志矛盾、漏分类等情况会被拦住，并检查AP分母、MRR截断和无事实回答不计满分。',
             '校验确认代码计算、数据对应及引文字面边界一致，不能证明每个语义评分或gold标注都正确。原始Context及停止原因未保存，token与耗时仅核对同学记录，图表视觉效果未检查。',
             '', '详细记录：results/t7/verification.json。单独检查命令：`python tools/verify_t7_updated.py`；这不是运行完整实验的一键复现脚本。','']
    (ROOT/'T7完整校验记录.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(failures_count > 0)


if __name__ == '__main__':
    raise SystemExit(main())
