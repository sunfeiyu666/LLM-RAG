"""Retain all diagnostic cases and render full, evidence-backed examples."""
import json
from pathlib import Path
import random
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kb import load_index
from t3_core import LocalLLM, RetrievalEngine, plan_query, retrieval_metrics, read_jsonl, append_jsonl, ENDPOINT, MODEL, rouge_l
from run_t3 import SYSTEM, context_for, subanswers, judge

OUT = ROOT / 'results/t3'
METHODS = ['baseline', 'rewrite', 'multi', 'decompose']


def main():
    originals = {q['id']: q for q in read_jsonl(ROOT/'data/eval/qa_set.jsonl')}
    questions = read_jsonl(ROOT/'data/eval/t3_colloquial_challenges.jsonl')
    # This one additional example is explicitly exploratory after inspecting the
    # first four. It does not enter the primary n=20 benchmark or a held-out claim.
    extra = {**originals['q012'], 'id': 'c05', 'source_question_id': 'q012', 'supplemental': True,
             'question': '云大2027学校那套保研，四年制软工应届生绩点三点二、四级四百三，一门专业必修当年没过后来补过的，别的都行，只看专业必修这项能算过吗？',
             'purpose': 'Additional exploratory L3 paraphrase after the first four diagnostics; not a held-out test.'}
    extra_path = ROOT/'data/eval/t3_colloquial_challenge_extra.jsonl'
    encoded = json.dumps(extra, ensure_ascii=False)+'\n'
    if extra_path.exists() and extra_path.read_text(encoding='utf-8') != encoded:
        raise ValueError('Exploratory input changed')
    extra_path.write_text(encoded, encoding='utf-8')
    questions.append(extra)
    with urlopen(ENDPOINT+'/api/tags', timeout=10) as r:
        info = next(m for m in json.load(r)['models'] if m['name'] == MODEL)
    llm = LocalLLM(OUT/'cache', info['digest'])
    engine = RetrievalEngine(load_index())
    retrieval = {r['id']: r for r in read_jsonl(OUT/'challenge_retrieval.jsonl')}
    retrieval.update({r['id']: r for r in read_jsonl(OUT/'challenge_retrieval_extra.jsonl')})
    if extra['id'] not in retrieval:
        plan = plan_query(llm, extra['question'])
        methods = engine.retrieve(extra['question'], plan)
        for m, data in methods.items():
            data['metrics'] = retrieval_metrics([x['chunk_id'] for x in data['top']], extra['gold_chunk_ids'])
        record = {'id': extra['id'], 'source_question_id': extra['source_question_id'], 'level': extra['level'],
                  'question': extra['question'], 'plan': plan, 'methods': methods}
        append_jsonl(OUT/'challenge_retrieval_extra.jsonl', record)
        retrieval[extra['id']] = record
    answers = {(r['id'], r['method']): r for r in read_jsonl(OUT/'challenge_answers.jsonl')}
    judgments = {r['id']: r for r in read_jsonl(OUT/'challenge_judgments.jsonl')}
    # Full generations for all five examples, not just the most favorable one.
    for q in questions:
        for method in METHODS:
            key = (q['id'], method)
            if key in answers:
                continue
            print('CASE_ANSWER', *key, flush=True)
            route = retrieval[q['id']]['methods'][method]
            docs = engine.materialize(route['top'])
            context = context_for(docs)
            parts, sub_meta = [], None
            if method == 'decompose':
                parts, sub_meta = subanswers(llm, q['question'], retrieval[q['id']]['plan']['sub_queries'], context)
            user = f'参考资料：\n{context}\n\n问题：{q["question"]}'
            if parts:
                user += '\n\n基于以上同一批资料的子问题短答（可能不完整，必须核对资料后综合；不得用短答补充资料外事实）：\n'+json.dumps(parts, ensure_ascii=False)
            answer, metadata = llm.call(SYSTEM, user, limit=768, temperature=0.2)
            record = {'id': q['id'], 'method': method, 'question': q['question'], 'answer': answer,
                      'context': context, 'metrics': route['metrics'], 'subanswers': parts, 'subanswer_llm': sub_meta,
                      'llm': metadata, 'rouge_l_char': rouge_l(answer, q['gold_answer'])}
            append_jsonl(OUT/'challenge_answers.jsonl', record)
            answers[key] = record
        if q['id'] not in judgments:
            result = judge(llm, q, {m: answers[(q['id'], m)] for m in METHODS})
            append_jsonl(OUT/'challenge_judgments.jsonl', result)
            judgments[q['id']] = result
        print('CASE_DONE', q['id'], {m: retrieval[q['id']]['methods'][m]['metrics'] for m in METHODS}, flush=True)
    full_recoveries = []
    for q in questions:
        row = retrieval[q['id']]
        for m in METHODS[1:]:
            if not row['methods']['baseline']['metrics']['recall_at_5'] and row['methods'][m]['metrics']['recall_at_5']:
                full_recoveries.append((row['methods'][m]['metrics']['gold_coverage_at_5'], judgments[q['id']]['scores'][m]['score'], q['id'], m))
    selected = max(full_recoveries) if full_recoveries else None
    selection = {'strict_recovery_selected': {'id': selected[2], 'method': selected[3]} if selected else None,
                 'overlap_annotation_case': {'id': 'c03', 'method': 'decompose'},
                 'selection_note': 'Post-hoc illustrative case selection; all five outcomes retained, primary 20-question statistics unchanged.'}
    (OUT/'case_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    names = {'baseline': '原始查询', 'rewrite': '查询改写', 'multi': '多查询', 'decompose': '查询分解'}
    text = ['# T3 查询优化案例分析',
            '以下是口语化诊断示例，与正式20题的统计完全分开。前4题在第一次调用前写定；c05在查看前4题后额外构造，是探索样例。展示案例按实际改善事后选择，所有结果保留，不能用它们声称独立泛化能力。',
            '## 全部诊断结果']
    table_lines = ['| 题号 | 等级 | 原始查询 | 改写 | 多查询 | 分解 |',
                   '| --- | --- | --- | --- | --- | --- |']
    for q in questions:
        values = []
        for m in METHODS:
            metrics = retrieval[q['id']]['methods'][m]['metrics']
            score = judgments[q['id']]['scores'][m]['score']
            values.append(f"命中{metrics['recall_at_5']}，gold覆盖{metrics['gold_coverage_at_5']:.0%}，正确性{score}/5")
        table_lines.append('| '+' | '.join([q['id'], q['level'], *values])+' |')
    text.append('\n'.join(table_lines))
    chosen = []
    if selected:
        chosen.append((selected[2], selected[3], '案例一 从未命中到命中'))
    if not selected or selected[2] != 'c03':
        chosen.append(('c03', 'decompose', '案例二 重叠分块与金标编号的边界'))
    for cid, optimized, heading in chosen:
        q = next(q for q in questions if q['id'] == cid)
        row = retrieval[cid]
        text += ['\n## '+heading, f"题号{cid}，{q['level']}，对应正式题{q['source_question_id']}。", '**原始问题：**'+q['question'],
                 '**参考答案：**'+q['gold_answer'], '**原始查询检索：**']
        for position, hit in enumerate(row['methods']['baseline']['top'], 1):
            doc = engine.documents[hit['chunk_id']]
            flag = 'gold' if hit['chunk_id'] in q['gold_chunk_ids'] else '非标注gold'
            text += [f"{position}. {hit['chunk_id']}，{doc.metadata['title']}（{flag}）", '> '+doc.page_content[:210].replace('\n',' ')]
        text += ['**模型生成的中间查询：**', '改写：'+row['plan']['rewrite'], '多查询：']
        text += [f'{i}. {query}' for i, query in enumerate(row['plan']['multi_queries'],1)]
        text += ['子问题：']+[f'{i}. {query}' for i, query in enumerate(row['plan']['sub_queries'],1)]
        text += ['**优化后采用'+names[optimized]+'，RRF融合的top-5：**']
        for position, hit in enumerate(row['methods'][optimized]['top'],1):
            doc = engine.documents[hit['chunk_id']]
            flag = 'gold' if hit['chunk_id'] in q['gold_chunk_ids'] else '非标注gold'
            text += [f"{position}. {hit['chunk_id']}，{doc.metadata['title']}（{flag}）", '> '+doc.page_content.replace('\n','\n> ')]
        for method in ['baseline', optimized]:
            answer = answers[(cid,method)]
            text += ['**'+names[method]+'的真实最终回答：**', answer['answer'],
                     f"模型盲评：{judgments[cid]['scores'][method]['score']}/5；理由：{judgments[cid]['scores'][method]['reason']}。"]
            if answer['subanswers']:
                text += ['**分解阶段的子问题短答：**']+[f"- {part['question']} → {part['answer']}" for part in answer['subanswers']]
        before = row['methods']['baseline']['metrics'];after = row['methods'][optimized]['metrics']
        text += [f"**检索变化：**作业口径命中{before['recall_at_5']}→{after['recall_at_5']}；gold覆盖{before['gold_coverage_at_5']:.0%}→{after['gold_coverage_at_5']:.0%}。"]
        if cid == 'c03':
            text += ['**原文复核与原因：**原始检索中的ynu029_p000_00191已包含辅修5月21日至26日，ynu030_p000_00196已包含微专业6月11日至15日，因此原始检索实际已具备两类时间依据。gold只标了前一个重叠片段ynu029_p000_00190，没有同时标ynu029_p000_00191，造成严格gold编号覆盖只有50%。分解后拿到被标注的p00190，使编号覆盖升到100%，但不是此前缺少日期证据，也不是Recall从0到1。原始最终答案本轮已正确回答全部日期，分解后的答案反而只说截止日，遗漏开始日。此例说明固定chunk金标可能低估等价重叠片段，必须核对原文，不能机械地把编号覆盖提升解释为答案改善。']
        elif cid == 'c02':
            text += ['**原文复核与原因：**分解后新召回ynu010_p000_00111，其中第四条明确给出转专业其他课程累计认定不超过10学分；原始top-5没有这一依据。因此按本作业任一gold命中的口径，确实实现0到1，答案模型评分也从1分升到3分。',
                     '**尚未解决的部分：**本次仍未找齐ynu010_p000_00112和ynu010_p000_00116。优化后回答的15学分引用指向微专业通知中针对2024级的条款，不能直接作为问题中2025级学生的依据；必须补查正式学分认定规定及其适用年级。故本例是检索与部分答案的改善，不是完整问题已经解决，更不能保证真实学生可获学分认定。']
        else:
            text += ['**边界：**命中任一gold只是作业的二元指标。若gold覆盖不足100%，必须继续说明缺了哪些依据；不能把部分证据恢复写成整个问题已经完全解决。具体短答和最终回答以上面保存的真实文本为准。']
    text += ['\n## 原文核验与复现',
             '案例引用来自本机固定的202片段及其版本信息。展示由助手核对，提交者仍应打开原文确认金标。模型评分仅辅助分析，不冒称独立人工评审。',
             '完整每路候选排名在results/t3/challenge_retrieval*.jsonl；完整回答在challenge_answers.jsonl；盲评分配和评分在challenge_judgments.jsonl。运行tools/run_t3_challenges.py后，再运行tools/build_t3_cases.py可复现。']
    (ROOT/'T3案例分析.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')
    print('CASE_REPORT_READY', selection, flush=True)


if __name__ == '__main__':
    main()
