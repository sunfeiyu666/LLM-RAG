"""Build verified T3 tables, plots and a concise Chinese report from raw outputs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from t3_core import ALPHAS, read_jsonl
from run_t3 import ANSWER_METHODS

OUT = ROOT / 'results/t3'
LABELS = {'baseline': '原始查询', 'rewrite': '查询改写', 'multi': '多查询 RRF', 'decompose': '查询分解 RRF'}


def mean(values):
    return statistics.fmean(values)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('\n', ' ') for x in row) + ' |' for row in rows])


def plots(summary, best):
    config = OUT / '.mplconfig'
    config.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(config)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams['font.sans-serif'] = [family, 'DejaVu Sans']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'savefig.dpi': 180, 'figure.dpi': 120})
    charts = OUT / 'figures'
    charts.mkdir(exist_ok=True)
    methods = ['baseline', 'rewrite', 'multi', 'decompose', best]
    labels = [LABELS[m] if m in LABELS else ('BM25单路\nα=0' if summary[m]['alpha'] == 0 else f'加权RRF\nα={summary[m]["alpha"]:g}') for m in methods]
    colors = ['#777777', '#0072B2', '#009E73', '#D55E00', '#CC79A7']
    for metric, name, ylabel, high in [
        ('recall_at_5', 'strategy_recall', 'Recall@5（命中任一 gold 片段）', 1),
        ('correctness_0_5', 'strategy_quality', '答案正确性（模型盲评，0–5分）', 5),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 4.7), layout='constrained')
        values = [summary[m][metric] for m in methods]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_ylim(0, high*1.13)
        ax.set_ylabel(ylabel)
        ax.set_title('固定20题上的查询策略对比', loc='left', fontweight='bold', pad=14)
        if high == 1:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        for bar, val, method in zip(bars, values, methods):
            label = f'{summary[method]["hits"]}/20' if high == 1 else f'{val:.2f}'
            ax.text(bar.get_x()+bar.get_width()/2, val+high*0.025, label, ha='center', va='bottom')
        ax.grid(axis='y', alpha=0.18)
        ax.set_axisbelow(True)
        fig.savefig(charts / (name+'.png'))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout='constrained')
    ax.plot(ALPHAS, [summary[f'rrf_{a:g}']['recall_at_5'] for a in ALPHAS], 'o-', color='#0072B2', label='加权 RRF（排名融合）', linewidth=2)
    ax.plot(ALPHAS, [summary[f'score_{a:g}']['recall_at_5'] for a in ALPHAS], 's--', color='#D55E00', label='归一化分数加权（敏感性检查）', linewidth=1.7)
    ax.set(xlim=(-0.03, 1.03), ylim=(0, 1.08), xticks=ALPHAS,
           xlabel='α：LSA向量路权重；BM25路权重为1−α', ylabel='Recall@5')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title('融合权重扫描：检索命中率', loc='left', fontweight='bold', pad=14)
    ax.legend(loc='lower left', frameon=False)
    ax.grid(alpha=0.18)
    fig.savefig(charts / 'fusion_alpha_recall.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout='constrained')
    ax.plot(ALPHAS, [summary[f'rrf_{a:g}']['correctness_0_5'] for a in ALPHAS], 'o-', color='#009E73', linewidth=2)
    ax.set(xlim=(-0.03, 1.03), ylim=(0, 5.25), xticks=ALPHAS,
           xlabel='α：LSA向量路权重；BM25路权重为1−α', ylabel='答案正确性（模型盲评，0–5分）')
    ax.set_title('加权RRF权重扫描：答案质量', loc='left', fontweight='bold', pad=14)
    ax.grid(alpha=0.18)
    fig.savefig(charts / 'fusion_alpha_quality.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    questions = read_jsonl(ROOT / 'data/eval/qa_set.jsonl')
    retrieval = {r['id']: r for r in read_jsonl(OUT / 'retrieval.jsonl')}
    answers_raw = read_jsonl(OUT / 'answers.jsonl')
    judgments = {r['id']: r for r in read_jsonl(OUT / 'judgments.jsonl')}
    answers = {(r['id'], r['method']): r for r in answers_raw}
    ids = {q['id'] for q in questions}
    expected = {(qid, method) for qid in ids for method in ANSWER_METHODS}
    if set(retrieval) != ids or set(judgments) != ids or set(answers) != expected or len(answers_raw) != len(expected):
        raise ValueError(f'Incomplete/duplicate data: questions={len(ids)}, retrieval={len(retrieval)}, answers={len(answers)}, judgments={len(judgments)}')
    plans = read_jsonl(OUT / 'query_plans.jsonl')
    summary = {}
    per_question = []
    all_methods = list(retrieval[next(iter(ids))]['methods'])
    for method in all_methods:
        entries = [retrieval[q['id']]['methods'][method] for q in questions]
        stats = {
            'questions': len(questions), 'hits': sum(e['metrics']['recall_at_5'] for e in entries),
            'recall_at_5': mean(e['metrics']['recall_at_5'] for e in entries),
            'gold_coverage_at_5': mean(e['metrics']['gold_coverage_at_5'] for e in entries),
            'mrr_at_5': mean(e['metrics']['reciprocal_rank_at_5'] for e in entries),
        }
        if method.startswith(('rrf_', 'score_')):
            stats['alpha'] = float(method.split('_')[1])
        if method in ANSWER_METHODS:
            stats.update({
                'correctness_0_5': mean(judgments[q['id']]['scores'][method]['score'] for q in questions),
                'rouge_l_char': mean(answers[(q['id'], method)]['rouge_l_char'] for q in questions),
                'truncated_answers': sum(answers[(q['id'], method)]['llm']['truncated'] for q in questions),
                'invalid_citation_numbers': sum(not answers[(q['id'], method)]['citation_numbers_valid'] for q in questions),
                'cached_answers': sum(answers[(q['id'], method)]['llm']['cache_hit'] for q in questions),
            })
        stats['by_level'] = {}
        for level in ['L1', 'L2', 'L3', 'L4']:
            subset = [q for q in questions if q['level'] == level]
            stats['by_level'][level] = {'n': len(subset), 'recall_at_5': mean(retrieval[q['id']]['methods'][method]['metrics']['recall_at_5'] for q in subset)}
            if method in ANSWER_METHODS:
                stats['by_level'][level]['correctness_0_5'] = mean(judgments[q['id']]['scores'][method]['score'] for q in subset)
        summary[method] = stats
    for q in questions:
        for method in ANSWER_METHODS:
            answer = answers[(q['id'], method)]
            per_question.append({'id': q['id'], 'level': q['level'], 'method': method,
                                 **retrieval[q['id']]['methods'][method]['metrics'],
                                 **judgments[q['id']]['scores'][method], 'rouge_l_char': answer['rouge_l_char'],
                                 'truncated': answer['llm']['truncated'], 'citation_numbers_valid': answer['citation_numbers_valid']})
    best = max([f'rrf_{a:g}' for a in ALPHAS], key=lambda m: (
        summary[m]['recall_at_5'], summary[m]['correctness_0_5'], summary[m]['gold_coverage_at_5'], -summary[m]['alpha']))
    baseline = summary['baseline']
    cached = sum(r['llm']['cache_hit'] for r in answers_raw)
    report_summary = {'primary_n': len(questions), 'levels': dict(Counter(q['level'] for q in questions)),
                      'best_exploratory_rrf': best, 'selection_rule': 'Recall@5, then mean blinded correctness, then gold coverage; same evaluation set, no held-out claim.',
                      'logical_answer_conditions': len(expected), 'cached_identical_answers': cached,
                      'metrics': summary, 'judge': 'same-model blinded LLM-as-Judge, 0–5; not independent human correctness'}
    (OUT / 'summary.json').write_text(json.dumps(report_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'per_question_scores.json').write_text(json.dumps(per_question, ensure_ascii=False, indent=2), encoding='utf-8')
    source_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in list((ROOT/'src').glob('*.py'))+list((ROOT/'tools').glob('*.py'))}
    (OUT / 'implementation_sha256.json').write_text(json.dumps(source_hashes, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.no_plots:
        plots(summary, best)
    shown = ['baseline', 'rewrite', 'multi', 'decompose', best]
    strategy_rows = []
    for method in shown:
        s = summary[method]
        strategy_rows.append([LABELS[method] if method in LABELS else ('BM25单路 α=0' if s['alpha'] == 0 else f'加权RRF α={s["alpha"]:g}'), f'{s["hits"]}/{len(questions)}',
                              f'{s["recall_at_5"]:.1%}', f'{s["gold_coverage_at_5"]:.1%}',
                              f'{s["correctness_0_5"]:.2f}', f'{s["rouge_l_char"]:.3f}', s['truncated_answers']])
    strategy_table = table(['策略', '命中题数', 'Recall@5', 'gold覆盖', '正确性0–5', '字符ROUGE-L', '截断数'], strategy_rows)
    alpha_table = table(['α', 'RRF Recall@5', 'RRF gold覆盖', 'RRF正确性0–5', '分数加权 Recall@5'], [
        [a, f'{summary[f"rrf_{a:g}"]["recall_at_5"]:.1%}', f'{summary[f"rrf_{a:g}"]["gold_coverage_at_5"]:.1%}',
         f'{summary[f"rrf_{a:g}"]["correctness_0_5"]:.2f}', f'{summary[f"score_{a:g}"]["recall_at_5"]:.1%}'] for a in ALPHAS])
    changes = [(method, summary[method]['recall_at_5'] - baseline['recall_at_5']) for method in ['rewrite', 'multi', 'decompose']]
    strategy_interpretation = '；'.join(f'{LABELS[m]}相对基线变化{v*100:+.1f}个百分点' for m, v in changes)
    best_stats = summary[best]
    quality_best = max([f'rrf_{a:g}' for a in ALPHAS], key=lambda m: summary[m]['correctness_0_5'])
    quality_stats = summary[quality_best]
    sparse_comparison = ('BM25单路（α=0）已达到相同Recall@5，因此不能声称混合检索在命中率上进一步超过纯BM25；权重的取舍还需看答案质量和证据覆盖。'
                         if summary['rrf_0']['recall_at_5'] == best_stats['recall_at_5'] else
                         'BM25单路与混合配置的差异以上表为准；不能把相对向量基线的改善自动解释成相对BM25也改善。')
    truncation = sum(r['llm']['truncated'] for r in answers_raw)
    text = f'''# T3 检索前查询优化与融合实验报告

## 任务与结论

本实验对应课程实践T3及课件9.6 p39–40、9.2.2 p15。实现大模型查询改写、多查询、查询分解，以及RRF结果融合，在固定20题上比较检索与答案质量。

原始查询Recall@5为{baseline['recall_at_5']:.1%}。{strategy_interpretation}。因此不能声称查询改写一定有效。按“先Recall、再正确性”的规则，加权RRF扫描选中α={best_stats['alpha']:g}，Recall@5为{best_stats['recall_at_5']:.1%}、答案正确性模型评分为{best_stats['correctness_0_5']:.2f}/5。α=0是纯BM25端点，不是非零混合。这个选择基于同一小评测集，属于探索性结果，不是独立测试集上的最优保证。

原始查询的答案评分为{baseline['correctness_0_5']:.2f}/5，优先召回选出的配置并未提高这一评分。因此本次不能总结为“融合同时提高了检索和答案”。若优先答案评分，扫描中的最高值反而是α={quality_stats['alpha']:g}、{quality_stats['correctness_0_5']:.2f}/5，体现了不同目标的取舍。

## 实验控制

- 数据来自固定的23篇学院相关资料，202个片段；切分650字符、重叠120字符。使用仓库正式20题，L1–L4各5题，gold片段均在本机索引中存在。
- 固定原有字符TF-IDF（1–4元组）+64维LSA、归一化及FAISS检索。此向量路是统计降维表示，不冒充神经嵌入模型。原有数字/英文移除规则保持不变，避免同时更换索引变量。
- BM25为新增的融合检索路，中文单字与双字词元、完整数字和英文标识；k1=1.5，b=0.75，不使用标准答案设计词表。
- 单路取前20候选，最终统一返回前5个不重复片段；多查询/子查询并行执行检索、按同一RRF常数60融合。
- 本地qwen3:4b-instruct，temperature=0.2、seed=42、上下文8192、输出上限768 token；各最终回答使用相同的证据约束与引用提示。查询规划与评分temperature=0。
- 为节省重复调用，一个只看原问题的规划调用同时产生三种查询计划。单事实问题不强行分解；复合问题拆成不同子意图。标准答案不进入查询规划或答案生成。
- 查询分解先分别检索，再基于最终同一批top-5证据生成各子问题短答，最后综合。短答不能补充top-5以外事实。这一策略多一个子答案生成阶段，不能解释成零成本改进。
- 重新运行全部20题，不复用仓库中尚未补齐的T1结果。{len(expected)}个逻辑回答条件中有{cached}个命中完整请求缓存；相同原问题、上下文及参数对应同一个输出，不是多次独立采样。

## 指标与评分

Recall@5严格按作业FAQ：任一gold片段在前5块中即记该题命中。另报gold覆盖比例，避免跨片段问题“只找到一块就看似成功”。字符ROUGE-L为次要词面指标，不用它替代复杂问答正确性。

正确性由同一Qwen进行LLM-as-Judge盲评：隐藏策略名、按固定种子打乱候选顺序，以仓库提供的参考答案为依据，按0–5分评价结论、条件及完整性。完全相同的答案只评一次并将分数共享给对应条件，避免同一文本因位置产生虚假差异。实际gold由仓库的助手辅助标注，尚无独立教师/学生审核声明；本实验评分也不冒称独立人工评分。自评偏差、小样本和单次运行限制仍存在。

## 策略对比

{strategy_table}

![查询策略命中率](results/t3/figures/strategy_recall.png)

![查询策略答案质量](results/t3/figures/strategy_quality.png)

正式题目的表述通常已包含文件名称、年份及明确子意图。改写有时删弱关键词，或使多路排名反复强调某一子意图，导致其他证据被挤出前5。因此本组数据不支持“查询越多越好”。融合路保留BM25精确词面匹配，有机会弥补LSA降维及移除年份造成的不足；具体仍应逐题检查证据。

一个可核对的负面案例是q011：子问题短答已给出GPA门槛3.0，但最终综合回答只说2.9未达标，漏掉了用户要求的明确规则与比较过程，模型评分从原始查询的5分降到2分。这提示分解链路除了检索，还可能在综合时丢失必要条件。另一个评分边界是q007：两种答案都给出了正确申请日期，但评分器给原始查询5分、分解4分。因此对0.1至0.25分的小差异不应作强结论，仍需独立人工复核。

严格片段编号也有边界：补充案例c03原始检索拿到的p00191已包含辅修日期，而gold只标了与其重叠的p00190，导致编号覆盖被低估。原始答案本轮已答对，不应将优化后编号覆盖增加写成“补齐缺失日期”。主表统一遵守作业的gold ID口径，案例则另行核对实际证据，详见案例报告。

## 融合权重实验

加权RRF使用 s(d)=α/(60+r向量(d))+(1−α)/(60+rBM25(d))；某路未召回该片段时，该项为0。α=1只使用原向量路，α=0只使用BM25。各路候选内去重，按片段ID确定并列次序。

为避免把“RRF”与题目示例中的分数加权混为一谈，另做min-max归一化后 s(d)=α·dense_norm(d)+(1−α)·bm25_norm(d) 的检索敏感性检查。它不叫RRF；其答案未另行生成，表中只报告检索指标。

{alpha_table}

![融合权重与检索命中率](results/t3/figures/fusion_alpha_recall.png)

![融合权重与答案质量](results/t3/figures/fusion_alpha_quality.png)

最佳配置按Recall@5、正确性模型均分、gold覆盖依次比较。曲线若出现平台，不能凭相同命中率断言唯一最佳权重；每题占5个百分点，20题不足以支持“显著提升”的统计结论。

{sparse_comparison}

## 案例与边界

完整中间查询、原始及优化后召回片段、gold证据、最终答案见《T3案例分析.md》。额外口语化挑战题与主20题分开存储，所有挑战结果都保留，不混入主表、不冒充独立泛化测试。它们用于展示查询优化适用条件，而不是掩盖正式题上的负优化。

本轮共{truncation}个最终回答触及输出上限，表中已列出各策略截断数；截断输出按实际文本评分，没有静默替换。引用编号合法性也单独记录，但编号合法不能证明内容有依据。

## 复现与交付

代码：src/t3_core.py、src/run_t3.py；补充案例：tools/run_t3_challenges.py。结果：results/t3/内的查询计划、检索排名、180条答案、20份盲评分配与评分、逐题指标和汇总；四张PNG曲线在figures/。具体命令见README_T3.md。

核心局限：原有LSA向量路不是神经稠密模型；评测集及金标仍需提交者复核；同模型生成与自评存在偏差；权重在同一小集合上探索；无独立多次采样或显著性检验。本次不代替T7完整忠实度、幻觉率与失败归因评估。

## 方法来源与辅助说明

- RRF原论文：[Cormack等，SIGIR 2009](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)。本实验采用其排名倒数思想的加权版本。
- BM25公式参考：[Stanford Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)。中文词元方案为本实现选择。
- 结构化查询输出：[Ollama Structured Outputs](https://ollama.com/blog/structured-outputs)。
- 基础代码和正式评测数据来自sunfeiyu666/LLM-RAG，起点提交53c4c7b。新增实现、实验整理和报告有大模型辅助；本地模型真实运行，未编造结果。提交者应复核并注明辅助与来源。
'''
    (ROOT / 'T3实验报告.md').write_text(text, encoding='utf-8')
    print(json.dumps({'complete': True, 'questions': len(questions), 'logical_answers': len(expected), 'best_rrf': best, 'summary': str(OUT/'summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
