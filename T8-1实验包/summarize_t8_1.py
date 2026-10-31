"""Validate completed human reviews and plot paired hallucination rates."""
import argparse
import json
from pathlib import Path
from statistics import mean

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('results/t8_1'))
    args = p.parse_args()
    folder = args.output_dir
    pairs = rows(folder/'paired_answers.jsonl')
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    expected_ids = set(metadata['binding']['ids'])
    if {r['id'] for r in pairs} != expected_ids or len(pairs) != len(expected_ids):
        raise ValueError('实验未完成，不能统计正式结果')
    reviews = rows(folder/'reviews.jsonl')
    expected = {(r['id'], arm) for r in pairs for arm in ('before','after')}
    if {(r['id'],r['condition']) for r in reviews} != expected or len(reviews) != len(expected):
        raise ValueError('评分缺失、重复或多余')
    pair_map = {r['id']:r for r in pairs}
    for r in reviews:
        maximum = 5 if pair_map[r['id']]['level']=='L4' else 2
        if type(r['hallucination']) is not bool or type(r['refusal']) is not bool:
            raise ValueError(f"{r['id']} {r['condition']}：请填写幻觉/拒答布尔值，未评分不能当0")
        if r['max_score'] != maximum or type(r['score']) is not int or not 0 <= r['score'] <= maximum:
            raise ValueError(f"{r['id']}：分数无效或未填")
        if maximum == 5:
            points = r.get('point_scores')
            if not isinstance(points,list) or len(points)!=5 or any(type(v) is not int or v not in (0,1) for v in points) or sum(points)!=r['score']:
                raise ValueError(f"{r['id']}：L4需填写五项0/1且合计等于score")
        if not r['reason'].strip() or (r['hallucination'] and not r['unsupported_quote'].strip()):
            raise ValueError('需填写评分理由；标记幻觉时需摘录无依据或冲突的原句')
    truncated = [f"{r['id']}:{arm}" for r in pairs for arm in ('before','after') if r[arm]['done_reason']=='length']
    if truncated:
        raise ValueError('存在截断输出，先检查或重新运行：'+','.join(truncated))
    summary = {'questions':len(pairs), 'arms':{},
               'hallucination_definition':'回答中至少一条事实性结论不被资料支持、与资料冲突或套用错误范围；漏答单独扣正确性分，不自动算幻觉。',
               'review_note':'依据原资料、适用版本及人工核实的题集评分；非模型自动幻觉率。'}
    for arm in ('before','after'):
        rs = [r for r in reviews if r['condition']==arm]
        answered = [r for r in rs if not r['refusal']]
        latency = [r['retrieval_seconds']+r[arm].get('total_seconds',r[arm]['seconds']) for r in pairs]
        summary['arms'][arm] = {'n':len(rs), 'hallucination_rate':mean(r['hallucination'] for r in rs),
            'refusal_rate':mean(r['refusal'] for r in rs),
            'hallucination_rate_among_answered':mean(r['hallucination'] for r in answered) if answered else None,
            'normalized_answer_score':mean(r['score']/r['max_score'] for r in rs),
            'mean_seconds_including_detection_and_retrieval':mean(latency)}
    summary['hallucination_reduction_percentage_points'] = 100*(summary['arms']['before']['hallucination_rate']-summary['arms']['after']['hallucination_rate'])
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Before', 'After']
    fig, axes = plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,metric,title in zip(axes, ['hallucination_rate','refusal_rate','normalized_answer_score'],
                              ['Hallucination rate','Refusal rate','Normalized answer score']):
        vals=[summary['arms'][arm][metric] for arm in ('before','after')]
        bars=ax.bar(labels,vals,color=['#8294ab','#276cbd'])
        ax.set_ylim(0,1.1); ax.set_title(title); ax.bar_label(bars,fmt='%.3f',padding=3)
    fig.savefig(folder/'comparison.png',dpi=200)
    plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
