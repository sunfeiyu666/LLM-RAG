"""Aggregate T7, draw stratified charts, and render reviewed failure analyses."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from t7_core import KS,read_jsonl,normalize
OUT=ROOT/'results/t7'


def avg(values):
    values=[v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def pct(value):
    return f'{value:.1%}' if value is not None else 'N/A'


def num(value):
    return f'{value:.3f}' if value is not None else 'N/A'


def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x).replace('\n',' ') for x in row)+' |' for row in rows])


def apply_repairs(raw,repairs):
    result=copy.deepcopy(raw)
    for r in result:
        sentences={s['id']:s for s in r['evidence_sentences']}
        for c in r['claims']:
            if c['id'] not in repairs:continue
            replacement=repairs[c['id']]
            c['original_assessment']={k:c[k] for k in ['score','verdict','evidence','supported','effective_score']}
            quotes=[]
            for sid in replacement['evidence_sentence_ids']:
                s=sentences[sid]
                quotes.append({'sentence_id':sid,'chunk_id':s['chunk_id'],'quote':s['text'],'quote_verified':True})
            supported=replacement['verdict']=='supported' and replacement['score']>=4 and bool(quotes)
            c.update(score=replacement['score'],verdict=replacement['verdict'],evidence=quotes,
                     supported=supported,effective_score=replacement['score'] if supported else min(replacement['score'],3),
                     link_recheck_reason=replacement['reason'],link_recheck_request=replacement['llm']['request_hash'])
        n=len(r['claims']);supported=sum(c['supported'] for c in r['claims'])
        r['raw_unsupported_fact_count']=r['unsupported_fact_count']
        r.update(fact_count=n,supported_fact_count=supported,unsupported_fact_count=n-supported,
                 faithfulness_1_5=avg(c['effective_score'] for c in r['claims']),
                 hallucination_rate=(n-supported)/n if n else None)
    return result


def aggregate(rows):
    keys=[f'{name}_at_{k}' for name in ['recall','precision','gold_coverage'] for k in KS]
    keys+=['mrr_at_10','map_at_10','mrr_full','map_full','correctness_0_5','rouge_l_char','faithfulness_1_5']
    result={k:avg(r[k] for r in rows) for k in keys}
    n=sum(r['fact_count'] for r in rows);unsupported=sum(r['unsupported_fact_count'] for r in rows)
    result.update(n=len(rows),fact_count=n,unsupported_fact_count=unsupported,
                  hallucination_micro=unsupported/n if n else None,
                  hallucination_macro=avg(r['hallucination_rate'] for r in rows),
                  no_fact_answers=sum(r['fact_count']==0 for r in rows),
                  truncated_answers=sum(r['generation_truncated'] for r in rows),
                  citation_number_errors=sum(not r['citation_numbers_valid'] for r in rows))
    return result


def plot(summary):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
    (OUT/'.mplconfig').mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.dpi':180})
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    levels=['L1','L2','L3','L4'];x=list(range(4));width=.18
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for index,(key,label,color) in enumerate([
        ('recall_at_5','Recall@5','#0072B2'),('precision_at_5','Precision@5','#D55E00'),
        ('mrr_at_10','MRR@10','#009E73'),('map_at_10','MAP@10','#CC79A7')]):
        ax.bar([v+(index-1.5)*width for v in x],[summary['by_level'][level][key] for level in levels],width,label=label,color=color)
    ax.set(xticks=x,xticklabels=[l+'\n(n=5)' for l in levels],ylim=(0,1.17),ylabel='指标值',title='BM25系统的分层检索指标')
    ax.legend(ncol=2,frameon=False,loc='upper right');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.savefig(figures/'stratified_retrieval.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.7),layout='constrained')
    for i,(key,label,color) in enumerate([('correctness_0_5','正确性（0–5）','#0072B2'),('faithfulness_1_5','忠实度（1–5）','#009E73')]):
        vals=[summary['by_level'][l][key] for l in levels]
        bars=ax.bar([v+(i-.5)*.32 for v in x],[v if v is not None else math.nan for v in vals],.32,label=label,color=color)
        for bar,val in zip(bars,vals):
            if val is not None:ax.text(bar.get_x()+bar.get_width()/2,val+.07,f'{val:.2f}',ha='center',va='bottom',fontsize=10)
    ax.set(xticks=x,xticklabels=levels,ylim=(0,5.8),ylabel='题均助手复核评分',title='BM25系统的分层生成质量')
    ax.legend(frameon=False,ncol=2,loc='upper right');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.savefig(figures/'stratified_generation.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.7),layout='constrained')
    vals=[summary['by_level'][l]['hallucination_micro'] for l in levels]
    bars=ax.bar(levels,[v if v is not None else math.nan for v in vals],color='#D55E00',width=.55)
    for bar,level,val in zip(bars,levels,vals):
        s=summary['by_level'][level]
        if val is not None:ax.text(bar.get_x()+bar.get_width()/2,val+.025,f"{s['unsupported_fact_count']}/{s['fact_count']}\n{val:.1%}",ha='center',va='bottom')
    ax.set(ylim=(0,1.15),ylabel='无依据事实数 / 全部事实数',title='BM25系统的分层幻觉率（按事实数汇总）')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.savefig(figures/'stratified_hallucination.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.7),layout='constrained')
    for name,label,marker,color in [('recall','作业口径 Recall@k','o','#0072B2'),('precision','Precision@k','s','#D55E00'),('gold_coverage','gold片段覆盖','^','#009E73')]:
        ax.plot(KS,[summary['overall'][f'{name}_at_{k}'] for k in KS],marker+'-',label=label,color=color)
    ax.set(xticks=KS,xlabel='k（只改变检索指标截断位置，答案始终用top-5）',ylim=(0,1.12),ylabel='指标值',title='BM25检索的不同k值表现')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,loc='center right',bbox_to_anchor=(1,.45));ax.grid(alpha=.17)
    fig.savefig(figures/'retrieval_k.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--draft',action='store_true');args=parser.parse_args()
    metrics=read_jsonl(OUT/'per_question_metrics.jsonl');raw=read_jsonl(OUT/'faithfulness.jsonl')
    qs={q['id']:q for q in read_jsonl(ROOT/'data/eval/qa_set.jsonl')}
    answers={q['id']:q for q in read_jsonl(ROOT/'data/system_answers.jsonl')}
    assert len(metrics)==20 and len({r['id'] for r in metrics})==20
    assert len(raw)==20 and {r['id'] for r in raw}==set(qs)
    repairs_list=read_jsonl(OUT/'evidence_link_repairs.jsonl');repairs={r['claim_id']:r for r in repairs_list}
    flagged={c['id'] for r in raw for c in r['claims'] if c['verification_flags']}
    assert set(repairs)==flagged,'Run evidence-link recheck for every flagged claim'
    automatic_checked=apply_repairs(raw,repairs)
    checked=read_jsonl(OUT/'faithfulness_reviewed.jsonl')
    assert len(checked)==20 and {r['id'] for r in checked}==set(qs)
    for r in checked:
        assert r['units']==raw[next(i for i,x in enumerate(raw) if x['id']==r['id'])]['units']
        assert {u['id'] for u in r['units']}==({c['unit_id'] for c in r['claims']}|{u['id'] for u in r['excluded_units']})
    faith={r['id']:r for r in checked}
    new_scores={r['id']:r for r in read_jsonl(OUT/'correctness_new.jsonl')}
    assert set(new_scores)==set(qs)
    reviewed_scores={r['id']:r for r in json.loads((OUT/'correctness_reviewed.json').read_text(encoding='utf-8'))}
    assert set(reviewed_scores)==set(qs)
    for qid,r in reviewed_scores.items():
        assert r['answer_sha256']==hashlib.sha256(answers[qid]['answer'].encode()).hexdigest()
        assert r['score']==sum(c['awarded'] for c in r['criteria'])
    rows=[];flat=[]
    for r in metrics:
        f=faith[r['id']]
        rows.append({**r,'t3_correctness_0_5':r['correctness_0_5'],
                     'new_model_correctness_0_5':new_scores[r['id']]['score'],
                     'correctness_0_5':reviewed_scores[r['id']]['score'],
                     'correctness_reason':'；'.join(c['observation'] for c in reviewed_scores[r['id']]['criteria'] if c['awarded']<c['available']),
                     **{k:f[k] for k in ['fact_count','supported_fact_count','unsupported_fact_count','faithfulness_1_5','hallucination_rate']}})
        flat.extend({'question_id':r['id'],'level':r['level'],**c} for c in f['claims'])
    summary={'target':'用户指定的本地T3 BM25端点（α=0）','n':20,'overall':aggregate(rows),
             'by_level':{level:aggregate([r for r in rows if r['level']==level]) for level in ['L1','L2','L3','L4']},
             'evidence_link_rechecks':len(repairs),'raw_unsupported_fact_count':sum(r['unsupported_fact_count'] for r in raw),
             'raw_fact_count':sum(r['fact_count'] for r in raw),
             'automatic_link_checked_fact_count':sum(r['fact_count'] for r in automatic_checked),
             'automatic_link_checked_unsupported_count':sum(r['unsupported_fact_count'] for r in automatic_checked),
             't3_correctness_mean':avg(r['correctness_0_5'] for r in metrics),
             'fresh_model_correctness_mean':avg(r['score'] for r in new_scores.values()),
             'assessment_source':'Final correctness: explicit assistant 0-5 rubric review. Final faithfulness: full assistant source-only review. Raw model calls retained; not independent human review.',
             'notes':'Frozen T3 real outputs, no re-generation. Raw scores and explicit source-only review are both preserved.'}
    sorted_rows=sorted(rows,key=lambda r:(r['correctness_0_5'],r['faithfulness_1_5'] if r['faithfulness_1_5'] is not None else -1,-(r['hallucination_rate'] or 0),r['id']))
    worst=sorted_rows[:5]
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'per_question.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'claim_evaluations.json').write_text(json.dumps(flat,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'worst_five.json').write_text(json.dumps(worst,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'faithfulness_checked.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in checked),encoding='utf-8')
    plot(summary)
    full_table=mdtable(['题号','等级','R@5','P@5','MRR@10','MAP@10','正确性/5','忠实度/5','无依据/事实','幻觉率','截断'],[
        [r['id'],r['level'],pct(r['recall_at_5']),num(r['precision_at_5']),num(r['mrr_at_10']),num(r['map_at_10']),
         num(r['correctness_0_5']),num(r['faithfulness_1_5']),f"{r['unsupported_fact_count']}/{r['fact_count']}",pct(r['hallucination_rate']),'是' if r['generation_truncated'] else '否'] for r in rows])
    (ROOT/'T7完整评估表.md').write_text('# T7 完整评估表\n\n'+full_table+'\n\nR@5按作业的任一gold命中口径；忠实度为事实1–5分的题内平均，幻觉率为严格未被本轮Context完整支持的事实比例。完整k值与全库指标见results/t7/per_question.json。\n',encoding='utf-8')
    if args.draft:
        print(json.dumps({'stage':'draft','summary':summary,'worst_ids':[r['id'] for r in worst]},ensure_ascii=False));return
    reviews_path=OUT/'failure_reviews.json'
    assert reviews_path.exists(),'Review worst_five.json and create failure_reviews.json before final report'
    reviews=json.loads(reviews_path.read_text(encoding='utf-8'))
    assert {r['id'] for r in reviews}=={r['id'] for r in worst} and len(reviews)==5
    allowed={'检索失败','排序失败','生成失败','评测集本身有歧义'}
    chunks={d['metadata']['chunk_id']:d for d in json.loads((ROOT/'data/frozen_chunks.json').read_text(encoding='utf-8'))}
    for review in reviews:
        assert review['primary_cause'] in allowed
        assert normalize(review['answer_excerpt']) in normalize(answers[review['id']]['answer'])
        for e in review['evidence']:
            assert normalize(e['quote']) in normalize(chunks[e['chunk_id']]['page_content']),e
    by_review={r['id']:r for r in reviews}
    case_lines=['# T7 最差五题的归因与改进',
        '选择规则在复核前固定：先按正确性升序，再按忠实度升序、幻觉率降序、题号排序。以下为助手基于真实回答、实际片段及金标原文的复核，不冒称教师或独立人工审核。']
    for row in worst:
        qid=row['id'];review=by_review[qid];q=qs[qid]
        case_lines += [f'## {qid} {row["level"]}',f'**问题：**{q["question"]}',f'**参考答案：**{q["gold_answer"]}',
            f"**指标：**正确性{row['correctness_0_5']}/5，忠实度{num(row['faithfulness_1_5'])}/5，幻觉率{pct(row['hallucination_rate'])}，输出截断={'是' if row['generation_truncated'] else '否'}。",
            f"**主归因：**{review['primary_cause']}。次要因素：{review['secondary_factors']}。",
            '**错误或缺漏原文：**\n\n> '+review['answer_excerpt'].replace('\n','\n> '),
            '**完整原始回答：**\n\n'+answers[qid]['answer'],
            '**实际top-5片段：**'+', '.join(row['actual_top5']),
            '**gold片段排名：**'+json.dumps(row['gold_ranks'],ensure_ascii=False),
            '**复核说明：**'+review['analysis']]
        for e in review['evidence']:
            location='实际Context' if e['chunk_id'] in row['actual_top5'] else '候选/金标资料（未进入实际Context）'
            case_lines += [f"**依据 {e['chunk_id']}（{location}）：**\n\n> "+e['quote'].replace('\n','\n> ')]
        case_lines += ['**改进方案：**'+review['improvement'], '**如何验证改进：**'+review['verification_plan']]
    (ROOT/'T7失败案例分析.md').write_text('\n\n'.join(case_lines)+'\n',encoding='utf-8')
    overall=summary['overall']
    lowest_correct=min(summary['by_level'],key=lambda l:summary['by_level'][l]['correctness_0_5'])
    lowest_faith=min(summary['by_level'],key=lambda l:summary['by_level'][l]['faithfulness_1_5'] if summary['by_level'][l]['faithfulness_1_5'] is not None else math.inf)
    highest_hall=max(summary['by_level'],key=lambda l:summary['by_level'][l]['hallucination_micro'] or 0)
    k_table=mdtable(['k','作业Recall@k','Precision@k','gold覆盖'],[[k,pct(overall[f'recall_at_{k}']),num(overall[f'precision_at_{k}']),pct(overall[f'gold_coverage_at_{k}'])] for k in KS])
    layer_table=mdtable(['等级','题数','MRR@10','MAP@10','正确性/5','忠实度/5','无依据/事实','幻觉率'],[
        [level,s['n'],num(s['mrr_at_10']),num(s['map_at_10']),num(s['correctness_0_5']),num(s['faithfulness_1_5']),f"{s['unsupported_fact_count']}/{s['fact_count']}",pct(s['hallucination_micro'])] for level,s in summary['by_level'].items()])
    failure_table=mdtable(['题号','等级','正确性','主归因'],[[r['id'],r['level'],f"{r['correctness_0_5']}/5",by_review[r['id']]['primary_cause']] for r in worst])
    report=f'''# T7 BM25系统完整评估报告

## 评估对象与结论

本任务对应课件9.5 p33–36。按用户明确选择，评估本地T3的BM25端点（α=0），不是擅自指定尚未整合的其他小组方案。对其已保存的20条真实回答、实际top-5 Context和固定202片段做离线评估，不重新生成答案。

该快照的MRR为{num(overall['mrr_full'])}、MAP为{num(overall['map_full'])}；MRR@10为{num(overall['mrr_at_10'])}、MAP@10为{num(overall['map_at_10'])}。Recall@5为{pct(overall['recall_at_5'])}，但只表示命中至少一块gold，不代表全部证据齐全或答案正确。

答案正确性助手量表复核均分{num(overall['correctness_0_5'])}/5，题均忠实度{num(overall['faithfulness_1_5'])}/5；严格无依据事实为{overall['unsupported_fact_count']}/{overall['fact_count']}，微平均幻觉率{pct(overall['hallucination_micro'])}。正确性最低的等级为{lowest_correct}，忠实度最低为{lowest_faith}，幻觉率最高为{highest_hall}；这些维度分别反映回答任务完成度、证据支持强度和无依据事实比例。

## 数据与运行边界

- 使用原项目23篇资料、202片段，650字符切分、120字符重叠。20题L1–L4各5题，金标由原仓库助手辅助编制，仍需提交者独立核验。
- 系统检索为中文单字/双字及完整数字英文标识的BM25，k1=1.5，b=0.75，最终提供5个片段。
- 原回答由qwen3:4b-instruct生成，temperature=0.2、seed=42、上下文8192、输出上限768 token；本次不提高上限来掩盖失败。{overall['truncated_answers']}/20条回答触及输出预算。
- 正确性重新按“单题问题＋参考答案＋实际回答”进行0–5分模型评分，20条新调用在correctness_new.jsonl中。但新评分仍出现引用编号误判，因此最终由助手按公开的逐题5分量表复核，得分为各子项实得分之和，详见correctness_reviewed.json。这是助手复核，不是独立人工评分；量表是在查看实际输出后制定，未预注册。旧T3均分{num(summary['t3_correctness_mean'])}/5、新单题模型均分{num(summary['fresh_model_correctness_mean'])}/5均保留作评分敏感性对照，不混同最终量表分。字符ROUGE-L均值为{num(overall['rouge_l_char'])}，仅作辅助。
- 忠实度先做模型逐句核验，再由助手对20条实际回答逐句复核并拆解事实、剔除题设与纯建议、核对条号年份。忠实度复核只使用实际召回片段、问题条件和待评答案，不使用gold_answer。最终统计来自faithfulness_reviewed.jsonl，明确属于助手复核，不是独立人类标注。原始模型记录完整保留。

## 指标口径

Recall@k按作业FAQ：某题任一gold片段进入前k即记1。另报标准gold覆盖比例。Precision@k=前k中标注相关片段数/k。MRR=首个gold排名倒数的题均值；AP=各gold命中位置的Precision均值，分母为该题所有标注gold数；MAP为AP题均值。全库指标使用完整BM25排名，@10版本截断前10，未命中的相关片段仍计入AP分母。

每个事实按1–5分：5直接支持；4可由片段与给定条件完整推导；3关键条件/范围仅部分支持；2未找到依据；1与证据矛盾。题内事实均分再按题平均得到忠实度。纯建议、假设、标题和纯拒答不进入事实分母；没有事实时指标为N/A，不记满分。

按作业的严格定义，未被当前Context完整支持（含部分支持、无依据、矛盾）的声明均计入幻觉分子。这不意味着它一定在现实中错误。微平均幻觉率是全体无依据事实数/全体事实数；题均比例另存在JSON中，不能混用两种平均。

{summary['evidence_link_rechecks']}条“模型判有依据却漏填证据句号”的记录进行了单独证据定位复核。原始模型共提取{summary['raw_fact_count']}个事实并判{summary['raw_unsupported_fact_count']}个未支持；证据定位后为{summary['automatic_link_checked_unsupported_count']}/{summary['automatic_link_checked_fact_count']}。逐句复核发现模型还会把标题扩写成事实、漏掉错误文号、把建议和题设计为幻觉，因此最终使用公开复核表的{overall['fact_count']}个事实，未对原始答案或召回做改写。拆分粒度与分母变了，两组比率不能解释为系统性能提升。复核表生成器tools/build_reviewed_facts.py逐项列明原句编号、判断理由与真实证据句号，可由提交者继续检查。

## 检索模块结果

{k_table}

![检索k值](results/t7/figures/retrieval_k.png)

![分层检索](results/t7/figures/stratified_retrieval.png)

严格gold编号未穷尽所有等价重叠片段，因此P@k、MAP和gold覆盖可能低估部分真实有用片段。归因时会检查原文，不能仅凭gold编号缺失判定检索失败。最终生成始终使用top-5，k曲线不代表重新生成了四组答案。

## 生成模块与分层结果

{layer_table}

![分层生成](results/t7/figures/stratified_generation.png)

![分层幻觉](results/t7/figures/stratified_hallucination.png)

每级只有5题。长答案包含更多事实，因此微平均幻觉率对事实较多的问题权重更大；不能把它读成“有多少题答错”。截断同时影响完整性，正确但漏掉关键限制也会降低正确性。忠实度高的拒答不等于任务完成，必须结合正确性看。

## 最差五题与改进

按“正确性升序→忠实度升序→幻觉率降序→题号”固定规则选题，完整逐题复核见《T7失败案例分析.md》。

{failure_table}

归因区分：必要证据未进入候选20为检索失败；候选已找到却未进入top-5为排序失败；证据已经给出但回答错误、无依据扩展或被预算截断为生成失败；问题/金标或等价片段标注不清为评测集歧义。最后结论依据实际文本复核，不机械使用上述编号启发式。改进方案是下一轮可验证的建议，不冒称已完成改造。

系统效率方面，本次没有把跨条件缓存复用的旧计时冒充新延迟基准；仅报告真实token数和截断情况。多模态扩展性没有在纯文本快照上测试；无法据此推断图片、复杂表格或OCR的处理能力。

## 复现与局限

完整20题表见《T7完整评估表.md》，逐原子事实及证据在results/t7/claim_evaluations.json。运行命令、固定输入和模块对应关系见README_T7.md。5项指标/证据检查已通过，源代码和数据摘要随包保存。

主要局限为20题小样本、同模型生成与自评偏差、助手事后制定的正确性量表及事实拆分和支持判断的主观性、金标重叠片段不完备及原输出预算截断。助手做了全部回答的证据复核及失败案例分析，不冒称教师或其他独立人工审核；引用句号真实存在也不自动保证语义蕴含正确。最终正确性属于助手量表评估；建议提交者重点复核低分和政策细节。报告不声称显著性、生产可靠性或其他版本同样表现。

## 方法与来源

- 课程实践作业T7及课件9.5为验收依据，主评对象由用户明确选择。
- 检索指标：[Stanford IR教材](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)。
- 原子事实与Context一致性的思路参考[Ragas Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)。本实现按课程使用1–5分，并未直接运行Ragas包，不能把分数当作其官方0–1指标。
- 基础项目及评测数据来源sunfeiyu666/LLM-RAG与本地T3快照；新增评估与报告有大模型辅助，模型调用真实发生，结果未编造。
'''
    (ROOT/'T7系统评估报告.md').write_text(report,encoding='utf-8')
    print(json.dumps({'complete':True,'overall':overall,'worst_ids':[r['id'] for r in worst]},ensure_ascii=False))


if __name__=='__main__':main()
