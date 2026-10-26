"""Publish T7 tables, stratified charts, failure analyses and updated report."""
import json,csv,sys,os,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from t7_updated_core import UP,OUT,KS,read_jsonl,csv_rows,aggregate,mean,validate_review,sha

def pct(v):return f'{v:.2%}' if v is not None else 'N/A'
def num(v):return f'{v:.3f}' if v is not None else 'N/A'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v).replace('\n',' ') for v in r)+' |' for r in rows])

def plots(summary):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    levels=['L1','L2','L3','L4'];x=list(range(4))
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    for i,(key,label,color) in enumerate([('recall_at_3','实际top-3 Recall','#0072B2'),('precision_at_3','实际top-3 Precision','#D55E00'),('mrr_at_10','MRR@10','#009E73'),('map_at_10','MAP@10','#CC79A7')]):
        ax.bar([v+(i-1.5)*.18 for v in x],[summary['by_level'][l][key] for l in levels],.18,label=label,color=color)
    ax.set(xticks=x,xticklabels=[l+'\n(n=5)' for l in levels],ylim=(0,1.18),ylabel='指标值',title='更新版T7：分层检索结果')
    ax.legend(loc='upper right',ncol=2,frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(directory/'stratified_retrieval.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.1),layout='constrained')
    settings=[('upstream_accuracy_0_2','准确性：上游项目复核','0–2分',2.35,'#0072B2'),
              ('rouge_l_char','正确性辅助：字符ROUGE-L','0–1',1.15,'#D55E00'),
              ('faithfulness_1_5','忠实度：本轮助手证据评估','1–5分',5.8,'#009E73')]
    for ax,(key,title,ylabel,maximum,color) in zip(axes,settings):
        vals=[summary['by_level'][l][key] for l in levels];bars=ax.bar(levels,vals,color=color,width=.58)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+maximum*.02,f'{v:.2f}',ha='center')
        ax.set(ylim=(0,maximum),title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('更新版T7：分层生成指标（不同量尺分开绘制）')
    fig.savefig(directory/'stratified_generation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    vals=[summary['by_level'][l]['hallucination_micro'] for l in levels];bars=ax.bar(levels,vals,color='#D55E00',width=.55)
    for b,l,v in zip(bars,levels,vals):
        s=summary['by_level'][l]
        ax.text(b.get_x()+b.get_width()/2,v+.025,f"{s['unsupported_fact_count']}/{s['fact_count']}\n{pct(v)}",ha='center')
    ax.set(ylim=(0,1.04),title='更新版T7：严格幻觉率',ylabel='未完整支持事实 / 全部事实')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(directory/'stratified_hallucination.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    for name,label,color,mark in [('recall','任一gold命中Recall','#0072B2','o'),('precision','Precision','#D55E00','s'),('gold_coverage','gold覆盖','#009E73','^')]:
        ax.plot(KS,[summary['overall'][f'{name}_at_{k}'] for k in KS],marker=mark,color=color,label=label)
    ax.axvline(3,color='#666666',ls=':',lw=1);ax.text(3.1,.45,'生成实际使用top-3',fontsize=9)
    ax.set(xticks=KS,ylim=(0,1.12),title='更新版T7：BM25不同k值的检索指标',xlabel='k（答案只使用前三块，其他k仅诊断）',ylabel='比例')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='center right',frameon=False);ax.grid(alpha=.15)
    fig.savefig(directory/'retrieval_k.png',dpi=180);plt.close(fig)

def main():
    metrics=read_jsonl(OUT/'retrieval_metrics.jsonl');faith={r['id']:r for r in read_jsonl(OUT/'faithfulness_reviewed.jsonl')}
    packets={r['id']:r for r in read_jsonl(OUT/'faithfulness_packets.jsonl')}
    qs={r['id']:r for r in read_jsonl(ROOT/'data/qa_set.jsonl')};answers={r['id']:r for r in read_jsonl(ROOT/'data/system_answers.jsonl')}
    assert len(metrics)==len(faith)==len(packets)==20
    rows=[]
    for m in metrics:
        f=faith[m['id']];validate_review(f,packets[m['id']])
        rows.append({**m,**{k:f[k] for k in ['fact_count','unsupported_fact_count','faithfulness_1_5','hallucination_rate']}})
    summary={'target':'T4/T5 BM25 -> direct top-3 -> no compression','n':20,
        'overall':aggregate(rows),'by_level':{l:aggregate([r for r in rows if r['level']==l]) for l in ['L1','L2','L3','L4']},
        'faithfulness_assessor':'Codex assistant source-only claim assessment in this conversation; not independent human annotation.',
        'correctness_method':'Character ROUGE-L and normalized EM recalculated; upstream project accuracy 0-2 retained without rescaling.',
        'faithfulness_protocol':'1-5 atomic-claim source support. Partial support counts unsupported. Repeated factual assertions count per occurrence; advice/givens/headings/refusal excluded.',
        'context_reconstruction':'Upstream saved IDs + exact formatter + unchanged chunk texts; per-question raw content character counts match. Original full Context string and stop reasons were not stored.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'per_question.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['id','level','mrr_full','map_full','mrr_at_10','map_at_10']+[f'{m}_at_{k}' for m in ['recall','precision','gold_coverage'] for k in KS]+[
        'upstream_accuracy_0_2','upstream_citation_0_2','rouge_l_char','exact_match','faithfulness_1_5','fact_count','unsupported_fact_count','hallucination_rate','context_tokens','generation_output_tokens','total_seconds']
    with (ROOT/'T7完整评估表_更新版.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    full_table=table(['题号','等级','R@3','P@3','MAP@10','正确性ROUGE-L','上游准确性/2','忠实度/5','未支持/事实','幻觉率'],[
        [r['id'],r['level'],pct(r['recall_at_3']),num(r['precision_at_3']),num(r['map_at_10']),num(r['rouge_l_char']),r['upstream_accuracy_0_2'],num(r['faithfulness_1_5']),f"{r['unsupported_fact_count']}/{r['fact_count']}",pct(r['hallucination_rate'])] for r in rows])
    (ROOT/'T7完整评估表_更新版.md').write_text('# T7更新版：完整20题评估表\n\n'+full_table+'\n\n完整MRR/MAP、k=1/3/5/10、EM及token/耗时见同名CSV和results/t7/per_question.json。ROUGE-L不是准确率；上游准确性采用0–2，不与旧版0–5均分直接比较。忠实度由本轮助手按实际Context逐事实评估，非独立人工评分。\n',encoding='utf-8')
    plots(summary)
    reviews=json.loads((OUT/'failure_reviews.json').read_text(encoding='utf-8'))
    expected=sorted(rows,key=lambda r:(r['upstream_accuracy_0_2'],r['upstream_citation_0_2'],r['rouge_l_char'],r['id']))[:5]
    assert [r['id'] for r in reviews]==[r['id'] for r in expected]
    byrow={r['id']:r for r in rows}
    case_lines=['# T7更新版：最差五题归因与改进',
        '选题规则在本轮忠实度评分前固定：上游准确性0–2升序、上游引文支撑0–2升序、字符ROUGE-L升序、题号。不是按助手希望展示的故事挑选。上游分数属于项目复核，非独立人工评测。',
        '失败分类以本次实际候选5/最终3为界：所需证据排在5之外是检索失败；在候选5但未进3是排序/截断失败；给定证据被误用或越界断言是生成失败；编号未穷尽等价文本时须检查评测集歧义。一个问题可有多个原因。']
    for review in reviews:
        qid=review['id'];r=byrow[qid];q=qs[qid];a=answers[qid]
        assert review['answer_excerpt'] in a['answer']
        case_lines += [f"## {qid} · {q['level']}",'**问题：**'+q['question'],'**参考答案：**'+q['gold_answer'],
            f"**指标：**上游准确性{r['upstream_accuracy_0_2']}/2；ROUGE-L {num(r['rouge_l_char'])}；忠实度{num(r['faithfulness_1_5'])}/5；严格幻觉率{pct(r['hallucination_rate'])}。",
            '**主归因：**'+review['primary_cause']+'；次要因素：'+review['secondary_factors'],
            '**错误或缺漏原文：**\n\n> '+review['answer_excerpt'].replace('\n','\n> '),
            '**候选top-5：**'+', '.join(r['candidate_top5']),'**实际top-3：**'+', '.join(r['actual_top3']),
            '**复核解释：**'+review['analysis']]
        for e in review['evidence']:
            location='实际Context' if e['in_actual_top3'] else ('候选内但未进入实际Context' if e['in_candidate_top5'] else '候选外，仅供事后定位')
            case_lines += [f"**原文证据 {e['chunk_id']}（排名{e['rank']}，{location}）：**\n\n> "+e['quote'].replace('\n','\n> ')]
        case_lines += ['**改进方案：**'+review['improvement'],'**验证办法：**'+review['verification_plan'],
            '**完整原始回答：**\n\n'+a['answer']]
    (ROOT/'T7失败案例分析_更新版.md').write_text('\n\n'.join(case_lines)+'\n',encoding='utf-8')
    overall=summary['overall'];by_level=summary['by_level']
    ktable=table(['k','任一gold Recall@k','Precision@k','全部gold平均覆盖'],[[k,pct(overall[f'recall_at_{k}']),num(overall[f'precision_at_{k}']),pct(overall[f'gold_coverage_at_{k}'])] for k in KS])
    ltable=table(['等级','题数','上游准确性/2','ROUGE-L','忠实度/5','未支持/事实','幻觉率'],[
        [l,r['n'],num(r['upstream_accuracy_0_2']),num(r['rouge_l_char']),num(r['faithfulness_1_5']),f"{r['unsupported_fact_count']}/{r['fact_count']}",pct(r['hallucination_micro'])] for l,r in by_level.items()])
    t4rows=csv_rows(UP/'results/t4_bge_20261025/t4_metrics.csv')
    t4table=table(['方法','Recall@1','Recall@3','Recall@5','MRR@10'],[[r['method'],r['recall@1'],r['recall@3'],r['recall@5'],r['mrr@10']] for r in t4rows])
    perf=csv_rows(UP/'results/t5_final_20261026/t5_full_summary.csv');quality=csv_rows(UP/'results/t5_final_20261026/t5_quality_summary.csv')
    # Upstream score CSV schemas are kept visible rather than guessed or rescaled.
    quality_keys=list(quality[0])
    (OUT/'upstream_quality_columns.json').write_text(json.dumps(quality_keys,ensure_ascii=False),encoding='utf-8')
    ptable=table(['排序','压缩','Context token/题','端到端秒/题','任一证据@3','全部证据@3'],[
        [r['strategy'],r['compressed'],f"{float(r['mean_context_tokens']):.2f}",f"{float(r['mean_total_seconds']):.2f}",r['any_gold_at_3'],r['all_gold_at_3']] for r in perf])
    failures=table(['题号','等级','上游准确性/2','主归因'],[[r['id'],qs[r['id']]['level'],byrow[r['id']]['upstream_accuracy_0_2'],r['primary_cause']] for r in reviews])
    manifest=json.loads((ROOT/'target_manifest.json').read_text(encoding='utf-8'))
    at_limit=sum(r['reported_output_at_limit'] for r in rows)
    report=f'''# T7系统评估报告（按T2/T4/T5更新）

## 评估对象与主要结论

本轮评估上游T4/T5选定的 **新版BM25候选top-5 → 直接取top-3 → 不压缩 → Qwen回答**。复用同学在2026-10-26实际生成并保存的20条回答；重新计算检索、正确性辅助指标，逐事实做本轮助手忠实度评估，不重新生成答案，也不沿用旧T7分数。

MRR={num(overall['mrr_full'])}、MAP={num(overall['map_full'])}；MRR@10={num(overall['mrr_at_10'])}、MAP@10={num(overall['map_at_10'])}。实际top-3的Recall={pct(overall['recall_at_3'])}、Precision={num(overall['precision_at_3'])}、标注gold平均覆盖={pct(overall['gold_coverage_at_3'])}。Recall为100%仍不代表证据齐全。

字符ROUGE-L均值{num(overall['rouge_l_char'])}、EM={pct(overall['exact_match'])}；上游项目复核准确性{num(overall['upstream_accuracy_0_2'])}/2、引文支撑{num(overall['upstream_citation_0_2'])}/2，保留原量尺，不换成旧版0–5。本轮题均忠实度{num(overall['faithfulness_1_5'])}/5；未完整支持事实{overall['unsupported_fact_count']}/{overall['fact_count']}，严格幻觉率{pct(overall['hallucination_micro'])}。

综合最弱的是 **L4**：上游准确性{num(by_level['L4']['upstream_accuracy_0_2'])}/2、幻觉率{pct(by_level['L4']['hallucination_micro'])}。L3与L4忠实度都约4.04，差不足0.001，不按这个微小差值宣称某一级显著更差。

## 一、更新来源与系统选择

- 主分支/T2快照：`{manifest['main_t2_commit']}`，T2已合并主分支。
- T4/T5快照：`{manifest['t4_t5_commit']}`，来源t4-t5-zhuxiangduo分支。源代码、结果和SHA256清单在upstream/及upstream_manifest.json中，未修改原文件。
- 正式评测集SHA256：`{manifest['qa_sha256']}`，L1–L4各5题，合计20题。
- 本轮生成模型摘要：`{manifest['model_digest']}`。

T2做了12组分块检索实验；记录显示generated_answers=false，answer_correctness实际标为retrieval_evidence_proxy，不能当作真实生成答案准确性。T2中sliding_1024和small_to_big_1024的检索指标均为0.95，但T4/T5正式实验仍使用原650字符、重叠120的202块。评测口径也不同：T2优先用证据原文匹配，T4按gold片段ID。不能直接拼接成“采用T2最优分块后的T4/T5最终系统”。本轮评估真实存在的202块链路；若组内决定整合T2最佳分块，必须重建证据映射、生成新回答并另评。

T4新增BGE-small-zh-v1.5神经向量检索，结果如下（按上游保存的正式排名复算核对）：

{t4table}

T4据Recall和首次命中排名选择BM25。T5在同一候选5上比较直接、规则、Cross-Encoder排序及压缩前后六组；上游准确性最高的是直接不压缩1.55/2，因此在新忠实度评分前就固定它为主评对象。选择与评估仍使用同一20题，不是独立留出验证；本报告不声称已经完成全小组所有模块的最终工程整合。

## 二、相对旧T7具体改变

|项目|旧T7|本轮更新|
|---|---|---|
|BM25实现|T3的单字加双字方案|T4 retrieval_compare.py的中文双字为主、查询词去重方案|
|最终Context|top-5|候选5直接取3、不压缩|
|生成预算|上下文8192，输出768|上下文16384，输出2048|
|真实回答来源|本地T3 rrf_0|同学T5 direct/False的20条回答|
|正确性|旧助手0–5量表|重算ROUGE-L/EM，保留上游0–2评分，不混量尺|
|忠实度与案例|针对旧回答的核验|针对新回答重新逐事实评估、重新选最差5题|

模型温度0.2、seed42，23篇语料和202块文本经比对保持一致。新版BM25排名已重算，并与T4保存的20题前10全部一致。T5中每题selected_chunk_ids等于新BM25前3，20题正文字符数均与记录一致。

**Context重建限制：**上游未直接保存完整Context字符串。本轮依据冻结代码的format_context、相同片段文本和实际selected_chunk_ids重建，并核对逐题正文字符数。token数采用上游记录，未另装tokenizer重算；不能把这说成与当时原始字符串作了直接逐字比对。重建的Context及SHA256随data/system_answers.jsonl保留。上游还未保存finish_reason；{at_limit}/20条输出token达到2048上限，只能说未观察到token触上限，不能声称已独立证明零截断。

## 三、指标与评估方法

检索Recall@k按课程FAQ：任一标注gold片段进入前k记1；另报gold覆盖=命中gold数/全部gold数。Precision@k=前k标注相关数/k。MRR=首个相关片段排名倒数的题均值；AP=各相关命中位置Precision之和/该题全部gold数，MAP为AP题均值。全库指标用完整202排名，@10截断后仍把未命中gold保留在AP分母。

生成正确性重算字符级ROUGE-L F1与归一化EM；两者均与gold_answer对比。ROUGE-L反映文字重合，不是事实准确率；长建议、正确简答和参考答案的额外提醒都会影响它，EM为0也不意味着20题全错。上游准确性及引文支撑0–2仅为项目复核参考，未冒称独立人工标注，也没有线性放大成旧0–5。

忠实度由本对话Codex助手作为LLM逐事实评估：5=实际Context直接支持；4=结合题设可完整推导；3=仅部分支持或范围扩大；2=没有足够依据；1=与实际证据矛盾。每题事实均分，再按题平均。纯建议、学习目标、题设、标题、引用标号、纯拒答不计事实；混合建议中的政策、数据或官方来源归属断言仍计入。重复出现的事实按每次声明计数，无事实回答记N/A而不记满分。

严格幻觉率=未被实际Context完整支持的事实/全部事实，故3分部分支持也入分子。它不等于现实中一定错误，更不等于答错题目比例。微平均按事实数汇总，题均比例另存JSON；L4长回答对微平均权重更大。

证据判定仅以实际top-3和题设为依据，不拿全库或gold补足未召回证据。来源专用packet不含gold；但本对话助手了解任务与评测背景，**不冒称盲评或独立人工审核**。本轮未额外调用本地Qwen评分器。全部213项声明、分数、原句编号、真实证据原文和理由公开；所有回答单元均有事实/排除记录。助手语义判断与拆分仍需提交者复核，代码校验只能保证数据和引文一致，不能替代独立语义评价。

## 四、检索结果

{ktable}

![检索k曲线](results/t7/figures/retrieval_k.png)

![分层检索](results/t7/figures/stratified_retrieval.png)

答案实际只看top-3，k=5/10是诊断，不代表重新生成了这些k值的答案。严格gold编号没有穷尽重叠等价片段，MAP、Precision和覆盖可能低估有用证据；失败归因必须核对原文。例如q009虽缺某个辅修通知gold，实际仍有学籍规定中的部分等价制度，不能机械地把全部错误归为检索失败。

## 五、生成与L1–L4分层

{ltable}

![分层生成](results/t7/figures/stratified_generation.png)

![分层幻觉率](results/t7/figures/stratified_hallucination.png)

L4需要跨条款、跨业务及版本匹配，并区分政策事实与自主建议，错误最多。q006可在证据不足时谨慎拒答，因此忠实度不一定低，但没有完成10/15/差5的任务；q018长路线涵盖不少真实主题，却把自主练习归为官方路径。两例说明正确性和忠实度必须分开报告。

每层只有5题，L1一个错误条号就会显著影响事实比例；不能从这些比例推断普遍可靠性，也不做统计显著性结论。新旧T7配置、回答和评分量尺不同，不能把均分变化单独归因于BM25、top-k或某一模块。

## 六、最差五题归因

选题在忠实度评分前固定：上游准确性升序 → 上游引文分升序 → ROUGE-L升序 → 题号。详细原回答、gold、实际候选5/最终3、正确片段排名、字面证据及改进验证方案见《T7失败案例分析_更新版.md》。

{failures}

本轮候选池是5，不是旧T3的20；“排序失败”包括候选已有但直接取3时被截掉，并不意味着本主评配置实际启用了重排模型。全库事后定位的正确证据明确标注为未进入Context，绝不用于抬高忠实度。

## 七、系统效率与多模态边界

以下是同学T5正式六组运行的原始记录重算，**不是在本机新测速度**。主评组平均Context为{overall['context_tokens']:.2f} token，端到端{overall['total_seconds']:.2f}秒/题。原T5记录单独warmup约48.86秒，不能把它当某一题处理耗时。旧T3缓存计时不作直接对照。

{ptable}

上游项目准确性均分：直接不压缩1.55/2，直接压缩1.35/2；规则不压缩1.40/2，模型不压缩1.35/2。因此压缩省token但有质量代价，重排也未获得整体质量收益。本轮据此选择直接不压缩，不为展示优化而默认堆叠模块。重排/压缩前后的片段ID可能不变，但删除关键句仍会损害答案，ID覆盖不能替代忠实度。

![上游T5质量与延迟](upstream/t4_t5/results/t5_final_20261026/t5_latency_quality.png)

本次主评为文本链路。部分原资料源自PDF/表格，但进入检索的是转换后的Markdown，不代表已经验证图文混合检索、复杂表格或OCR准确性。q018跨页内容遗漏说明需要文档结构与续页覆盖；对未来多模态扩展，应保留表头、整行、页号和条款关系，再单设含图表的评测集。本报告没有虚构多模态性能指标。

## 八、交付、复现与局限

交付包括20题完整表（MD/CSV/JSON）、4张新图、213事实及证据、5个失败案例、上游版本快照、评分表和复现校验。README_T7更新版.md给出命令。复现是从固定源记录重算排名、Context、指标和公开判断表，不等于重新独立采样或重新生成20答案；改变系统后必须重新评。

主要限制：20题小样本、同评测集选配置与评估的偏乐观风险、上游正确性评分主观性、助手事实拆分与支持判断主观性、gold重叠标注不完备，以及原始Context/停止原因未直接保存。T2最佳分块和T3优化尚未构成一条经本轮运行验证的整合链路。旧T7保留为历史，不能与本轮混表。提交者应重点复核政策年份、年级、条号、数字、拒答和官方/建议边界。

课程验收依据为《检索增强生成-课程实践作业》T7：检索四类指标、生成三类指标、L1–L4柱状图及最差5题归因改进。来源为仓库固定提交和用户提供的任务要求；实现、逐事实评估及报告有大模型辅助，未冒称独立人工评审。
'''
    (ROOT/'T7系统评估报告_更新版.md').write_text(report,encoding='utf-8')
    print(json.dumps({'complete':True,'overall':overall,'worst_ids':[r['id'] for r in expected]},ensure_ascii=False))

if __name__=='__main__':main()
