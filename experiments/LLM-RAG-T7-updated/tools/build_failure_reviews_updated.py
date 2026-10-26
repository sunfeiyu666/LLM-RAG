"""Five failures chosen by a published upstream-score/ROUGE ordering."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from t7_updated_core import OUT,read_jsonl
from review_faithfulness_updated import literal_quote

def main():
    answers={r['id']:r for r in read_jsonl(ROOT/'data/system_answers.jsonl')}
    metrics={r['id']:r for r in read_jsonl(OUT/'retrieval_metrics.jsonl')}
    chunks={r['metadata']['chunk_id']:r for r in json.loads((ROOT/'data/frozen_chunks.json').read_text(encoding='utf-8'))}
    def e(qid,cid,anchor):
        quote=literal_quote(chunks[cid]['page_content'],anchor);m=metrics[qid]
        return {'chunk_id':cid,'quote':quote,'rank':m['ranking'].index(cid)+1,
                'in_candidate_top5':cid in m['candidate_top5'],'in_actual_top3':cid in m['actual_top3']}
    reviews=[
    {'id':'q016','primary_cause':'生成失败','secondary_factors':'微专业日期/资格段排12，证书段排5未进入最终top-3；生成仍把辅修业务信息套入微专业',
     'answer_excerpt':'于2026年5月21日至26日登录“云南大学网上办事服务大厅”“方案中心”模块，选择“微专业办理”，提交微专业申请[1]。',
     'evidence':[e('q016','ynu029_p000_00190','选择“辅修办理”'),e('q016','ynu030_p000_00196','于6月11日至15日'),e('q016','ynu030_p000_00196','2025级、2024级在籍在校本科学生均可申请'),e('q016','ynu030_p000_00198','微专业不授予学位')],
     'analysis':'回答把top-3中辅修通知的5月时段和方案中心入口用于微专业，并另造微专业办理名称。微专业真正的6月日期及年级资格段排12，未进候选5；证书不授学位段排5，虽进候选却被直接取3截掉。缺证据时本应标记待核实，而不是借用另一业务。回答还把学分互认扩大成避免额外负担，把仅限新开设微专业的人数条件泛化为任意微专业。上游给本题0/2，不意味着每个事实都错：15/5认定及冲突处理等内容仍有实际证据。',
     'improvement':'按辅修/微专业分别绑定业务、年份、申请入口及日期，缺字段时明确未查到；引入同通知相邻片段扩展，优先保留申请资格与证书规则；建议部分先核对主修缺口、收费及课表，不承诺减负。',
     'verification_plan':'记录微专业日期段能否进入候选和实际Context、业务日期错绑次数、无依据减负承诺数；在原20题上同时检查要点覆盖、token和延迟，避免只展示本题。'},
    {'id':'q018','primary_cause':'检索失败','secondary_factors':'大纲后半排53，未进候选5；生成把自主练习说成官方学习路径，并添加未支持的章节细节',
     'answer_excerpt':'该阶段内容完全来源于课程大纲[1]，属于课程明确设定的学习路径，可直接依据执行。',
     'evidence':[e('q018','ynu006_p000_00088','本课程要求的先修课为数据结构'),e('q018','ynu006_p000_00089','8.1事务的ACID性质介绍'),e('q018','ynu006_p000_00089','8.6事务恢复'),e('q018','ynu006_p000_00089','9.2Java访问数据库')],
     'analysis':'实际Context只有公开课前半与两份培养方案片段，真正包含事务ACID、调度封锁、恢复和Java访问数据库的大纲续块排53。长回答大量展开前六章，却漏掉参考路线要求的可靠性与集成部分。生成又把MySQL/SQLite验证练习说成大纲明确设定的路径；大纲仅给主题，不给这些指定项目。个人建议本身可以合理，错误在于把建议归为官方事实。本题无finish_reason可直接核验，保存的输出token未触2048上限，不能继续套用旧版768截断解释。',
     'improvement':'命中课程大纲后按文档ID扩展完整章节，再以基础、可靠性、应用集成建立覆盖清单；将来源明确标成大纲主题/自主练习，禁止所有路径均来自大纲的笼统背书。',
     'verification_plan':'检查续块是否进入实际Context，以及事务、并发、恢复、集成四类覆盖；逐项核验章节来源和建议标签。另记录输出长度与真实停止原因，不能仅靠长回答或token未超限判断完成质量。'},
    {'id':'q020','primary_cause':'检索失败','secondary_factors':'2027续页排19，缺课程硬限制和排序/学院加标条款；生成仍有范围扩大、背景臆断及版本概括错误',
     'answer_excerpt':'所有现有资料均基于2025年版本，且未提供2027年软件学院推免实施细则的更新内容。',
     'evidence':[e('q020','ynu018_p006_00138','平均学习绩点（GPA）不低于3.0'),e('q020','ynu018_p007_00139','专业必修课不得有补考、重修情况'),e('q020','ynu018_p007_00139','以专业或专业方向为单位'),e('q020','ynu018_p007_00139','学院可结合自身情况'),e('q020','ynu018_p006_00138','不得单列名额进行选拔')],
     'analysis':'2027推荐条件续页排19，未进BM25候选5，实际Context因此缺不得补考重修的完整续句、专业排序及学院加标规则。生成同时漏掉已经给出的科研竞赛不得单列名额，故不能把全部遗漏归给检索。实际第1及第3块都是2027学校文件，回答却概括所有资料都基于2025；正确说法应区分2027学校办法与缺失的2027学院细则。普通专业身份也不能推出一定非联合培养。学院受奖励不能推出学生个体绝不间接受益。',
     'improvement':'先把学校门槛、课程限制、科研计分、排序、学院细则与名额列成证据槽，再按同版本续页补齐；只对已知身份作判断，按学校/学院分层说明年份；建议保留必要边界，不列未经证据验证的竞赛认可名单。',
     'verification_plan':'统计六个证据槽是否齐全，核对不得单列名额是否被保留、是否仍出现所有资料2025和无依据个人背景；同时报告原20题整体效果。'},
    {'id':'q006','primary_cause':'排序失败','secondary_factors':'直接top-3截掉候选中排5的第八条；年级适用范围排30未进候选；回答部分合理拒答但任务未完成',
     'answer_excerpt':'资料不足，无法准确回答“后者比前者多多少”。',
     'evidence':[e('q006','ynu010_p000_00111','累计认定不超过10学分'),e('q006','ynu010_p000_00112','第八条学生辅修专业或学位、修读微专业'),e('q006','ynu010_p000_00116','本规定自2025级学生开始执行')],
     'analysis':'直接组确实召回第4条10学分，但第8条15学分排5，已经进入BM25候选5，最终直接取3时丢失。因此主要是排序及截断未覆盖第二子意图，而不是知识库没有此规则。当前实际Context只有2024级微专业15/5及2025另按新规的转引，回答谨慎指出资料不足在忠实度上可以得分，但没能完成10、15和差5的任务，正确性仍低。答复中把辅修未给上限进一步概括成辅修未被提及，范围也过宽。',
     'improvement':'采用两类认定信息槽的覆盖约束，从候选5中分别保留第4和第8条；按文档内转引补年级适用范围，禁止把缺上限说成没有认定制度。可与直接增加top-k对比，避免无上限增加Context。',
     'verification_plan':'先固定候选5，仅改选取规则，检查第8条进top-3及10/15/5回答覆盖，单列Context token和延迟；再独立测试文档内适用范围补全，区分排序收益和召回收益。'},
    {'id':'q019','primary_cause':'检索失败','secondary_factors':'第八条排12、适用范围排18，均在候选5外；生成又把2024条款改成2025，并混淆认定总量与主修课程总量',
     'answer_excerpt':'根据《2026年秋季学期微专业申请通知》第二条，2025级学生获得的微专业课程学分可同时认定为主修专业的选修课程学分，但最多不超过15学分，其中专业教育不超过5学分[3]。',
     'evidence':[e('q019','ynu030_p000_00197','2024级学生获得的微专业课程学分'),e('q019','ynu030_p000_00197','2025级学生按'),e('q019','ynu010_p000_00112','第八条学生辅修专业或学位、修读微专业'),e('q019','ynu010_p000_00116','本规定自2025级学生开始执行'),e('q019','ynu010_p000_00115','上述各类情况认定为选修课程的学分')],
     'analysis':'实际Context给出了转专业10/5及各类认定原则总计20/5，但缺辅修类别15/5的新规第8条及第15条适用年级。两块分别排12、18，均未进候选5，主要是召回不足。生成仍可依据当前材料给去重和等效审核建议，却不能把微专业通知中的2024直接改为2025。15数额在完整新规中可能相同，不等于这个引用链正确。20是上述各类选修认定的原则总量，不是主修培养方案全部选修课程的上限。',
     'improvement':'沿2025按新规的明确转引回查第8及第15条；以类别上限、专业子上限、合并总量、适用年级四列建立规划表。让模型区分主修课程学分与外部课程认定额度，对未提供课程清单保持不确定性。',
     'verification_plan':'检查4个信息槽、年份及年级引用链；为每个数额记录适用对象与条号，统计将20错当主修选修总量的次数；再核对原20题整体指标及开销。'}]
    selected=json.loads((OUT/'worst_five_preselected.json').read_text(encoding='utf-8'))
    assert [r['id'] for r in reviews]==[r['id'] for r in selected]
    for r in reviews:
        assert r['answer_excerpt'] in answers[r['id']]['answer'],r['id']
        for evidence in r['evidence']:assert evidence['quote'] in chunks[evidence['chunk_id']]['page_content']
    (OUT/'failure_reviews.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FAILURE_REVIEWS_VALIDATED',len(reviews))

if __name__=='__main__':main()
