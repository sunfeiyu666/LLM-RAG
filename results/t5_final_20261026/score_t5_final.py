"""将逐题核对后的准确性与引文支撑度评分绑定到本次原始答案。"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = (("direct", False), ("direct", True), ("rule", False),
         ("rule", True), ("model", False), ("model", True))

# 每题六组，顺序见 ORDER；每组 (答案准确性, 引文支撑度)，均为 0—2 分。
# 2=关键结论正确且完整/引用直接支持；1=部分正确/部分支持；0=主要结论错误或无有效支持。
SCORES = {
    "q001": ((2, 2),) * 6,
    "q002": ((2, 2),) * 6,
    "q003": ((2, 2),) * 6,
    "q004": ((2, 2),) * 6,
    "q005": ((2, 2), (2, 1), (2, 2), (2, 2), (2, 2), (2, 2)),
    "q006": ((1, 1), (1, 1), (2, 2), (2, 2), (1, 1), (0, 1)),
    "q007": ((1, 1), (1, 1), (0, 0), (0, 0), (1, 1), (1, 1)),
    "q008": ((2, 2), (2, 2), (0, 1), (0, 1), (1, 1), (1, 1)),
    "q009": ((1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (1, 1)),
    "q010": ((2, 2),) * 6,
    "q011": ((2, 2), (0, 0), (2, 2), (0, 0), (2, 2), (0, 0)),
    "q012": ((2, 2), (2, 2), (2, 1), (2, 2), (2, 2), (2, 2)),
    "q013": ((2, 2), (0, 0), (2, 2), (0, 0), (2, 2), (0, 0)),
    "q014": ((2, 2),) * 6,
    "q015": ((1, 1), (0, 1), (2, 2), (2, 1), (1, 1), (1, 1)),
    "q016": ((0, 1), (1, 1), (0, 1), (1, 1), (0, 1), (1, 1)),
    "q017": ((2, 2), (2, 2), (2, 2), (2, 2), (1, 1), (1, 1)),
    "q018": ((1, 1),) * 6,
    "q019": ((1, 1), (1, 1), (0, 1), (1, 1), (0, 1), (1, 1)),
    "q020": ((1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0)),
}

REASONS = {
    "q001": "按指定版本给出第一课堂161.5学分，引用对应培养方案。",
    "q002": "给出先修课数据结构，引用公开慕课页面。",
    "q003": "给出缓考一般不超过两门，引用2020版原文。",
    "q004": "给出前三学年GPA不低于3.0，引用2027年学校办法。",
    "q005": "不授学位与修读证书均正确；压缩后的直接组引文未指向证书条款。",
    "q006": "规则组取得第八条，正确区分10和15学分；其他组缺少该条或误用2024级通知。",
    "q007": "缺少微专业6月11—15日窗口；规则组两项时间都未答出。",
    "q008": "直接组取到两份通知并算出0.25学分；重排组丢失辅修34学分条款。",
    "q009": "多数组能区分学位与证书，但缺辅修55学分等条件并扩展法律效力说法；规则未压缩组主结论不清。",
    "q010": "161.5与6相加得167.5，并指出单纯相加不能保证毕业。",
    "q011": "压缩组混淆GPA 2.9与综合测评80分；未压缩组按3.0门槛判断正确。",
    "q012": "专业必修曾补考不满足条件；规则未压缩组引用编号与该句所在块不完全对应。",
    "q013": "未压缩组按前一学期理工科2.8判断；压缩组改按文科或误解年级，主结论错误。",
    "q014": "14人不足15人，不开课、结果无效且下学期重申。",
    "q015": "规则组取到第八条15/5上限；直接与模型组缺该条或误用其他认定类别上限。",
    "q016": "多数组缺辅修/微专业完整证书和费用比较，部分把微专业说成必然更轻或误写申请时间。",
    "q017": "直接与规则组同时得到智慧医疗和智能传播介绍；模型组缺智能传播证据块。",
    "q018": "均未召回课程后半段事务与应用开发证据块，学习路线不完整且部分把建议写成课程要求。",
    "q019": "未召回第八条15学分类别上限，多数组误用2024级通知或把转专业10学分上限套到辅修；仍有部分课程清单规划建议。",
    "q020": "直接与规则组至少说明GPA门槛和课程投入风险，但缺补考重修等完整条件且夹杂旧名额推断；模型组丢失2027推荐条件页，错称GPA门槛不明。",
}


def main():
    path = HERE / "t5_full_rows.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != 120 or len(SCORES) != 20:
        raise ValueError("必须先完成20题×6组的生成与逐题评分")
    expected = {(qid, strategy, compressed) for qid in SCORES for strategy, compressed in ORDER}
    actual = {(row["id"], row["strategy"], row["compressed"]) for row in rows}
    if expected != actual or len(actual) != len(rows) or {row["retrieval"] for row in rows} != {"bm25"}:
        raise ValueError("原始答案的题号、策略、压缩设置或检索器不匹配")
    lookup = {(qid, strategy, compressed): score for qid, scores in SCORES.items()
              for (strategy, compressed), score in zip(ORDER, scores)}
    fields = ["id", "retrieval", "strategy", "compressed", "question", "gold_answer",
              "answer", "selected_chunk_ids", "accuracy_0_2", "citation_support_0_2", "notes"]
    with (HERE / "t5_answer_review.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            key = row["id"], row["strategy"], row["compressed"]
            accuracy, support = lookup[key]
            writer.writerow({"id": row["id"], "retrieval": row["retrieval"],
                             "strategy": row["strategy"], "compressed": row["compressed"],
                             "question": row["question"], "gold_answer": row["gold_answer"],
                             "answer": row["answer"], "selected_chunk_ids": row["selected_chunk_ids"],
                             "accuracy_0_2": accuracy, "citation_support_0_2": support,
                             "notes": REASONS[row["id"]]})
    summary = []
    groups = defaultdict(list)
    for row in rows:
        key = row["id"], row["strategy"], row["compressed"]
        groups[(row["strategy"], row["compressed"])].append(lookup[key])
    for strategy, compressed in ORDER:
        scores = groups[(strategy, compressed)]
        summary.append({"strategy": strategy, "compressed": compressed, "questions": len(scores),
                        "mean_accuracy_0_2": sum(x[0] for x in scores) / len(scores),
                        "full_accuracy_count": sum(x[0] == 2 for x in scores),
                        "mean_citation_support_0_2": sum(x[1] for x in scores) / len(scores),
                        "full_support_count": sum(x[1] == 2 for x in scores)})
    with (HERE / "t5_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (HERE / "quality_review_metadata.json").write_text(json.dumps({
        "raw_rows_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "hash_normalization": "CRLF_to_LF",
        "questions": 20, "rows": 120,
        "rubric": "准确性/引文支撑度各0-2；2=关键结论完整或引用直接支持；1=部分正确或部分支持；0=主结论错误或无有效支持",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已保存120条逐题评分与6组质量汇总")


if __name__ == "__main__":
    main()
