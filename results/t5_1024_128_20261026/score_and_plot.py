"""保存1024/128分块实验的逐题评分、质量汇总与延迟—质量图。"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).resolve().parent
ORDER = (("direct", False), ("direct", True), ("rule", False),
         ("rule", True), ("model", False), ("model", True))

# 每题六组，顺序见 ORDER；每组为（答案准确性，引文支撑度），两项均为0—2分。
# 2=关键结论正确且完整/引用直接支持；1=部分正确/部分支持；0=主结论错误或无有效支持。
SCORES = {
    "q001": ((2, 2),) * 6,
    "q002": ((2, 2),) * 6,
    "q003": ((2, 2),) * 6,
    "q004": ((2, 2),) * 6,
    "q005": ((2, 2),) * 6,
    "q006": ((2, 2),) * 6,
    "q007": ((0, 1), (1, 1), (1, 1), (0, 1), (2, 2), (2, 2)),
    "q008": ((2, 2),) * 6,
    "q009": ((1, 1), (1, 1), (0, 1), (1, 1), (1, 1), (1, 1)),
    "q010": ((2, 2),) * 6,
    "q011": ((2, 2), (1, 0), (2, 2), (1, 0), (2, 2), (0, 0)),
    "q012": ((2, 2),) * 6,
    "q013": ((2, 2), (0, 0), (2, 2), (0, 0), (2, 2), (0, 0)),
    "q014": ((2, 2),) * 6,
    "q015": ((2, 2),) * 6,
    "q016": ((1, 1),) * 6,
    "q017": ((2, 2),) * 6,
    "q018": ((2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1)),
    "q019": ((2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)),
    "q020": ((1, 1),) * 6,
}

REASONS = {
    "q001": "给出第一课堂161.5学分，引用对应培养方案。",
    "q002": "给出先修课数据结构，引用公开慕课页面。",
    "q003": "给出缓考一般不超过两门，引用2020版原文。",
    "q004": "给出前三学年GPA不低于3.0，引用2027年学校办法。",
    "q005": "不授学位与微专业修读证书两项结论均正确。",
    "q006": "均正确区分转专业10学分、辅修或微专业15学分及5学分差值。",
    "q007": "只有模型重排同时召回两份通知并给出两个不同窗口；其余组缺一项或合并错时间。",
    "q008": "均答出辅修34、微专业34.25、相差0.25且包含主修课程。",
    "q009": "均能区分学位与非学位证书，但缺少55学分、主修学位及审议条件；部分表述超出证据。",
    "q010": "均答出161.5、6、167.5，并说明单纯相加不能保证毕业。",
    "q011": "未压缩组按GPA 3.0正确判断；压缩后混入综合测评80分，模型组甚至称无法判断。",
    "q012": "均正确指出专业必修课曾补考不满足条件，GPA和英语不能抵消。",
    "q013": "未压缩组按前一学期理工科2.8判断；压缩后误按转入后的文科3.0判断。",
    "q014": "均答出14人不足15人、不开课、结果无效及下学期重申。",
    "q015": "均识别总量14未超15，但专业教育6超过5，不能原样全部认定。",
    "q016": "各组给出部分可用规划，但均遗漏收费或辅修/微专业完整比较，并含‘一定更轻、一定不冲突’等过度推断。",
    "q017": "均能依据方向介绍选择智慧医疗，并与智能传播作有效区分。",
    "q018": "直接与规则未压缩组覆盖主要基础、可靠性和实现路线；其余组缺事务/并发/恢复或应用开发等关键阶段。",
    "q019": "未压缩组覆盖10、15、总计20及专业教育5学分限制；压缩组漏掉各类认定总计20的约束。",
    "q020": "均指出不能靠竞赛科研保证推免，但未完整覆盖英语、补考重修、按综合测评排序和学院细则等约束，且有少量过度推断。",
}

COLORS = {"direct": "#2563eb", "rule": "#2a9d8f", "model": "#e76f51"}
LABELS = {"direct": "直接 top-3", "rule": "规则重排", "model": "Cross-Encoder 重排"}
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")


def read_csv(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as file:
        return {(row["strategy"], row["compressed"]): row for row in csv.DictReader(file)}


def main():
    raw = (HERE / "t5_full_rows.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    expected = {(qid, strategy, compressed) for qid in SCORES for strategy, compressed in ORDER}
    actual = {(row["id"], row["strategy"], row["compressed"]) for row in rows}
    if len(rows) != 120 or expected != actual or len(actual) != 120:
        raise ValueError("必须是20题×6组且组合不能重复")
    if {row["retrieval"] for row in rows} != {"bm25"}:
        raise ValueError("本次正式T5应建立在T4最优BM25之上")

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
                             "answer": row["answer"],
                             "selected_chunk_ids": row["selected_chunk_ids"],
                             "accuracy_0_2": accuracy, "citation_support_0_2": support,
                             "notes": REASONS[row["id"]]})

    groups = defaultdict(list)
    for row in rows:
        key = row["id"], row["strategy"], row["compressed"]
        groups[(row["strategy"], row["compressed"])].append(lookup[key])
    quality = []
    for strategy, compressed in ORDER:
        scores = groups[(strategy, compressed)]
        quality.append({"strategy": strategy, "compressed": compressed, "questions": len(scores),
                        "mean_accuracy_0_2": sum(x[0] for x in scores) / len(scores),
                        "full_accuracy_count": sum(x[0] == 2 for x in scores),
                        "mean_citation_support_0_2": sum(x[1] for x in scores) / len(scores),
                        "full_support_count": sum(x[1] == 2 for x in scores)})
    with (HERE / "t5_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(quality[0]))
        writer.writeheader()
        writer.writerows(quality)

    (HERE / "quality_review_metadata.json").write_text(json.dumps({
        "raw_rows_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        "hash_normalization": "CRLF_to_LF", "questions": 20, "rows": 120,
        "rubric": "准确性/引文支撑度各0-2；2=关键结论正确完整或引用直接支持；1=部分正确或部分支持；0=主结论错误或无有效支持",
        "review_note": "逐题对照标准答案、实际回答与所选文本块核对。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latency = read_csv("t5_full_summary.csv")
    quality_rows = read_csv("t5_quality_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 5.7))
    all_x, all_y = [], []
    for strategy in COLORS:
        points = []
        for compressed in ("True", "False"):
            key = strategy, compressed
            point = (float(latency[key]["mean_total_seconds"]),
                     float(quality_rows[key]["mean_accuracy_0_2"]))
            points.append(point); all_x.append(point[0]); all_y.append(point[1])
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=COLORS[strategy], linewidth=2.2, label=LABELS[strategy])
        ax.scatter([xs[0]], [ys[0]], color=COLORS[strategy], marker="o", s=90, zorder=3)
        ax.scatter([xs[1]], [ys[1]], color=COLORS[strategy], marker="s", s=72, zorder=3)
        compressed_offset = {"direct": (-4, 15), "rule": (8, 8), "model": (8, -18)}[strategy]
        ax.annotate("压缩后", points[0], textcoords="offset points", xytext=compressed_offset,
                    fontproperties=FONT, fontsize=9)
        ax.annotate("压缩前", points[1], textcoords="offset points", xytext=(8, -18),
                    fontproperties=FONT, fontsize=9)
    x_pad = max(0.5, (max(all_x) - min(all_x)) * 0.15)
    y_pad = max(0.1, (max(all_y) - min(all_y)) * 0.2)
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(max(0, min(all_y) - y_pad), min(2, max(all_y) + y_pad))
    ax.set_xlabel("平均端到端延迟（秒/题）", fontproperties=FONT, fontsize=11)
    ax.set_ylabel("平均答案准确性评分（0—2）", fontproperties=FONT, fontsize=11)
    ax.set_title("T5：1024/128 分块的延迟—质量权衡（20题）", fontproperties=FONT, fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(prop=FONT, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "t5_latency_quality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已保存120条逐题评分、6组质量汇总和延迟—质量图")


if __name__ == "__main__":
    main()
