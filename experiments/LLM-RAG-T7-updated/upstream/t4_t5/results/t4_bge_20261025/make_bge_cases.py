"""按保存的真实排序生成 5+5 案例，避免转录名次或块编号时出错。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
qa = {row["id"]: row for row in (json.loads(line) for line in
      (ROOT / "data/eval/qa_set.jsonl").read_text(encoding="utf-8").splitlines())}
rankings = {(row["id"], row["method"]): row for row in (json.loads(line) for line in
            (HERE / "t4_rankings.jsonl").read_text(encoding="utf-8").splitlines())}
probes = {row["id"]: row for row in json.loads(
    (HERE / "t4_bge_candidate_probe.json").read_text(encoding="utf-8"))}

cases = [
    ("q016", "BGE", "问题询问个人学习目标与安排，BGE 较早检到微专业学期上限块；两路 top-3 都没有覆盖全部四个标注证据，不能据此断言能完整作答。"),
    ("p008", "BGE", "‘不想申请第二学位、只想拿完成证明’是对辅修专业证书要求的口语化描述。BGE 将证书条款排在第2，BM25 排在第5。"),
    ("p016", "BGE", "问题用‘以前没过、后来补考通过’描述专业必修课经历。BGE 将推免补考限制排在第2，BM25 排在第4。"),
    ("p019", "BGE", "问题用‘推荐免试的门槛、平均绩点以外的学业表现’概括文件中的综合测评、英语和课程条件。BGE 首位命中学校规则，BM25 的前10位均未命中该标注块。"),
    ("p038", "BGE", "问题用‘读完部分课程后的完成证明和办理步骤’改写辅修专业证书条款。BGE 首位命中证书发放块，BM25 将它排在第9。"),
    ("p044", "BGE", "‘医学场景的软件应用、不做媒体作品’以用途描述智慧医疗与智能传播的取舍，没有直接使用项目名称。BGE 首位命中智慧医疗介绍，BM25 第4位才命中。"),
    ("q001", "BM25", "2022版、2023年修订和第一课堂学分构成较精确的版本与数值线索。BM25 首位命中目标块，BGE 第3位才命中。"),
    ("q003", "BM25", "2020版和‘缓考不得超过几门’定位到具体规定。BM25 首位命中，BGE 前3位是相邻考务条款，目标块第5位。"),
    ("q005", "BM25", "两路前三位均有相近微专业材料；BM25 第3位命中证书条款，BGE 第4位。优势仅一位且前3证据覆盖不同。"),
    ("q011", "BM25", "2027年政策、GPA 2.9 与 3.0 形成精确版本和数值线索。BM25 首位命中学校门槛块，BGE 第5位才命中。"),
    ("q010", "BM25", "培养方案版本与第一、第二课堂的具体学分数适合精确匹配。BM25 首位命中，BGE 第9位，前3位缺目标块。"),
]


def first_gold(ids, gold):
    return next((index for index, chunk_id in enumerate(ids, 1) if chunk_id in gold), None)


def format_top3(ids, gold):
    return "<br>".join(("★ " if chunk_id in gold else "") + f"`{chunk_id}`"
                        for chunk_id in ids[:3])


lines = ["# T4 BGE 与 BM25 排序案例（BGE 6 条、BM25 5 条）", "",
         "比较标准：标注证据块**首次出现的名次**，名次越小越好；★表示已标注证据。",
         "q 编号来自正式20题；p 编号来自另拟的45条诊断问题，全部候选及排序见 `t4_bge_candidate_probe.json`。",
         "诊断题的标注块根据知识库原文确定，但案例有选择偏差，不计入正式 Recall/MRR，也不证明总体优势。", ""]
for number, (case_id, winner, analysis) in enumerate(cases, 1):
    if case_id.startswith("q"):
        source = qa[case_id]
        question = source["question"]
        gold = set(source["gold_chunk_ids"])
        sparse = rankings[(case_id, "bm25")]["retrieved_chunk_ids"]
        dense = rankings[(case_id, "dense")]["retrieved_chunk_ids"]
    else:
        source = probes[case_id]
        question = source["question"]
        gold = {source["gold_chunk_id"]}
        sparse, dense = source["bm25_top10"], source["bge_top10"]
    sparse_rank, dense_rank = first_gold(sparse, gold), first_gold(dense, gold)
    assert sparse_rank != dense_rank and (
        (dense_rank or 11) < (sparse_rank or 11) if winner == "BGE" else
        (sparse_rank or 11) < (dense_rank or 11)
    ), case_id
    if case_id.startswith("p") and winner == "BGE":
        assert dense_rank is not None and dense_rank <= 3 and (sparse_rank or 11) > 3, case_id
    lines += [f"## {number}. {case_id}：{winner} 占优", "", f"问题：{question}", "",
              "标注证据：" + "、".join(f"`{chunk_id}`" for chunk_id in sorted(gold)), "",
              "| 方法 | 首次证据名次 | top-3 文本块 |", "|---|---:|---|",
              f"| BM25 | {sparse_rank or '>10'} | {format_top3(sparse, gold)} |",
              f"| BGE + FAISS | {dense_rank or '>10'} | {format_top3(dense, gold)} |",
              "", f"分析：{analysis}", ""]
lines += ["## 综合判断", "",
          "正式20题的 BGE 首次证据占优仅 q016，BM25 占优8题、持平11题。",
          "6条 BGE 案例中的5条来自有选择偏差的额外诊断题，不能用于总体胜率统计。",
          "本次混合 RRF 的 Recall@1=0.60、MRR@10=0.754，均低于 BM25 的0.80、0.875；",
          "但 top-5 完整覆盖所有标注块为13/20，略高于 BM25 的12/20。",
          "所以只能说混合在这个辅助覆盖指标上有小幅提升，不能声称主指标全面改善。", ""]
(HERE / "t4_bge_cases.md").write_text("\n".join(lines), encoding="utf-8")
print("已生成11条案例，全部名次和 top-3 已与原始排序核对。")
