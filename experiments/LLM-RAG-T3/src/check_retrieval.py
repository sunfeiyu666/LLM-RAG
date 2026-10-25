"""Small evidence-location smoke set; not a validated course evaluation set."""
import json
import argparse
from pathlib import Path

from kb import ROOT, load_index, to_chunk

CASES = [
    ("L1", "软件工程2022版2023年修订培养方案基本学制几年？", ["ynu001"], "基本学制为4"),
    ("L1", "软件工程2022版培养方案总学分是多少？", ["ynu001"], "161.5"),
    ("L1", "云南大学数据库概论先修课是什么？", ["ynu006"], "数据结构"),
    ("L1", "云南大学操作系统原理课程是否涉及死锁？", ["ynu007"], "死锁"),
    ("L1", "大数据分析及应用课程有哪些集群搭建实验？", ["ynu008"], "集群搭建"),
    ("L2", "2026年学籍管理规定，单个学期平均学分绩点低于多少会学业警示？", ["ynu009"], "1.0"),
    ("L2", "2026年课程学分认定规定，转专业选修课程累计认定上限多少？", ["ynu010"], "不超过10学分"),
    ("L2", "2026年课程学分认定规定，辅修和微专业认定为主修选修课程上限多少？", ["ynu010"], "不超过15学分"),
    ("L1", "2016版选课管理规定每学期选课原则上不超过多少学分？", ["ynu011"], "30"),
    ("L1", "2020版成绩与考务规定，每学期一般最多申请几门缓考？", ["ynu012"], "两门"),
    ("L2", "2027年云南大学推荐推免基本条件GPA下限是多少？", ["ynu018"], "3.0"),
    ("L2", "2025年软件学院推荐推免细则GPA下限是多少？", ["ynu021"], "2．8"),
    ("L1", "2025年软件学院推免英语四级成绩下限多少？", ["ynu021"], "425"),
    ("L2", "2027年软件学院接收推免复试外语成绩占多少？", ["ynu022"], "20%"),
    ("L1", "软件学院2026年夏令营拟招收多少营员？", ["ynu024"], "60人"),
    ("L1", "2025年软件工程拔尖班实践教学学分占比多少？", ["ynu025"], "34.75%"),
    ("L2", "2026年秋季辅修每学期主修和辅修合计最多多少学分？", ["ynu029"], "34学分"),
    ("L2", "2026年秋季微专业能同时修读几个？", ["ynu030"], "只能修读一个"),
    ("L2", "2026年秋季微专业修满后授予学位吗？", ["ynu030"], "不授予学位"),
    ("L3", "2027年学校推免资格GPA和2025年软件学院细则GPA是否一样？", ["ynu018", "ynu021"], None),
]


def main():
    parser = argparse.ArgumentParser(description="LangChain检索开发检查")
    parser.parse_args()
    retriever = load_index()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    rows = []
    for i, (level, question, gold_docs, needle) in enumerate(CASES, 1):
        hits = [
            (to_chunk(doc), None) for doc in retriever.as_retriever(k=5).invoke(question)
        ]
        ids = [c.doc_id for c, _ in hits]
        evidence_hit = None if needle is None else any(
            needle in c.context and c.doc_id in gold_docs for c, _ in hits
        )
        row = dict(id=f"smoke{i:03d}", level=level, question=question,
                   gold_doc_ids=gold_docs, evidence_keyword=needle,
                   retrieved=[c.chunk_id for c, _ in hits],
                   all_gold_docs_hit=all(d in ids for d in gold_docs),
                   evidence_keyword_hit=evidence_hit)
        rows.append(row)
        print(row["id"], row["all_gold_docs_hit"], evidence_hit, question)
    prefix = "langchain_baseline"
    (output / f"{prefix}_retrieval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = dict(questions=len(rows),
                   all_required_documents_at_5=sum(r["all_gold_docs_hit"] for r in rows) / len(rows),
                   evidence_keyword_at_5=sum(r["evidence_keyword_hit"] is True for r in rows) / sum(r["evidence_keyword_hit"] is not None for r in rows),
                   note="Developer smoke set, keyword-based evidence check; not independently annotated accuracy or LLM answer evaluation.")
    summary["mode"] = prefix
    (output / f"{prefix}_retrieval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
