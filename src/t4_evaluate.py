"""使用人工核对的证据块编号评估 T4 检索排序。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from kb import ROOT
from retrieval_compare import T4Retrievers

KS = (1, 3, 5, 10)
METHODS = ("bm25", "dense", "hybrid")


def read_questions(path):
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("id") or not row.get("question") or not row.get("gold_chunk_ids"):
            raise ValueError(f"第 {number} 行缺少 id、question 或 gold_chunk_ids")
        questions.append(row)
    if not questions:
        raise ValueError("评测集为空")
    return questions


def score_rankings(rows):
    """按课程定义：top-k 中出现任一标准证据块，即视为该题召回成功。"""
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result = {"method": method, "questions": len(selected)}
        for k in KS:
            result[f"recall@{k}"] = sum(
                bool(set(row["gold_chunk_ids"]) & set(row["retrieved_chunk_ids"][:k]))
                for row in selected
            ) / len(selected)
        result["mrr@10"] = sum(
            next((1 / rank for rank, chunk_id in enumerate(row["retrieved_chunk_ids"][:10], 1)
                  if chunk_id in row["gold_chunk_ids"]), 0)
            for row in selected
        ) / len(selected)
        summary.append(result)
    return summary


def evaluate(questions, retrievers):
    known_ids = {doc.metadata["chunk_id"] for doc in retrievers.kb.documents}
    rows = []
    for question in questions:
        unknown = set(question["gold_chunk_ids"]) - known_ids
        if unknown:
            raise ValueError(f"{question['id']} 的 gold_chunk_ids 不在当前索引中：{sorted(unknown)}")
        for method in METHODS:
            hits = retrievers.search(question["question"], method, max(KS))
            rows.append({
                "id": question["id"], "level": question.get("level"),
                "question": question["question"], "method": method,
                "gold_chunk_ids": question["gold_chunk_ids"],
                "retrieved_chunk_ids": [doc.metadata["chunk_id"] for doc in hits],
            })
    return rows, score_rankings(rows)


def main():
    parser = argparse.ArgumentParser(description="T4 三种检索策略评测")
    parser.add_argument("--qa", type=Path, default=ROOT / "data/eval/qa_set.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--dense-backend", choices=("lsa", "bge"), default="lsa")
    args = parser.parse_args()
    questions = read_questions(args.qa)
    retrievers = T4Retrievers(dense_backend=args.dense_backend)
    rows, summary = evaluate(questions, retrievers)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "t4_rankings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with (args.out / "t4_metrics.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "dense_backend": args.dense_backend,
        "qa_sha256": hashlib.sha256(args.qa.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "questions": len(questions),
        "note": "BM25、稠密和 RRF 混合均使用同一批知识库文本块。",
    }
    if args.dense_backend == "bge":
        from neural_retrieval import INDEX_DIR

        metadata["bge_index"] = json.loads((INDEX_DIR / "index_info.json").read_text(encoding="utf-8"))
    (args.out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
