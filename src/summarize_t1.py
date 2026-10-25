"""Summarize actual T1 calls plus explicit assistant review; no judge model."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]

def main():
    output = ROOT / "results"
    dataset = load(ROOT / "data/eval/qa_set.jsonl")
    runs = {"rag": load(output / "t1_baseline.jsonl"), "bare": load(output / "t1_no_retrieval.jsonl")}
    review_rows = load(output / "t1_answer_reviews.jsonl")
    reviews = {(r["id"], r["condition"]): r for r in review_rows}
    assert len(reviews) == len(review_rows) == len(dataset) * 2
    actual = {}
    for arm, rows in runs.items():
        assert len(rows) == len(dataset)
        assert len({r["id"] for r in rows}) == len(rows)
        assert {r["id"] for r in rows} == {q["id"] for q in dataset}
        for q, r in zip(dataset, rows):
            assert r["id"] == q["id"] and r["question"] == q["question"]
            assert r["gold_answer"] == q["gold_answer"]
            assert r["status"] == "ok" and isinstance(r["answer"], str) and r["answer"].strip()
            assert r["condition"] == arm
            assert (len(r["retrieved"]) == 5) if arm == "rag" else (not r["retrieved"] and not r["context"])
            review = reviews[(r["id"], arm)]
            assert review["level"] == q["level"]
            maximum = 5 if q["level"] == "L4" else 2
            assert review["max_score"] == maximum and 0 <= review["score"] <= maximum
            assert review["reason"] and review["answer_quotes"]
            for quote in review["answer_quotes"]:
                assert quote in r["answer"], (r["id"], arm, quote)
            if q["level"] == "L4":
                assert len(review["point_scores"]) == 5 and sum(review["point_scores"]) == review["score"]
            actual[(r["id"], arm)] = r
    summary = {"questions": len(dataset), "levels": dict(Counter(q["level"] for q in dataset)),
               "grading": "Assistant qualitative review, not independent human assessment. L1-L3 0/1/2; L4 five 0/1 points; aggregate means use normalized per-question score. Not EM accuracy, not LLM judge scores.",
               "conditions": {}}
    for arm, rows in runs.items():
        rs = [reviews[(r["id"], arm)] for r in rows]
        stats = {"completed": len(rows), "errors": sum(r["status"] != "ok" for r in rows),
                 "truncated": sum(r["truncated"] for r in rows),
                 "normalized_mean_score": statistics.mean(r["score"] / r["max_score"] for r in rs),
                 "full_score_count": sum(r["score"] == r["max_score"] for r in rs),
                 "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in rows),
                 "median_seconds": statistics.median(r["elapsed_seconds"] for r in rows),
                 "p95_seconds": sorted(r["elapsed_seconds"] for r in rows)[max(0, int(len(rows) * .95 + .9999) - 1)],
                 "total_seconds": sum(r["elapsed_seconds"] for r in rows),
                 "by_level": {}}
        for level in ["L1", "L2", "L3", "L4"]:
            selected = [r for r in rs if r["level"] == level]
            stats["by_level"][level] = {"n": len(selected), "normalized_mean_score": statistics.mean(r["score"] / r["max_score"] for r in selected),
                                        "full_score_count": sum(r["score"] == r["max_score"] for r in selected)}
        if arm == "rag":
            stats.update(hit_recall_at_5=statistics.mean(r["hit_at_5"] for r in rows),
                         mean_gold_coverage_at_5=statistics.mean(r["gold_coverage"] for r in rows),
                         all_gold_retrieved_count=sum(r["gold_coverage"] == 1 for r in rows),
                         mrr_at_5=statistics.mean(r["reciprocal_rank_at_5"] for r in rows),
                         invalid_citation_id_answers=sum(not r["citation_ids_valid"] for r in rows))
        summary["conditions"][arm] = stats
    comparisons = []
    for q in dataset:
        r, b = actual[(q["id"], "rag")], actual[(q["id"], "bare")]
        rr, br = reviews[(q["id"], "rag")], reviews[(q["id"], "bare")]
        delta = rr["score"] / rr["max_score"] - br["score"] / br["max_score"]
        comparisons.append({"id": q["id"], "level": q["level"], "question": q["question"],
                            "rag_score": rr["score"], "bare_score": br["score"], "max_score": rr["max_score"],
                            "normalized_delta": delta, "rag_gold_hit_at_5": r["hit_at_5"],
                            "rag_gold_coverage_at_5": r["gold_coverage"], "rag_failure_stage": rr["failure_stage"],
                            "rag_seconds": r["elapsed_seconds"], "bare_seconds": b["elapsed_seconds"],
                            "rag_reason": rr["reason"], "bare_reason": br["reason"]})
    summary["paired_comparison"] = {"rag_better": sum(r["normalized_delta"] > 0 for r in comparisons),
                                     "equal": sum(r["normalized_delta"] == 0 for r in comparisons),
                                     "rag_worse": sum(r["normalized_delta"] < 0 for r in comparisons)}
    summary["failure_stages"] = dict(Counter(reviews[(q["id"], "rag")]["failure_stage"] for q in dataset))
    (output / "t1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "t1_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    text = ["# T1 基础RAG与无检索对照：逐题原始答案", "本文件只排版真实运行输出，不改写模型答案。运行设置见t1_run_metadata.json，评分见t1_answer_reviews.jsonl。"]
    for q, comparison in zip(dataset, comparisons):
        r, b = actual[(q["id"], "rag")], actual[(q["id"], "bare")]
        text += [f"## {q['id']} · {q['level']}\n\n{q['question']}", "### 参考答案\n\n" + q["gold_answer"],
                 "### RAG原始答案\n\n" + r["answer"], "### 无检索原始答案\n\n" + b["answer"],
                 "### 核对\n\n" + f"RAG {comparison['rag_score']}/{comparison['max_score']}；无检索 {comparison['bare_score']}/{comparison['max_score']}。\n\nRAG：{comparison['rag_reason']}\n\n无检索：{comparison['bare_reason']}",
                 "RAG实际召回：" + "、".join(f"`{x['chunk_id']}`" for x in r["retrieved"]),
                 "Gold：" + "、".join(f"`{x}`" for x in q["gold_chunk_ids"])]
    (output / "t1_逐题答案对照.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
