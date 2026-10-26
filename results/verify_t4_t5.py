"""核对 T4/T5 的正式交付文件与原始数据是否一致。"""

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T4 = ROOT / "results/t4_bge_20261025"
T5 = ROOT / "results/t5_final_20261026"
QA = ROOT / "data/eval/qa_set.jsonl"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def png(path):
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), path


def main():
    qa = jsonl(QA)
    assert len(qa) == 20 and len({row["id"] for row in qa}) == 20
    # Git 在不同平台可能转换换行符；按统一换行校验同一份题目内容。
    qa_hash = hashlib.sha256(QA.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    metrics = csv_rows(T4 / "t4_metrics.csv")
    t4meta = json.loads((T4 / "run_metadata.json").read_text(encoding="utf-8"))
    assert t4meta["qa_sha256"] == qa_hash
    assert {row["method"] for row in metrics} == {"bm25", "dense", "hybrid"}
    for row in metrics:
        assert row["questions"] == "20"
        for name in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"):
            assert 0 <= float(row[name]) <= 1
    rankings = jsonl(T4 / "t4_rankings.jsonl")
    assert len(rankings) == 60
    assert {(r["id"], r["method"]) for r in rankings} == {
        (q["id"], method) for q in qa for method in ("bm25", "dense", "hybrid")
    }
    assert all(len(r["retrieved_chunk_ids"]) == 10 for r in rankings)
    for metric in metrics:
        relevant = [r for r in rankings if r["method"] == metric["method"]]
        ranks = [next((i for i, chunk_id in enumerate(r["retrieved_chunk_ids"], 1)
                       if chunk_id in r["gold_chunk_ids"]), None) for r in relevant]
        for k in (1, 3, 5, 10):
            measured = statistics.mean(rank is not None and rank <= k for rank in ranks)
            assert abs(measured - float(metric[f"recall@{k}"])) < 1e-9
        measured_mrr = statistics.mean(1 / rank if rank else 0 for rank in ranks)
        assert abs(measured_mrr - float(metric["mrr@10"])) < 1e-9
    png(T4 / "t4_bge_metrics.png")
    probes = json.loads((T4 / "t4_bge_candidate_probe.json").read_text(encoding="utf-8"))
    assert len(probes) == 45
    chosen = {"p008", "p016", "p019", "p038", "p044"}
    for row in probes:
        if row["id"] in chosen:
            assert row["bge_rank"] <= 3 and (row["bm25_rank"] or 11) > 3
    text = (T4 / "t4_bge_cases.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^## \d+\. .*BGE 占优", text, re.M)) == 6
    assert len(re.findall(r"^## \d+\. .*BM25 占优", text, re.M)) == 5

    rows = jsonl(T5 / "t5_full_rows.jsonl")
    assert len(rows) == 120 and {row["retrieval"] for row in rows} == {"bm25"}
    expected = {(q["id"], strategy, compressed) for q in qa
                for strategy in ("direct", "rule", "model") for compressed in (False, True)}
    actual = {(row["id"], row["strategy"], row["compressed"]) for row in rows}
    assert actual == expected and len(actual) == len(rows)
    by_key = {(row["id"], row["strategy"], row["compressed"]): row for row in rows}
    for q in qa:
        for strategy in ("direct", "rule", "model"):
            before = by_key[q["id"], strategy, False]
            after = by_key[q["id"], strategy, True]
            assert before["selected_chunk_ids"] == after["selected_chunk_ids"]
            assert isinstance(before["context_tokens"], int) and before["context_tokens"] > 0
            assert isinstance(after["context_tokens"], int) and 0 <= after["context_tokens"] <= before["context_tokens"]
            for row in (before, after):
                assert row["generation_input_tokens"] is not None and row["total_seconds"] >= 0
    summary = csv_rows(T5 / "t5_full_summary.csv")
    quality = csv_rows(T5 / "t5_quality_summary.csv")
    assert len(summary) == len(quality) == 6
    assert all(row["questions"] == "20" for row in summary + quality)
    assert all(row["mean_context_tokens"] and row["mean_rerank_seconds"] for row in summary)
    reviews = csv_rows(T5 / "t5_answer_review.csv")
    assert len(reviews) == 120
    assert {(r["id"], r["strategy"], r["compressed"] == "True") for r in reviews} == expected
    assert all(r["accuracy_0_2"] in ("0", "1", "2") and
               r["citation_support_0_2"] in ("0", "1", "2") for r in reviews)
    score_meta = json.loads((T5 / "quality_review_metadata.json").read_text(encoding="utf-8"))
    assert score_meta["raw_rows_sha256"] == hashlib.sha256(
        (T5 / "t5_full_rows.jsonl").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    for row in summary:
        selected = [r for r in rows if r["strategy"] == row["strategy"] and
                    str(r["compressed"]) == row["compressed"]]
        assert abs(statistics.mean(r["context_tokens"] for r in selected) -
                   float(row["mean_context_tokens"])) < 1e-9
        assert abs(statistics.mean(r["rerank_seconds"] for r in selected) -
                   float(row["mean_rerank_seconds"])) < 1e-9
        assert abs(statistics.mean(r["total_seconds"] for r in selected) -
                   float(row["mean_total_seconds"])) < 1e-9
    for row in quality:
        selected = [r for r in reviews if r["strategy"] == row["strategy"] and
                    r["compressed"] == row["compressed"]]
        assert abs(statistics.mean(int(r["accuracy_0_2"]) for r in selected) -
                   float(row["mean_accuracy_0_2"])) < 1e-9
        assert abs(statistics.mean(int(r["citation_support_0_2"]) for r in selected) -
                   float(row["mean_citation_support_0_2"])) < 1e-9
    png(T5 / "t5_latency_quality.png")
    t5meta = json.loads((T5 / "run_metadata.json").read_text(encoding="utf-8"))
    assert t5meta["qa_sha256"] == qa_hash and t5meta["retrieval"] == "bm25"
    assert t5meta["model_reranker"] == "cross_encoder" and t5meta["compression"] == "both"
    assert t5meta["compression_method"] == "question_term_self_information_sentence_selection"
    print("T4/T5 交付文件、20题×6组、指标、评分、Context token、延迟和PNG图均已核对。")


if __name__ == "__main__":
    main()
