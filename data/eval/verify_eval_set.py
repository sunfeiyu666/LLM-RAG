"""Validate D3 JSONL, evidence, source hashes and annotation/index bindings; stdlib only."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def chunk_digest(rows):
    fields = ("chunk_id", "doc_id", "title", "page", "version", "content_url")
    canonical = [{"page_content": r["page_content"], "metadata": {k: r["metadata"].get(k) for k in fields}} for r in rows]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="校验评测集、证据及索引绑定，不调用LLM")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project_root.resolve()
    folder = root / "data/eval"
    read = lambda name: [json.loads(s) for s in (folder / name).read_text(encoding="utf-8").splitlines() if s.strip()]
    dataset = read("qa_set.jsonl")
    audit = read("qa_evidence.jsonl")
    binding = json.loads((folder / "qa_manifest.json").read_text(encoding="utf-8"))
    assert len(dataset) >= 20
    counts = Counter(q["level"] for q in dataset)
    assert set(counts) == {"L1", "L2", "L3", "L4"} and min(counts.values()) >= 5
    assert len({q["id"] for q in dataset}) == len(dataset)
    assert len(audit) == len(dataset)
    proofs = {r["id"]: r for r in audit}
    assert set(proofs) == {q["id"] for q in dataset}
    rows = json.loads((root / "index_langchain/documents.json").read_text(encoding="utf-8"))
    assert chunk_digest(rows) == binding["chunk_content_metadata_sha256"], "分块内容或元数据改变，应重新标注；不能只沿用旧编号"
    chunks = {r["metadata"]["chunk_id"]: r for r in rows}
    assert len(chunks) == len(rows)
    for relative, expected in binding["bound_file_sha256"].items():
        assert sha(root / relative) == expected, f"已变更：{relative}；请复核标注"
    for name, expected in binding["artifact_sha256"].items():
        assert sha(folder / name) == expected, f"评测文件已变更：{name}；请复核并更新manifest"
    quotes = 0
    original_files = set()
    for q in dataset:
        assert set(q) == {"id", "level", "question", "gold_answer", "gold_chunk_ids", "evidence_required"}
        assert re.fullmatch(r"q\d{3}", q["id"])
        assert q["question"].strip() and q["gold_answer"].strip()
        assert q["evidence_required"] is True
        ids = q["gold_chunk_ids"]
        assert ids and len(set(ids)) == len(ids) and all(cid in chunks for cid in ids)
        proof = proofs[q["id"]]
        assert proof["level"] == q["level"]
        assert {e["chunk_id"] for e in proof["evidence"]} == set(ids)
        for evidence in proof["evidence"]:
            row = chunks[evidence["chunk_id"]]
            assert row["metadata"]["doc_id"] == evidence["doc_id"]
            assert evidence["quote"] in row["page_content"]
            normalize = lambda s: re.sub(r"\s+", "", s)
            assert normalize(evidence["quote"]) in normalize((root / evidence["corpus_file"]).read_text(encoding="utf-8"))
            assert evidence["quote_in_original_extraction"] is True
            assert sha(root / evidence["original_file"]) == evidence["original_sha256"]
            original_files.add(evidence["original_file"])
            quotes += 1
        if q["level"] == "L4":
            assert len(proof["scoring_points"]) == 5
    print(json.dumps({"status": "passed", "questions": len(dataset), "levels": dict(counts),
                      "index_chunks": len(rows), "evidence_quotes": quotes,
                      "original_snapshots_verified": len(original_files),
                      "note": "结构、证据、哈希及分块绑定校验通过；不代表独立人工审核或模型答题结果"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
