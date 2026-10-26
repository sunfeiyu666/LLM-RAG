"""把正式评测证据从旧分块确定性映射到当前索引的文本块。

不修改 data/eval 下的正式文件；输出一份实验专用 QA 和逐条映射报告。
"""

import argparse
import json
import re
from pathlib import Path

from kb import INDEX, ROOT


def normalize(text):
    return re.sub(r"\s+", "", text)


def choose_chunk(evidence, records):
    quote = normalize(evidence["quote"])
    candidates = [
        row for row in records
        if row["metadata"].get("doc_id") == evidence["doc_id"]
        and row["metadata"].get("page") == evidence.get("page")
        and quote in normalize(row["page_content"])
    ]
    if not candidates:
        raise ValueError(
            f"证据无法映射：{evidence['doc_id']} 第{evidence.get('page')}页：{evidence['quote'][:80]}"
        )

    # 重叠窗口可能让同一证据出现在两块中。选择证据离块边界更远的一块，
    # 再以 start_index 和 chunk_id 保证结果可复现。
    def key(row):
        content = normalize(row["page_content"])
        start = content.index(quote)
        margin = min(start, len(content) - start - len(quote))
        meta = row["metadata"]
        return (-margin, meta.get("start_index", 0), meta["chunk_id"])

    candidates.sort(key=key)
    return candidates[0], [row["metadata"]["chunk_id"] for row in candidates]


def main():
    parser = argparse.ArgumentParser(description="把正式证据映射到当前分块")
    parser.add_argument("--qa", type=Path, default=ROOT / "data/eval/qa_set.jsonl")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/eval/qa_evidence.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads((INDEX / "documents.json").read_text(encoding="utf-8"))
    questions = {
        row["id"]: row
        for line in args.qa.read_text(encoding="utf-8").splitlines() if line.strip()
        for row in [json.loads(line)]
    }
    evidence_rows = [
        json.loads(line) for line in args.evidence.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapped_questions = []
    report = []
    for evidence_row in evidence_rows:
        question = dict(questions[evidence_row["id"]])
        selected_ids = []
        mappings = []
        for evidence in evidence_row["evidence"]:
            selected, acceptable = choose_chunk(evidence, records)
            chunk_id = selected["metadata"]["chunk_id"]
            if chunk_id not in selected_ids:
                selected_ids.append(chunk_id)
            mappings.append({
                "old_chunk_id": evidence["chunk_id"],
                "selected_chunk_id": chunk_id,
                "acceptable_exact_match_chunk_ids": acceptable,
                "doc_id": evidence["doc_id"],
                "page": evidence.get("page"),
                "quote": evidence["quote"],
            })
        question["gold_chunk_ids"] = selected_ids
        mapped_questions.append(question)
        report.append({"id": question["id"], "mappings": mappings})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapped_questions),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps({
        "index": str(INDEX),
        "questions": len(mapped_questions),
        "mapping_rule": "同doc_id与page内精确匹配规范化证据；重叠时选择证据离块边界更远者",
        "rows": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已映射 {len(mapped_questions)} 题，输出 {args.out}")


if __name__ == "__main__":
    main()
