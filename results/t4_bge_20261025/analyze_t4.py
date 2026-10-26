"""从本地 T4 逐题排名计算补充诊断，不调用模型。"""

import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (OUT / "t4_rankings.jsonl").read_text(encoding="utf-8").splitlines() if line]
by_question = defaultdict(dict)
for row in rows:
    by_question[row["id"]][row["method"]] = row

def first_rank(row):
    return next((rank for rank, chunk_id in enumerate(row["retrieved_chunk_ids"], 1)
                 if chunk_id in row["gold_chunk_ids"]), None)

summary = {"questions": len(by_question), "methods": {}, "first_gold_rank_comparison": {}}
for method in ("bm25", "dense", "hybrid"):
    selected = [row for row in rows if row["method"] == method]
    full = sum(set(row["gold_chunk_ids"]).issubset(row["retrieved_chunk_ids"][:5]) for row in selected)
    summary["methods"][method] = {"all_gold_at_5": full / len(selected), "all_gold_count": full}

comparison = {"dense_better": [], "bm25_better": [], "tie": []}
for question_id, methods in by_question.items():
    dense = first_rank(methods["dense"])
    bm25 = first_rank(methods["bm25"])
    if dense < bm25:
        comparison["dense_better"].append(question_id)
    elif bm25 < dense:
        comparison["bm25_better"].append(question_id)
    else:
        comparison["tie"].append(question_id)
summary["first_gold_rank_comparison"] = comparison
(OUT / "t4_diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
