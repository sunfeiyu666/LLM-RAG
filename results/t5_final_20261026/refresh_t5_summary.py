"""由不可更改的原始120条结果重新汇总 token 与分段延迟。"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from t5_evaluate import summarize

rows = [json.loads(line) for line in (HERE / "t5_full_rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 120:
    raise ValueError("需要完整120条结果")
summary = summarize(rows)
with (HERE / "t5_full_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
print("已从120条原始结果重算六组 token 与分段延迟")
