"""Run a real streamed baseline question, without modifying T1 result files."""
import argparse
import json
import time

from preflight import check_ollama
from runtime import answer_stream, kb

parser = argparse.ArgumentParser()
parser.add_argument("--question", default="云南大学数据库概论的先修课是什么？")
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()
status = check_ollama()
knowledge_base = kb.load_index()
record = {"question": args.question, "k": args.top_k, **status}
started = time.perf_counter()
count = 0
for token in answer_stream(args.question, knowledge_base, args.top_k, record):
    count += 1
    if count == 1:
        record["first_token_seconds"] = round(time.perf_counter() - started, 3)
    print(token, end="", flush=True)
record.update(seconds=round(time.perf_counter() - started, 3), stream_events=count,
              chunks=len(knowledge_base.chunks))
print("\nSMOKE_RESULT=" + json.dumps(record, ensure_ascii=False))
if record["invalid_citations"]:
    raise SystemExit(2)
