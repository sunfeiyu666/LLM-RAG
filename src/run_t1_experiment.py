"""T1: paired baseline RAG and no-retrieval calls, append-only checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time
from urllib.request import urlopen

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from kb import INDEX, load_index
from rag_chain import MODEL, BASE_URL, build_rag_chain

ROOT = Path(__file__).resolve().parents[1]
BARE_SYSTEM = (
    "你是云南大学软件学院相关问题的中文问答助手。请直接根据自身已有知识回答用户问题。"
    "严格核对专业、方案版本、入学年级、毕业届别和选课学期。区分推荐推免资格和接收推免生。"
    "不能用旧年政策推断新年政策。问题缺少年级且存在版本差异时请用户补充。"
    "不确定的事实请明确说明不知道或无法确认，不要编造。"
    "本次没有接入知识库、检索工具或外部资料，不要声称检索了资料，不要伪造来源编号。"
    "直接输出中文正文，不要输出JSON。"
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []

class Capture(BaseCallbackHandler):
    def __init__(self):
        self.responses = []
    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for item in generation:
                message = getattr(item, "message", None)
                self.responses.append({"response_metadata": getattr(message, "response_metadata", {}),
                                       "usage_metadata": getattr(message, "usage_metadata", {}),
                                       "llm_output": response.llm_output})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "data/eval/qa_set.jsonl"
    questions = read_rows(dataset)
    assert len(questions) >= 20 and len({q['id'] for q in questions}) == len(questions)
    params = dict(model=MODEL, base_url=BASE_URL, temperature=0.2, seed=42, num_ctx=16384, num_predict=2048)
    meta_path = output / "t1_run_metadata.json"
    paths = {"rag": output / "t1_baseline.jsonl", "bare": output / "t1_no_retrieval.jsonl"}
    if not args.resume and any(p.exists() for p in [meta_path, *paths.values()]):
        raise FileExistsError("Existing T1 outputs preserved. Use --resume or a new --output-dir.")
    with urlopen(BASE_URL + "/api/tags", timeout=10) as response:
        model_info = next(m for m in json.load(response)["models"] if m["name"] == MODEL)
    with urlopen(BASE_URL + "/api/version", timeout=10) as response:
        server = json.load(response)
    kb = load_index()
    run_binding = {"qa_sha256": sha(dataset), "index_info_sha256": sha(INDEX / "index_info.json"),
                   "model_digest": model_info["digest"], "model_parameters": params, "top_k": args.top_k,
                   "rag_chain_sha256": sha(ROOT / "src/rag_chain.py"), "runner_sha256": sha(Path(__file__))}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["binding"] == run_binding, "Resume binding differs; use a new output directory"
    else:
        meta = {"started_at": datetime.now(timezone.utc).isoformat(), "binding": run_binding,
                "question_count": len(questions), "ollama": server, "model_info": model_info,
                "python": sys.version, "platform": platform.platform(),
                "packages": {n: importlib.metadata.version(n) for n in ["langchain-core", "langchain-ollama", "faiss-cpu", "scikit-learn"]},
                "bare_system_prompt": BARE_SYSTEM,
                "control": "Same questions/model/digest/temperature/seed/context limit/output limit, independent chats, no tools. RAG uses original build_rag_chain prompt; bare prompt removes evidence-only and citation demands. Thus prompt adaptation is a disclosed confound.",
                "order": "Question order fixed; odd IDs RAG then bare, even IDs bare then RAG; one call at a time; cold-start cost retained.",
                "scoring": "Gold not supplied to either model; responses scored later against frozen D3 evidence; no model judge used.",
                "limits": "Single run per question/condition; 2048 generated tokens may truncate long answers; seed does not guarantee cross-platform bitwise reproducibility."}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = {arm: {r["id"] for r in read_rows(path)} for arm, path in paths.items()}
    llm = ChatOllama(**params, client_kwargs={"timeout": 900.0})
    chain = build_rag_chain(kb.as_retriever(k=args.top_k), llm=llm)
    for ordinal, q in enumerate(questions, 1):
        for arm in (["rag", "bare"] if ordinal % 2 else ["bare", "rag"]):
            if q["id"] in done[arm]:
                continue
            captured = Capture()
            started = datetime.now(timezone.utc).isoformat()
            begin = time.perf_counter()
            print(f"START {q['id']} {q['level']} {arm}", flush=True)
            row = {**q, "condition": arm, "model": MODEL, "started_at": started,
                   "answer": "", "status": "ok", "retrieved": [], "context": ""}
            try:
                if arm == "rag":
                    result = chain.invoke({"question": q["question"]}, config={"callbacks": [captured]})
                    row["answer"] = result["answer"]
                    row["context"] = result["context"]
                    row["retrieved"] = [{"rank": i, "chunk_id": d.metadata["chunk_id"],
                                          "doc_id": d.metadata["doc_id"], "title": d.metadata["title"],
                                          "page_content": d.page_content, "metadata": d.metadata}
                                         for i, d in enumerate(result["documents"], 1)]
                else:
                    message = llm.invoke([SystemMessage(content=BARE_SYSTEM), HumanMessage(content=q["question"])],
                                         config={"callbacks": [captured]})
                    row["answer"] = message.content
                row["citations"] = sorted({int(n) for n in re.findall(r"\[(\d+)\]", row["answer"])})
                row["citation_ids_valid"] = all(1 <= n <= len(row["retrieved"]) for n in row["citations"])
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            row["elapsed_seconds"] = round(time.perf_counter() - begin, 4)
            row["llm_calls"] = captured.responses
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            row["truncated"] = any(r["response_metadata"].get("done_reason") == "length" for r in captured.responses)
            hits = [r["chunk_id"] for r in row["retrieved"]]
            row["gold_hits"] = [cid for cid in q["gold_chunk_ids"] if cid in hits]
            row["hit_at_5"] = bool(row["gold_hits"]) if arm == "rag" else None
            row["gold_coverage"] = len(row["gold_hits"]) / len(q["gold_chunk_ids"]) if arm == "rag" else None
            row["reciprocal_rank_at_5"] = next((1 / i for i, cid in enumerate(hits, 1) if cid in q["gold_chunk_ids"]), 0.0) if arm == "rag" else None
            with paths[arm].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
            print(f"DONE {q['id']} {arm} status={row['status']} seconds={row['elapsed_seconds']} chars={len(row['answer'])} gold={len(row['gold_hits'])}/{len(q['gold_chunk_ids'])} truncated={row['truncated']}", flush=True)
    final_rows = {arm: read_rows(path) for arm, path in paths.items()}
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    meta["completion"] = {arm: {"rows": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
                                 "errors": sum(r["status"] != "ok" for r in rows),
                                 "truncated": sum(r["truncated"] for r in rows)} for arm, rows in final_rows.items()}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta["completion"], ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
