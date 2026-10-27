"""Deployment adapter. Original source files, prompt and retrieval stay unchanged."""
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import kb
import rag_chain

rag_chain.BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
rag_chain.MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
kb.INDEX = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT / "Docker" / ".index"))).resolve()
kb.INDEX.mkdir(parents=True, exist_ok=True)


def source_records(documents):
    return [{"number": i, "chunk_id": d.metadata["chunk_id"],
             "title": d.metadata["title"], "url": d.metadata["content_url"],
             "version": d.metadata["version"], "page": d.metadata.get("page"),
             "text": d.page_content} for i, d in enumerate(documents, 1)]


def validate_citations(answer, count):
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    invalid = [n for n in numbers if not 1 <= n <= count]
    return [n for n in numbers if 1 <= n <= count], invalid


def answer_stream(question, knowledge_base, k, record):
    record.update(answer="", retrieved=[], cited_numbers=[], invalid_citations=[], complete=False)
    chain = rag_chain.build_rag_chain(knowledge_base.as_retriever(k=k))
    for event in chain.stream({"question": question}):
        if "documents" in event:
            record["retrieved"] = source_records(event["documents"])
        if event.get("answer"):
            token = event["answer"]
            record["answer"] += token
            yield token
    cited, invalid = validate_citations(record["answer"], len(record["retrieved"]))
    record.update(cited_numbers=cited, invalid_citations=invalid, complete=True)
