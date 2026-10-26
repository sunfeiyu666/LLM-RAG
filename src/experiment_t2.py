"""T2 分块策略网格实验。

固定检索模型（现有 TF-IDF/LSA + FAISS）和 top-k，只改变分块策略。
默认运行 4 种方法 x 3 种 chunk_size，共 12 组配置。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

from kb import LSAEmbeddings, ROOT, format_document, load_pages


METHOD_LABELS = {
    "fixed": "固定长度",
    "sliding": "滑动窗口",
    "semantic": "语义分块",
    "small_to_big": "Small-to-Big",
}
PLOT_LABELS = {
    "fixed": "Fixed length",
    "sliding": "Sliding window",
    "semantic": "Semantic",
    "small_to_big": "Small-to-Big",
}
SIZES = (256, 512, 1024)


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    chunk_size: int
    chunk_overlap: int

    @property
    def name(self) -> str:
        return f"{self.method}_{self.chunk_size}"


def default_configs() -> list[ExperimentConfig]:
    configs = []
    for method in METHOD_LABELS:
        for size in SIZES:
            overlap = 128 if method == "sliding" else 0
            configs.append(ExperimentConfig(method, size, overlap))
    return configs


def copy_document(document: Document, text: str, **metadata) -> Document:
    return Document(page_content=text, metadata={**document.metadata, **metadata})


def character_windows(text: str, size: int, overlap: int = 0) -> Iterable[tuple[int, str]]:
    if overlap >= size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            yield start, piece
        if start + size >= len(text):
            break


def split_fixed(pages: list[Document], config: ExperimentConfig) -> list[Document]:
    chunks = []
    for page in pages:
        for start, text in character_windows(page.page_content, config.chunk_size):
            chunks.append(copy_document(page, text, start_index=start))
    return chunks


def split_sliding(pages: list[Document], config: ExperimentConfig) -> list[Document]:
    chunks = []
    overlap = min(config.chunk_overlap, config.chunk_size // 2)
    for page in pages:
        for start, text in character_windows(page.page_content, config.chunk_size, overlap):
            chunks.append(copy_document(page, text, start_index=start))
    return chunks


def sentence_units(text: str, max_size: int) -> list[str]:
    """按中文句末标点和段落切成语义候选单元，超长单元再做硬切。"""
    rough = re.findall(r".*?(?:[。！？；]\s*|\n{2,}|$)", text, flags=re.S)
    units = []
    for unit in rough:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) <= max_size:
            units.append(unit)
        else:
            units.extend(piece for _, piece in character_windows(unit, max_size))
    return units


def semantic_page_chunks(text: str, target_size: int) -> list[tuple[int, str]]:
    """在相邻句子词面相似度低的位置优先断开，再受目标长度约束。"""
    units = sentence_units(text, target_size)
    if len(units) <= 1:
        return [(0, text.strip())] if text.strip() else []

    matrix = TfidfVectorizer(analyzer="char", ngram_range=(1, 2)).fit_transform(units)
    similarities = np.asarray(matrix[:-1].multiply(matrix[1:]).sum(axis=1)).ravel()
    boundary = float(np.quantile(similarities, 0.35)) if similarities.size else 0.0
    min_size = max(64, target_size // 2)
    result: list[tuple[int, str]] = []
    current: list[str] = []
    current_len = 0
    cursor = 0
    start = 0

    for index, unit in enumerate(units):
        would_overflow = current and current_len + len(unit) > target_size
        topic_shift = (
            current
            and current_len >= min_size
            and index > 0
            and similarities[index - 1] <= boundary
        )
        if would_overflow or topic_shift:
            result.append((start, "\n".join(current)))
            start = cursor
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit)
        cursor += len(unit)
    if current:
        result.append((start, "\n".join(current)))
    return result


def split_semantic(pages: list[Document], config: ExperimentConfig) -> list[Document]:
    chunks = []
    for page in pages:
        for start, text in semantic_page_chunks(page.page_content, config.chunk_size):
            chunks.append(copy_document(page, text, start_index=start))
    return chunks


def split_small_to_big(pages: list[Document], config: ExperimentConfig) -> list[Document]:
    """以小块建索引；命中后把所属大块作为生成上下文。"""
    parent_size = min(3072, max(1024, config.chunk_size * 2))
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=min(128, parent_size // 8),
        add_start_index=True,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        keep_separator=True,
    )
    parents = parent_splitter.split_documents(pages)
    children = []
    child_overlap = min(64, config.chunk_size // 4)
    for parent_index, parent in enumerate(parents):
        parent_id = f"{parent.metadata['doc_id']}_parent_{parent_index:05d}"
        for child_start, child_text in character_windows(
            parent.page_content, config.chunk_size, child_overlap
        ):
            children.append(
                copy_document(
                    parent,
                    child_text,
                    parent_id=parent_id,
                    parent_content=parent.page_content,
                    child_start_index=child_start,
                    parent_size=parent_size,
                )
            )
    return children


def assign_chunk_ids(documents: list[Document], config: ExperimentConfig) -> list[Document]:
    counters: dict[tuple[str, int], int] = {}
    for document in documents:
        page = int(document.metadata.get("page") or 0)
        key = (document.metadata["doc_id"], page)
        local_index = counters.get(key, 0)
        counters[key] = local_index + 1
        document.metadata.update(
            chunk_id=(
                f"{document.metadata['doc_id']}_p{page:03d}_"
                f"{config.method}_{config.chunk_size}_{local_index:05d}"
            ),
            split_method=config.method,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    return documents


def split_for_config(pages: list[Document], config: ExperimentConfig) -> list[Document]:
    splitters = {
        "fixed": split_fixed,
        "sliding": split_sliding,
        "semantic": split_semantic,
        "small_to_big": split_small_to_big,
    }
    return assign_chunk_ids(splitters[config.method](pages, config), config)


def infer_doc_id(chunk_id: str) -> str | None:
    match = re.match(r"(ynu\d+)", chunk_id)
    return match.group(1) if match else None


def normalize_question(row: dict, index: int) -> dict:
    chunk_ids = row.get("gold_chunk_ids") or []
    doc_ids = row.get("gold_doc_ids") or []
    if not doc_ids:
        doc_ids = sorted({doc for cid in chunk_ids if (doc := infer_doc_id(str(cid)))})
    evidence = row.get("gold_evidence", row.get("evidence", row.get("evidence_text", [])))
    if isinstance(evidence, str):
        evidence = [evidence]
    return {
        "id": row.get("id", f"q{index:03d}"),
        "level": row.get("level", "unknown"),
        "question": row["question"],
        "gold_answer": row.get("gold_answer"),
        "gold_chunk_ids": [str(value) for value in chunk_ids],
        "gold_doc_ids": [str(value) for value in doc_ids],
        "gold_evidence": [str(value) for value in evidence if str(value).strip()],
    }


def load_questions(path: Path | None) -> tuple[list[dict], str]:
    if path and path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        evidence_path = path.with_name("qa_evidence.jsonl")
        if evidence_path.exists():
            evidence_rows = [
                json.loads(line)
                for line in evidence_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            evidence_by_id = {row["id"]: row for row in evidence_rows}
            for row in rows:
                annotation = evidence_by_id.get(row.get("id"), {})
                evidence = annotation.get("evidence", [])
                row["gold_evidence"] = [item["quote"] for item in evidence if item.get("quote")]
                row["gold_doc_ids"] = sorted(
                    {item["doc_id"] for item in evidence if item.get("doc_id")}
                )
        return [normalize_question(row, i) for i, row in enumerate(rows, 1)], "formal_qa_set"

    from check_retrieval import CASES

    rows = []
    for index, (level, question, gold_docs, needle) in enumerate(CASES, 1):
        rows.append(
            normalize_question(
                {
                    "id": f"smoke{index:03d}",
                    "level": level,
                    "question": question,
                    "gold_answer": needle,
                    "gold_doc_ids": gold_docs,
                    "gold_evidence": [needle] if needle else [],
                },
                index,
            )
        )
    return rows, "development_smoke_set"


def retrieve(store: FAISS, question: str, config: ExperimentConfig, k: int) -> list[Document]:
    candidates = store.similarity_search(question, k=max(k, 20) if config.method == "small_to_big" else k)
    if config.method != "small_to_big":
        return candidates[:k]
    result, seen = [], set()
    for document in candidates:
        parent_id = document.metadata["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        result.append(document)
        if len(result) == k:
            break
    return result


def context_content(document: Document) -> str:
    return document.metadata.get("parent_content", document.page_content)


def retrieval_hit(question: dict, documents: list[Document]) -> tuple[bool, str]:
    contexts = [context_content(document) for document in documents]
    if question["gold_evidence"]:
        compact_contexts = [re.sub(r"\s+", "", context) for context in contexts]
        return any(
            re.sub(r"\s+", "", evidence) in context
            for evidence in question["gold_evidence"]
            for context in compact_contexts
        ), "evidence_text"

    retrieved_ids = {document.metadata["chunk_id"] for document in documents}
    if question["gold_chunk_ids"] and retrieved_ids.intersection(question["gold_chunk_ids"]):
        return True, "exact_chunk_id"

    retrieved_docs = {document.metadata["doc_id"] for document in documents}
    if question["gold_doc_ids"]:
        return bool(retrieved_docs.intersection(question["gold_doc_ids"])), "document_proxy"
    raise ValueError(f"{question['id']} 没有可用于检索评估的证据、chunk ID 或 doc ID")


def normalized_chars(text: str | None) -> list[str]:
    return list(re.sub(r"\s+|[^0-9A-Za-z\u4e00-\u9fff.%]", "", text or "").lower())


def rouge_l_f1(answer: str, gold: str | None) -> float | None:
    left, right = normalized_chars(answer), normalized_chars(gold)
    if not right:
        return None
    previous = [0] * (len(right) + 1)
    for char in left:
        current = [0]
        for j, gold_char in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if char == gold_char else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    return (2 * lcs / (len(left) + len(right))) if left or right else 0.0


def generate_answer(question: str, documents: list[Document], model: str) -> str:
    from langchain_ollama import ChatOllama

    context = "\n\n".join(
        f"[{index}] {format_document(copy_document(doc, context_content(doc)))}"
        for index, doc in enumerate(documents, 1)
    )
    prompt = (
        "你是云南大学软件学院知识库助手。只根据参考资料简洁回答，并用[序号]标注依据；"
        "资料不足时明确说明。\n\n"
        f"参考资料：\n{context}\n\n问题：{question}"
    )
    response = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0.0).invoke(prompt)
    return str(response.content)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def mean_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def run_config(
    pages: list[Document], questions: list[dict], config: ExperimentConfig,
    output_dir: Path, top_k: int, generate: bool, model: str,
) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    documents = split_for_config(pages, config)
    embeddings = LSAEmbeddings().fit([document.page_content for document in documents])
    store = FAISS.from_documents(documents, embeddings, normalize_L2=True)
    index_seconds = time.perf_counter() - started

    rows = []
    for question in questions:
        retrieve_started = time.perf_counter()
        hits = retrieve(store, question["question"], config, top_k)
        retrieval_ms = (time.perf_counter() - retrieve_started) * 1000
        hit, recall_basis = retrieval_hit(question, hits)
        answer = generate_answer(question["question"], hits, model) if generate else ""
        correctness = rouge_l_f1(answer, question["gold_answer"]) if generate else (
            1.0 if hit else 0.0
        )
        rows.append(
            {
                "question_id": question["id"],
                "level": question["level"],
                "question": question["question"],
                "method": config.method,
                "method_label": METHOD_LABELS[config.method],
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retrieved_chunk_ids": [doc.metadata["chunk_id"] for doc in hits],
                "retrieved_doc_ids": [doc.metadata["doc_id"] for doc in hits],
                "retrieved_contexts": [context_content(doc) for doc in hits],
                "recall_at_5": int(hit),
                "recall_basis": recall_basis,
                "answer": answer,
                "gold_answer": question["gold_answer"],
                "answer_correctness": correctness,
                "correctness_basis": "rouge_l_f1" if generate else "retrieval_evidence_proxy",
                "retrieval_ms": round(retrieval_ms, 3),
            }
        )

    write_jsonl(output_dir / f"{config.name}.jsonl", rows)
    summary = {
        **asdict(config),
        "config": config.name,
        "method_label": METHOD_LABELS[config.method],
        "chunks": len(documents),
        "questions": len(rows),
        "recall_at_5": sum(row["recall_at_5"] for row in rows) / len(rows),
        "answer_correctness": mean_present(row["answer_correctness"] for row in rows),
        "index_seconds": round(index_seconds, 3),
        "avg_retrieval_ms": round(sum(row["retrieval_ms"] for row in rows) / len(rows), 3),
        "correctness_basis": "rouge_l_f1" if generate else "retrieval_evidence_proxy",
    }
    return summary, rows


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def make_plots(output_dir: Path, summaries: list[dict]) -> None:
    import matplotlib.pyplot as plt

    for metric, filename, ylabel in [
        ("recall_at_5", "recall_at_5.png", "Recall@5"),
        ("answer_correctness", "answer_correctness.png", "Answer correctness"),
    ]:
        figure, axis = plt.subplots(figsize=(8, 5))
        for method, label in PLOT_LABELS.items():
            rows = sorted((row for row in summaries if row["method"] == method), key=lambda row: row["chunk_size"])
            axis.plot([row["chunk_size"] for row in rows], [row[metric] for row in rows], marker="o", label=label)
        axis.set_xlabel("Chunk size (characters)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(SIZES)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def write_failure_case(output_dir: Path, all_rows: list[dict]) -> None:
    by_question: dict[str, list[dict]] = {}
    for row in all_rows:
        by_question.setdefault(row["question_id"], []).append(row)
    selected = None
    for rows in by_question.values():
        failed = [row for row in rows if not row["recall_at_5"]]
        succeeded = [row for row in rows if row["recall_at_5"]]
        if failed and succeeded:
            selected = {"failed": failed[0], "succeeded": succeeded[0]}
            break
    if selected is None:
        failed = [row for row in all_rows if not row["recall_at_5"]]
        if failed:
            selected = {"failed": failed[0], "succeeded": None}
    if selected is None:
        selected = {"note": "本次评测所有配置均命中，正式评测集加入后重新运行以寻找切分失败案例。"}
    (output_dir / "failure_case.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_report(output_dir: Path, summaries: list[dict], dataset_name: str, generated: bool) -> None:
    best = max(summaries, key=lambda row: (row["recall_at_5"], row["answer_correctness"] or 0))
    lines = [
        "# T2 分块策略实验结果\n",
        f"- 评测集：`{dataset_name}`",
        f"- 配置数：{len(summaries)}",
        "- 控制变量：语料、TF-IDF/LSA 向量化、FAISS、top-k=5 均保持不变。",
        f"- 答案正确性：{'ROUGE-L F1' if generated else '检索证据命中代理分；正式报告需使用生成答案重跑'}。\n",
        "| 方法 | chunk_size | overlap | 分块数 | Recall@5 | 答案正确性 | 建索引/秒 | 平均检索/ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method_label']} | {row['chunk_size']} | {row['chunk_overlap']} | "
            f"{row['chunks']} | {row['recall_at_5']:.3f} | {row['answer_correctness']:.3f} | "
            f"{row['index_seconds']:.3f} | {row['avg_retrieval_ms']:.3f} |"
        )
    lines.extend(
        [
            "\n## 当前最优配置\n",
            f"按 Recall@5 优先、答案正确性次优排序，当前最优配置为 **{best['method_label']} / "
            f"chunk_size={best['chunk_size']} / overlap={best['chunk_overlap']}**，Recall@5={best['recall_at_5']:.3f}。",
            "\n最终报告应结合 `failure_case.json` 中的原始召回片段，说明过小切块是否丢失完整语义、"
            "过大切块是否引入噪声，以及重叠或 Small-to-Big 是否改善证据完整性。",
        ]
    )
    (output_dir / "T2实验报告草稿.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 T2 分块策略网格实验")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data/eval/qa_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/t2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="调用本地 Ollama 生成答案，并计算 ROUGE-L F1")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--config", choices=[config.name for config in default_configs()], help="只运行一组配置")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions, dataset_name = load_questions(args.eval_file)
    configs = [config for config in default_configs() if args.config in (None, config.name)]
    pages = load_pages()
    summaries, all_rows = [], []
    for config in configs:
        print(f"运行 {config.name} ...", flush=True)
        summary, rows = run_config(
            pages, questions, config, args.output_dir, args.top_k, args.generate, args.model
        )
        summaries.append(summary)
        all_rows.extend(rows)
        print(f"  chunks={summary['chunks']} Recall@5={summary['recall_at_5']:.3f}")

    write_summary_csv(args.output_dir / "t2_summary.csv", summaries)
    make_plots(args.output_dir, summaries)
    write_failure_case(args.output_dir, all_rows)
    write_report(args.output_dir, summaries, dataset_name, args.generate)
    metadata = {
        "dataset": dataset_name,
        "eval_file": str(args.eval_file),
        "top_k": args.top_k,
        "generated_answers": args.generate,
        "model": args.model if args.generate else None,
        "configs": [asdict(config) for config in configs],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成，结果位于：{args.output_dir}")


if __name__ == "__main__":
    main()
