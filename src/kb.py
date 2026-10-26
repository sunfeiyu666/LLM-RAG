"""LangChain loading -> splitting -> embeddings -> FAISS index -> top-k.

The original TF-IDF/LSA vectorization is preserved behind the LangChain
Embeddings interface. It is deliberately not called a neural embedding model.
"""
import hashlib
import importlib.metadata
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import faiss
import joblib
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag_core import Chunk, dense_prepare

ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable, default):
    value = os.environ.get(variable)
    if not value:
        return ROOT / default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


INDEX = _configured_path("RAG_INDEX_DIR", "index_1024_128_langchain")
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "128"))
if CHUNK_SIZE < 1 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("分块参数必须满足 chunk_size > chunk_overlap >= 0")
EMBEDDING_NAME = "existing-character-TFIDF-LSA-64"



def read_manifest():
    return json.loads((ROOT / "metadata/sources.json").read_text(encoding="utf-8"))


def corpus_fingerprint():
    digest = hashlib.sha256((ROOT / "metadata/sources.json").read_bytes())
    for path in sorted((ROOT / "data/corpus").glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update((ROOT / "src/rag_core.py").read_bytes())
    for name in ["langchain-core", "langchain-community", "langchain-text-splitters", "scikit-learn", "faiss-cpu"]:
        digest.update(f"{name}={importlib.metadata.version(name)}".encode())
    return digest.hexdigest()


class LSAEmbeddings(Embeddings):
    """Adapt the existing vectorizer to the LangChain embedding contract."""
    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 4), min_df=1, sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform([dense_prepare(t) for t in texts])
        dimensions = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if dimensions < 1:
            raise ValueError("语料不足，无法训练现有LSA向量化器。")
        self.svd = TruncatedSVD(n_components=dimensions, random_state=42)
        self.svd.fit(matrix)
        self.dimension = dimensions
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform([dense_prepare(t) for t in texts])
        return normalize(self.svd.transform(matrix)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_pages():
    """1. Load the same 23 Markdown files with LangChain TextLoader."""
    pages = []
    for row in read_manifest():
        if row["status"] != "ready":
            continue
        document = TextLoader(str(ROOT / row["corpus_file"]), encoding="utf-8").load()[0]
        if "<!-- BODY -->" not in document.page_content:
            raise ValueError(f"缺少正文边界：{row['doc_id']}")
        body = document.page_content.split("<!-- BODY -->", 1)[1].strip()
        sections = re.split(r"(?m)^## 原文件第 (\d+) 页\s*$", body)
        parts = [(None, sections[0])] if len(sections) == 1 else [
            (int(sections[i]), sections[i + 1]) for i in range(1, len(sections), 2)
        ]
        for page, text in parts:
            text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text).strip()
            if not text:
                continue
            metadata = {
                **document.metadata, "doc_id": row["doc_id"], "title": row["title"],
                "category": row["category"], "version": row["version"],
                "source_url": row["url"], "content_url": row["content_url"], "page": page,
            }
            # Include the document title in embedding text, as the old index did.
            pages.append(Document(page_content=row["title"] + "\n" + text, metadata=metadata))
    return pages


def split_documents(pages):
    """2. Split via LangChain while preserving PDF page and source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        meta["chunk_id"] = f"{meta['doc_id']}_p{meta['page'] or 0:03d}_{i:05d}"
        meta["scope_years"] = re.findall(r"20\d{2}", meta["version"].split("；", 1)[0])
    return chunks


def format_document(document):
    meta = document.metadata
    location = f"原文件第{meta['page']}页" if meta.get("page") else "官网网页正文"
    return (
        f"文档：{meta['title']}\n类别：{meta['category']}\n"
        f"版本与范围：{meta['version']}\n定位：{location}\n"
        f"来源：{meta['content_url']}\n正文：\n{document.page_content}"
    )


def to_chunk(document):
    meta = document.metadata
    return Chunk(meta["chunk_id"], meta["doc_id"], meta["title"], document.page_content, format_document(document))


class LangChainKnowledgeBase:
    """Expose the LangChain vector similarity retriever."""
    def __init__(self, store, documents):
        self.store = store
        self.documents = documents
        self.chunks = [to_chunk(d) for d in documents]

    def as_retriever(self, k=5):
        # 5. The homework baseline uses direct vector similarity top-k.
        return self.store.as_retriever(search_type="similarity", search_kwargs={"k": k})



def build_index():
    INDEX.mkdir(exist_ok=True)
    pages = load_pages()
    documents = split_documents(pages)
    if not documents:
        raise ValueError("知识库正文为空")
    # 3. Vectorize using the unchanged TF-IDF/LSA algorithm, via Embeddings.
    embeddings = LSAEmbeddings().fit([d.page_content for d in documents])
    # 4. Build a real FAISS index through the LangChain vector-store integration.
    store = FAISS.from_documents(documents, embeddings, normalize_L2=True)
    # FAISS's C++ filename API cannot open Chinese Windows paths reliably.
    # Python owns file I/O, while FAISS still serializes the actual index.
    (INDEX / "index.faiss").write_bytes(faiss.serialize_index(store.index).tobytes())
    joblib.dump(embeddings, INDEX / "embeddings.joblib")
    records = [{"id": store.index_to_docstore_id[i], "page_content": d.page_content, "metadata": d.metadata}
               for i, d in enumerate(documents)]
    (INDEX / "documents.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (INDEX / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    info = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": sum(r["status"] == "ready" for r in read_manifest()),
        "page_documents": len(pages), "chunks": len(documents),
        "loader": "langchain_community.document_loaders.TextLoader",
        "splitter": "RecursiveCharacterTextSplitter", "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
        "embedding": EMBEDDING_NAME, "neural_embedding": False, "dimension": embeddings.dimension,
        "vector_store": "LangChain FAISS / IndexFlatL2", "default_top_k": 5,
        "generation_model": "qwen3:4b-instruct", "fingerprint": corpus_fingerprint(),
        "package_versions": {name: importlib.metadata.version(name) for name in [
            "langchain", "langchain-core", "langchain-community", "langchain-text-splitters",
            "langchain-ollama", "faiss-cpu", "numpy", "scikit-learn", "joblib"]},
    }
    (INDEX / "index_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LangChain索引已建立：{info['documents']}篇文档，{len(documents)}个分块，{embeddings.dimension}维LSA，FAISS。")
    return LangChainKnowledgeBase(store, documents)


def load_index():
    info_file = INDEX / "index_info.json"
    if info_file.exists() and all((INDEX / f).exists() for f in ["index.faiss", "embeddings.joblib", "documents.json"]):
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if info["fingerprint"] == corpus_fingerprint():
            embeddings = joblib.load(INDEX / "embeddings.joblib")
            records = json.loads((INDEX / "documents.json").read_text(encoding="utf-8"))
            documents = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in records]
            docstore = InMemoryDocstore({r["id"]: d for r, d in zip(records, documents)})
            index_data = np.frombuffer((INDEX / "index.faiss").read_bytes(), dtype=np.uint8).copy()
            store = FAISS(embeddings, faiss.deserialize_index(index_data), docstore,
                          {i: r["id"] for i, r in enumerate(records)}, normalize_L2=True)
            return LangChainKnowledgeBase(store, documents)
    return build_index()


if __name__ == "__main__":
    from kb import build_index as build_from_module
    build_from_module()
