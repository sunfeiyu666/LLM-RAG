"""T4 的本地 BGE 中文 Embedding + FAISS 检索分支。"""

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from kb import ROOT, corpus_fingerprint, load_index


MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_DIR = ROOT / "models/bge-small-zh-v1.5"
_index_value = os.environ.get("RAG_BGE_INDEX_DIR")
INDEX_DIR = (Path(_index_value) if _index_value and Path(_index_value).is_absolute()
             else ROOT / (_index_value or "index_1024_128_bge"))
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def model_fingerprint():
    """对实际使用的模型文件生成摘要，防止权重变化后误用旧索引。"""
    digest = hashlib.sha256()
    files = sorted(path for path in MODEL_DIR.rglob("*") if path.is_file() and ".cache" not in path.parts)
    if not files:
        raise FileNotFoundError(f"本地 BGE 模型不存在：{MODEL_DIR}")
    for path in files:
        digest.update(path.relative_to(MODEL_DIR).as_posix().encode("utf-8"))
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def download_model():
    """把官方模型的固定版本下载到项目目录，后续实验可离线重跑。"""
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(MODEL_ID).sha
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=revision, local_dir=MODEL_DIR)
    (MODEL_DIR / "model_revision.txt").write_text(revision + "\n", encoding="ascii")
    print(f"已下载 {MODEL_ID}，版本 {revision}，位置 {MODEL_DIR}")


class BGERetriever:
    """使用与原 LSA 索引完全相同的文本块，只更换 Embedding。"""

    def __init__(self, knowledge_base=None, *, device="cpu"):
        from sentence_transformers import SentenceTransformer

        self.kb = knowledge_base or load_index()
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"先执行 python {Path(__file__).name} --download 下载模型")
        self.model = SentenceTransformer(str(MODEL_DIR), device=device, local_files_only=True)
        self.documents = self.kb.documents
        self.chunk_ids = [doc.metadata["chunk_id"] for doc in self.documents]
        self.fingerprint = model_fingerprint()
        self.index = self._load_or_build_index()

    def _load_or_build_index(self):
        info_path = INDEX_DIR / "index_info.json"
        data_path = INDEX_DIR / "index.faiss"
        expected = {
            "model": MODEL_ID,
            "model_fingerprint": self.fingerprint,
            "corpus_fingerprint": corpus_fingerprint(),
            "chunk_ids": self.chunk_ids,
            "query_prefix": QUERY_PREFIX,
        }
        if info_path.exists() and data_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if all(info.get(key) == value for key, value in expected.items()):
                data = np.frombuffer(data_path.read_bytes(), dtype=np.uint8).copy()
                index = faiss.deserialize_index(data)
                if index.ntotal == len(self.documents):
                    return index

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectors = self.model.encode(
            [doc.page_content for doc in self.documents],
            batch_size=16, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        data_path.write_bytes(faiss.serialize_index(index).tobytes())
        info_path.write_text(json.dumps({
            **expected,
            "dimension": int(vectors.shape[1]),
            "documents": len(self.documents),
            "similarity": "normalized inner product (cosine)",
            "model_revision": (MODEL_DIR / "model_revision.txt").read_text(encoding="ascii").strip()
            if (MODEL_DIR / "model_revision.txt").exists() else None,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return index

    @lru_cache(maxsize=128)
    def _ranked_indices(self, question):
        vector = self.model.encode(
            [QUERY_PREFIX + question], convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        _, indices = self.index.search(vector, len(self.documents))
        return tuple(int(i) for i in indices[0] if i >= 0)

    def search(self, question, k=5):
        return [self.documents[i] for i in self._ranked_indices(question)[:k]]


def main():
    parser = argparse.ArgumentParser(description="下载或建立本地 BGE + FAISS 索引")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.download:
        download_model()
    if args.build:
        retriever = BGERetriever(device=args.device)
        print(f"BGE 索引可用：{len(retriever.documents)} 个文本块，{retriever.index.d} 维")
    if not args.download and not args.build:
        parser.error("至少指定 --download 或 --build")


if __name__ == "__main__":
    main()
