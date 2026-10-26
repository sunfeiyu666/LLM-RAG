"""T4：对比 BM25、FAISS 稠密检索和 RRF 混合检索。

稠密分支默认保留原有 TF-IDF/LSA；传入 bge 后端可改用本地神经 Embedding。
"""

import argparse
import math
import re
from collections import Counter

from kb import load_index


def tokenize(text):
    """提取中文双字词和英文数字词，无需额外分词依赖。"""
    words = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.-]+", text.lower())
    tokens = []
    for word in words:
        if "\u4e00" <= word[0] <= "\u9fff":
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                tokens.append(word)
        else:
            tokens.append(word)
    return tokens


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.terms = [Counter(tokenize(d.page_content)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.terms]
        self.average_length = sum(self.lengths) / len(documents) if documents else 0
        self.df = Counter(term for counts in self.terms for term in counts)

    def search(self, question, k):
        if not self.documents:
            return []
        query_terms = set(tokenize(question))
        n = len(self.documents)
        scores = []
        for i, counts in enumerate(self.terms):
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average_length)
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        return [self.documents[i] for i in order[:k]]


def reciprocal_rank_fusion(rankings, k, constant=60):
    """按文本块编号融合多路排序，并保持结果顺序可复现。"""
    scores, documents = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            chunk_id = doc.metadata["chunk_id"]
            documents[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (constant + rank)
    ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [documents[chunk_id] for chunk_id in ids[:k]]


class T4Retrievers:
    def __init__(self, knowledge_base=None, *, dense_backend="lsa"):
        self.kb = knowledge_base or load_index()
        self.bm25 = BM25(self.kb.documents)
        if dense_backend not in ("lsa", "bge"):
            raise ValueError(f"未知的稠密检索后端：{dense_backend}")
        self.dense_backend = dense_backend
        self.bge = None
        if dense_backend == "bge":
            from neural_retrieval import BGERetriever
            self.bge = BGERetriever(self.kb)

    def dense_search(self, question, k):
        if self.bge is not None:
            return self.bge.search(question, k)
        return self.kb.as_retriever(k=k).invoke(question)

    def search(self, question, method="hybrid", k=5):
        if k < 1:
            raise ValueError("k 必须大于 0")
        if method == "bm25":
            return self.bm25.search(question, k)
        if method == "dense":
            return self.dense_search(question, k)
        if method == "hybrid":
            pool = max(20, k)
            return reciprocal_rank_fusion([
                self.bm25.search(question, pool),
                self.dense_search(question, pool),
            ], k)
        raise ValueError(f"未知的检索方法：{method}")


def main():
    parser = argparse.ArgumentParser(description="T4 检索方法对比")
    parser.add_argument("question")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--dense-backend", choices=["lsa", "bge"], default="lsa")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    for rank, doc in enumerate(T4Retrievers(dense_backend=args.dense_backend).search(args.question, args.method, args.top_k), 1):
        meta = doc.metadata
        print(f"[{rank}] {meta['chunk_id']} | {meta['title']} | {doc.page_content[:160]}")


if __name__ == "__main__":
    main()
