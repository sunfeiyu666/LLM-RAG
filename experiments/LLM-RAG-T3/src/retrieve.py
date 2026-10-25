"""独立查看FAISS向量top-k检索结果，不调用生成模型。"""
import argparse
from kb import load_index, to_chunk

def retrieve(question, k=5):
    if k < 1:
        raise ValueError("k必须大于0")
    return [(to_chunk(doc), None) for doc in load_index().as_retriever(k=k).invoke(question)]

def main():
    parser = argparse.ArgumentParser(description="基础RAG向量检索")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k必须大于0")
    for i, (chunk, _) in enumerate(retrieve(args.question, args.top_k), 1):
        print(f"\n[{i}] {chunk.chunk_id}\n{chunk.context}")

if __name__ == "__main__":
    main()
