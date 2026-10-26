"""基础RAG命令行入口：向量检索、拼接上下文、千问流式回答。"""
import argparse

from kb import load_index, read_manifest
from rag_chain import MODEL, answer_baseline

def main():
    parser = argparse.ArgumentParser(description="云南大学软件学院 LangChain RAG")
    parser.add_argument("--top-k", type=int, default=5, help="基础链路检索数量，默认5")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须大于0")
    docs = [d for d in read_manifest() if d["status"] == "ready"]
    retriever = load_index()
    source_map = {d["doc_id"]: d for d in docs}
    print(f"云南大学软件学院 LangChain 知识库：{len(docs)} 篇文档、{len(retriever.chunks)} 个文本块；模型：{MODEL}。")
    print(f"基础RAG（FAISS top-{args.top_k}）")
    print("流式输出；输入 exit、quit 或 /bye 退出。")
    def emit_text(fragment):
        for character in fragment:
            print(character, end="", flush=True)

    def show_mode(mode):
        print(f"\n模式：{mode}\n回答：", end="", flush=True)

    while True:
        try:
            question = input("\n问题：").strip()
            if question.lower() in {"exit", "quit", "/bye"}:
                break
            if not question:
                continue
            print("正在检索知识库…", flush=True)
            answer, sources, mode = answer_baseline(
                question, retriever, k=args.top_k, on_token=emit_text, on_mode=show_mode,
            )
            print(flush=True)
            if sources:
                print("\n回答引用来源：")
                for i, chunk in sources:
                    print(f"[{i}] {chunk.title}（{chunk.chunk_id}）")
                    print("    " + source_map[chunk.doc_id]["content_url"])
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        except Exception as exc:
            print(f"\n问答失败：{exc}")
            print("请检查 Ollama 服务和模型名称；格式或引用错误可以重新提问。")


if __name__ == "__main__":
    main()
