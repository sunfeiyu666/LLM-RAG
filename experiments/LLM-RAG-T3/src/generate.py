"""基础生成与提示链入口；实际LCEL链见rag_chain.py。"""
from rag_chain import MODEL, BASE_URL, build_rag_chain, answer_baseline, format_context

__all__ = ["MODEL", "BASE_URL", "build_rag_chain", "answer_baseline", "format_context"]
