"""Streamlit UI for the existing single-turn LangChain baseline."""
import threading
import time

import streamlit as st

from preflight import check_ollama
from runtime import answer_stream, kb, rag_chain

st.set_page_config(page_title="云南大学软件学院 · 知识库问答", page_icon="📚", layout="wide")


@st.cache_resource
def resources():
    return kb.load_index(), threading.Lock()


def render_evidence(record):
    if record.get("error"):
        st.error(record["error"])
    if record.get("invalid_citations"):
        st.warning("回答出现超出检索范围的引用编号：" +
                   ", ".join(map(str, record["invalid_citations"])) + "。这些引用无效，请核对原文。")
    cited = set(record.get("cited_numbers", []))
    used = [r for r in record.get("retrieved", []) if r["number"] in cited]
    if used:
        st.caption("回答引用来源（编号有效不代表结论已核实，请结合原文判断）")
        for r in used:
            st.markdown(f"**[{r['number']}] {r['title']}**")
            st.caption(f"{r['chunk_id']} · {r['version']}")
            st.markdown(f"[打开原文]({r['url']})")
    elif record.get("complete"):
        st.caption("本次回答没有有效的引用标注。检索候选见下方详情。")
    if record.get("retrieved"):
        with st.expander(f"查看本次检索的 {len(record['retrieved'])} 个候选片段"):
            for r in record["retrieved"]:
                st.markdown(f"**[{r['number']}] {r['title']}** · `{r['chunk_id']}`")
                st.caption(r["version"] + (f" · 原文件第 {r['page']} 页" if r["page"] else ""))
                st.text(r["text"])
    if "seconds" in record:
        st.caption(f"本次耗时 {record['seconds']:.1f} 秒 · top-{record['k']} · 单轮基础 RAG")


st.title("云南大学软件学院 · 知识库问答")
st.caption("培养方案、课程大纲、选课与推免政策｜回答逐步输出，引用可追溯")
try:
    knowledge_base, generation_lock = resources()
except Exception as exc:
    st.error(f"知识库加载失败：{exc}")
    st.stop()

with st.sidebar:
    st.header("知识库状态")
    st.write("模型：", rag_chain.MODEL)
    st.write(f"文档：{sum(r['status'] == 'ready' for r in kb.read_manifest())} 篇")
    st.write(f"文本块：{len(knowledge_base.chunks)} 个")
    k = st.slider("检索片段数量 top-k", 1, 10, 5)
    if st.button("检查模型连接", use_container_width=True):
        try:
            check_ollama()
            st.success("Ollama 连接正常，模型已存在。")
        except Exception as exc:
            st.error(str(exc))
    if st.button("清空问答记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("历史记录仅用于展示；每次回答独立检索。请注明专业、年级或政策年份。")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_evidence(message["record"])

question = st.chat_input("例如：云南大学数据库概论的先修课是什么？", max_chars=2000)
if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    record = {"answer": "", "retrieved": [], "k": k}
    with st.chat_message("assistant"):
        if not generation_lock.acquire(blocking=False):
            record["error"] = "模型正在处理其他请求，请稍后重新提问。"
        else:
            started = time.perf_counter()
            try:
                check_ollama()
                with st.spinner("正在检索并等待模型输出…"):
                    st.write_stream(answer_stream(question, knowledge_base, k, record))
            except Exception as exc:
                record["error"] = f"问答未完成：{exc}。已显示的部分回答请勿作为完整结论。"
            finally:
                record["seconds"] = time.perf_counter() - started
                generation_lock.release()
        render_evidence(record)
    st.session_state.messages.append({"role": "assistant", "content": record["answer"], "record": record})
