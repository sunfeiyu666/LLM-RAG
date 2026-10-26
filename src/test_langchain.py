import unittest

import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeStreamingListLLM
from langchain_core.runnables import RunnableLambda

from kb import CHUNK_SIZE, load_index, load_pages, split_documents
from rag_chain import build_rag_chain


class LangChainPipelineTests(unittest.TestCase):
    def test_load_and_split_keep_original_document_identity_and_pdf_page(self):
        pages = load_pages()
        chunks = split_documents(pages)
        self.assertEqual(len({p.metadata["doc_id"] for p in pages}), 23)
        self.assertTrue(all(isinstance(c, Document) for c in chunks))
        self.assertTrue(all(len(c.page_content) <= CHUNK_SIZE for c in chunks))
        self.assertTrue(all("page" in c.metadata and "content_url" in c.metadata for c in chunks))
        self.assertFalse(any(c.metadata["doc_id"] in {"ynu013", "ynu023"} for c in chunks))

    def test_real_faiss_reload_and_embedding_contract(self):
        kb = load_index()
        self.assertEqual(kb.store.index.ntotal, len(kb.documents))
        vector = kb.store.embedding_function.embed_query("数据库概论先修课程")
        self.assertEqual(len(vector), 64)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        hits = kb.as_retriever(k=2).invoke("云南大学数据库概论先修课")
        self.assertEqual(len(hits), 2)
        self.assertIn("ynu006", {d.metadata["doc_id"] for d in hits})

    def test_lcel_passes_retrieved_text_to_prompt_and_streams_answer(self):
        docs = [Document(page_content="本课程先修课为数据结构。", metadata={
            "title": "数据库概论", "category": "课程大纲", "version": "公开课程",
            "content_url": "https://example.test/course", "page": None,
        })]
        questions = []
        def retrieve(question):
            questions.append(question)
            return docs
        llm = FakeStreamingListLLM(responses=["先修课为数据结构。[1]"])
        chain = build_rag_chain(RunnableLambda(retrieve), llm=llm)
        events = list(chain.stream({"question": "先修课是什么？"}))
        output = {}
        answer_parts = []
        for event in events:
            output.update(event)
            if "answer" in event:
                answer_parts.append(event["answer"])
        self.assertEqual(questions, ["先修课是什么？"])
        self.assertEqual(output["documents"], docs)
        self.assertIn("[1]", output["context"])
        self.assertIn("数据结构", output["context"])
        self.assertGreater(len(answer_parts), 1)
        self.assertEqual("".join(answer_parts), "先修课为数据结构。[1]")


if __name__ == "__main__":
    unittest.main()
