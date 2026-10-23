import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
import rag_chain

class StreamingTests(unittest.TestCase):
    def test_baseline_forwards_tokens_before_completion_and_keeps_source(self):
        seen = []
        doc = Document(page_content="先修课为数据结构。", metadata={
            "chunk_id": "course_0", "doc_id": "course", "title": "课程大纲",
            "category": "课程大纲", "version": "公开课程", "page": None,
            "content_url": "https://example.test/course",
        })
        def events(_):
            yield {"documents": [doc]}
            yield {"answer": "先修课为"}
            self.assertEqual(seen, ["先修课为"])
            yield {"answer": "数据结构。[1]"}
        chain = Mock()
        chain.stream.side_effect = events
        with patch.object(rag_chain, "build_rag_chain", return_value=chain):
            answer, sources, _ = rag_chain.answer_baseline("先修课？", Mock(), on_token=seen.append)
        self.assertEqual(answer, "".join(seen))
        self.assertEqual(sources[0][1].doc_id, "course")

    def test_partial_stream_failure_is_propagated(self):
        seen = []
        def events(_):
            yield {"answer": "部分文字"}
            raise ConnectionError("stream interrupted")
        chain = Mock()
        chain.stream.side_effect = events
        with patch.object(rag_chain, "build_rag_chain", return_value=chain):
            with self.assertRaises(ConnectionError):
                rag_chain.answer_baseline("问题", Mock(), on_token=seen.append)
        self.assertEqual(seen, ["部分文字"])

if __name__ == "__main__":
    unittest.main()
