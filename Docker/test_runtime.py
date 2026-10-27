"""Check source attribution, incremental delivery and interrupted streams."""
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
import runtime


class DeploymentAdapterTests(unittest.TestCase):
    def test_citations_are_deduplicated_and_out_of_range_are_separate(self):
        self.assertEqual(runtime.validate_citations("结论[2][2]。补充[1][0][8]。", 5), ([1, 2], [0, 8]))

    def test_tokens_arrive_before_completion_and_keep_original_source_numbers(self):
        doc = Document(page_content="先修课为数据结构。", metadata={
            "chunk_id": "course_1", "title": "课程大纲", "content_url": "https://example.test",
            "version": "公开大纲", "page": None})
        state = {}
        def events(_):
            yield {"documents": [doc]}
            yield {"answer": "先修课为"}
            self.assertEqual(state["answer"], "先修课为")
            self.assertFalse(state["complete"])
            yield {"answer": "数据结构。[1][8]"}
        chain = Mock()
        chain.stream.side_effect = events
        with patch.object(runtime.rag_chain, "build_rag_chain", return_value=chain):
            tokens = list(runtime.answer_stream("先修课？", Mock(), 5, state))
        self.assertEqual(len(tokens), 2)
        self.assertTrue(state["complete"])
        self.assertEqual(state["retrieved"][0]["chunk_id"], "course_1")
        self.assertEqual(state["cited_numbers"], [1])
        self.assertEqual(state["invalid_citations"], [8])

    def test_interruption_keeps_partial_answer_and_marks_incomplete(self):
        state = {}
        def events(_):
            yield {"answer": "部分回答"}
            raise ConnectionError("interrupted")
        chain = Mock()
        chain.stream.side_effect = events
        with patch.object(runtime.rag_chain, "build_rag_chain", return_value=chain):
            with self.assertRaises(ConnectionError):
                list(runtime.answer_stream("问题", Mock(), 5, state))
        self.assertEqual(state["answer"], "部分回答")
        self.assertFalse(state["complete"])


if __name__ == "__main__":
    unittest.main()
