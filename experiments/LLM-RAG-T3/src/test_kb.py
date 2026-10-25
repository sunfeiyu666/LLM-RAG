import hashlib
import unittest

from kb import ROOT, load_index, read_manifest


class CorpusTests(unittest.TestCase):
    def test_documents_are_distinct_and_match_archived_hash(self):
        rows = [r for r in read_manifest() if r["status"] == "ready"]
        self.assertTrue(20 <= len(rows) <= 50)
        self.assertEqual(len(rows), len({r["text_sha256"] for r in rows}))
        for row in rows:
            text = (ROOT / row["corpus_file"]).read_text(encoding="utf-8").split("<!-- BODY -->", 1)[1].strip()
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), row["text_sha256"], row["doc_id"])
            self.assertNotIn("请输入验证码", text)
            self.assertNotIn("\ufffd", text)

    def test_pending_attachment_is_not_indexed(self):
        ids = {c.doc_id for c in load_index().chunks}
        self.assertNotIn("ynu013", ids)
        self.assertNotIn("ynu023", ids)  # archived historical reception policy


if __name__ == "__main__":
    unittest.main()
