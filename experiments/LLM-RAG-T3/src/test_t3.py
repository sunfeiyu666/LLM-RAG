import unittest
import json
from langchain_core.documents import Document
from t3_core import BM25, lexical_tokens, rrf, score_fusion, retrieval_metrics
from run_t3 import judge


class T3Tests(unittest.TestCase):
    def test_weighted_rrf_endpoints_and_no_zero_weight_leakage(self):
        d = [{'chunk_id': x, 'score': s} for x, s in [('a', 0.9), ('b', 0.7)]]
        s = [{'chunk_id': x, 'score': v} for x, v in [('b', 10), ('c', 5)]]
        self.assertEqual([x['chunk_id'] for x in rrf([d, s], [1, 0])], ['a', 'b'])
        self.assertEqual([x['chunk_id'] for x in rrf([d, s], [0, 1])], ['b', 'c'])
        self.assertEqual(rrf([d, s], [0.5, 0.5])[0]['chunk_id'], 'b')
        self.assertAlmostEqual(rrf([d, s], [0.5, 0.5])[0]['score'], 0.5/62+0.5/61)

    def test_course_recall_and_gold_coverage_are_distinct(self):
        m = retrieval_metrics(['a', 'z'], ['a', 'b'])
        self.assertEqual(m['recall_at_5'], 1)
        self.assertEqual(m['gold_coverage_at_5'], 0.5)
        self.assertEqual(retrieval_metrics(['z', 'a'], ['a'])['reciprocal_rank_at_5'], 0.5)

    def test_bm25_preserves_years_and_identifier(self):
        docs = [Document(page_content=t, metadata={'chunk_id': str(i)}) for i, t in enumerate(['2025年推免GPA要求', '2027年推免GPA要求'])]
        rank = BM25(docs).rank('2027 GPA')
        self.assertEqual(rank[0]['chunk_id'], '1')
        self.assertIn('2027', lexical_tokens('2027 GPA'))
        self.assertIn('gpa', lexical_tokens('2027 GPA'))

    def test_raw_score_fusion_normalizes_different_scales(self):
        d = [{'chunk_id': 'a', 'score': 0.9}, {'chunk_id': 'b', 'score': 0.1}]
        s = [{'chunk_id': 'b', 'score': 1000}, {'chunk_id': 'a', 'score': 1}]
        self.assertEqual(score_fusion(d, s, 0.7)[0]['chunk_id'], 'a')
        self.assertEqual(score_fusion(d, s, 0.3)[0]['chunk_id'], 'b')

    def test_identical_answers_receive_one_shared_blinded_score(self):
        class FakeJudge:
            count = 0
            def call(self, system, user, **kwargs):
                candidates = json.loads(user)['candidates']
                self.count = len(candidates)
                return json.dumps({'scores': [{'label': c['label'], 'score': 3, 'reason': 'test'} for c in candidates]}), {}
        fake = FakeJudge()
        answers = {'baseline': {'answer': '相同答案'}, 'rrf_1': {'answer': '相同答案'}, 'rewrite': {'answer': '另一答案'}}
        result = judge(fake, {'id': 'q001', 'level': 'L1', 'question': 'test', 'gold_answer': 'gold'}, answers)
        self.assertEqual(fake.count, 2)
        self.assertEqual(set(result['scores']), set(answers))
        self.assertEqual(result['scores']['baseline']['blind_label'], result['scores']['rrf_1']['blind_label'])


if __name__ == '__main__':
    unittest.main()
