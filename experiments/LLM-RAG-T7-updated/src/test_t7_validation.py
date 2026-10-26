"""Metric boundary cases and corruption checks; no generated-answer/model calls."""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from t7_updated_core import aggregate, rank_metrics, read_jsonl, rouge_l

spec = importlib.util.spec_from_file_location('t7_audit', ROOT / 'tools/verify_t7_updated.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class MetricBoundaries(unittest.TestCase):
    def test_missed_gold_stays_in_ap_denominator(self):
        result = rank_metrics(['x', 'a', 'y', 'b'], ['a', 'b', 'missed'])
        self.assertAlmostEqual(result['mrr_full'], .5)
        self.assertAlmostEqual(result['map_full'], (1/2 + 2/4)/3)
        self.assertEqual(result['recall_at_3'], 1)
        self.assertAlmostEqual(result['gold_coverage_at_3'], 1/3)
        self.assertAlmostEqual(result['precision_at_3'], 1/3)
        self.assertEqual(result, audit.reference_metrics(['x','a','y','b'], ['a','b','missed']))

    def test_mrr_cutoff_excludes_rank_11(self):
        result = rank_metrics([f'x{i}' for i in range(10)] + ['gold'], ['gold'])
        self.assertEqual(result['mrr_at_10'], 0)
        self.assertEqual(result['map_at_10'], 0)
        self.assertAlmostEqual(result['mrr_full'], 1/11)

    def test_invalid_gold_and_duplicate_ranking(self):
        for ranking, gold in [(['a'], []), (['a','a'], ['a'])]:
            with self.assertRaises(ValueError):
                rank_metrics(ranking, gold)

    def test_lcs_reference_handles_repetition_and_empty(self):
        for a,b,expected in [('abc','ac',.8), ('aaa','aa',.8), ('a','b',0), ('','a',0), ('学分161.5','学分1615',1)]:
            self.assertAlmostEqual(rouge_l(a,b), expected)
            self.assertAlmostEqual(audit.reference_rouge(a,b),expected)


class EvidenceCorruptionChecks(unittest.TestCase):
    def setUp(self):
        self.packet = read_jsonl(ROOT/'results/t7/faithfulness_packets.jsonl')[0]
        self.review = read_jsonl(ROOT/'results/t7/faithfulness_reviewed.jsonl')[0]

    def test_original_review_passes(self):
        audit.check_review(self.review, self.packet)

    def test_cannot_use_outside_context_document(self):
        self.review['claims'][0]['evidence'][0]['chunk_id'] = 'valid_elsewhere_but_not_actual_top3'
        with self.assertRaisesRegex(ValueError, 'outside actual Context'):
            audit.check_review(self.review,self.packet)

    def test_quote_must_be_literal(self):
        self.review['claims'][0]['evidence'][0]['quote'] = '这是没有出现在资料中的伪造原句。'
        with self.assertRaisesRegex(ValueError, 'not literal'):
            audit.check_review(self.review,self.packet)

    def test_supported_partial_and_wrong_block_rejected(self):
        original = copy.deepcopy(self.review)
        self.review['claims'][0]['score'] = 3
        with self.assertRaisesRegex(ValueError, 'Supported flag'):
            audit.check_review(self.review,self.packet)
        self.review = original
        self.review['claims'][0]['evidence'][0]['block'] = 99
        with self.assertRaisesRegex(ValueError, 'block number'):
            audit.check_review(self.review,self.packet)

    def test_changed_packet_and_omitted_unit_rejected(self):
        self.packet['answer'] += ' 篡改'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.check_review(self.review,self.packet)
        self.packet = read_jsonl(ROOT/'results/t7/faithfulness_packets.jsonl')[0]
        self.review['claims'] = []
        with self.assertRaisesRegex(ValueError, 'Unclassified'):
            audit.check_review(self.review,self.packet)

    def test_no_fact_answer_is_not_given_full_score(self):
        self.review.update(claims=[],fact_count=0,unsupported_fact_count=0,faithfulness_1_5=None,hallucination_rate=None,
            excluded_units=[{**u,'reason':'Synthetic no-fact fixture'} for u in self.packet['answer_units']])
        audit.check_review(self.review,self.packet)
        result=aggregate([self.review])
        self.assertIsNone(result['faithfulness_1_5'])
        self.assertIsNone(result['hallucination_micro'])
        self.assertEqual(result['no_fact_answers'],1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
