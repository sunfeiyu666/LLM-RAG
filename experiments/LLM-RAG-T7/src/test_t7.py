import unittest
import json
from t7_core import rank_metrics, answer_units, normalize, judge_faithfulness


class T7Tests(unittest.TestCase):
    def test_metrics_and_total_gold_ap_denominator(self):
        m = rank_metrics(['a','x','b','y'], ['a','b','missing'])
        self.assertEqual(m['recall_at_1'], 1)
        self.assertEqual(m['gold_coverage_at_1'], 1/3)
        self.assertEqual(m['precision_at_3'], 2/3)
        self.assertAlmostEqual(m['map_at_10'], (1+2/3)/3)
        self.assertEqual(m['mrr_at_10'],1)

    def test_mrr_cutoff_is_explicit(self):
        rows = [str(i) for i in range(12)]
        m = rank_metrics(rows,['11'])
        self.assertEqual(m['mrr_at_10'],0)
        self.assertEqual(m['mrr_full'],1/12)

    def test_empty_gold_and_duplicate_ranking_rejected(self):
        with self.assertRaises(ValueError):rank_metrics(['a'],[])
        with self.assertRaises(ValueError):rank_metrics(['a','a'],['a'])

    def test_all_answer_sentences_have_unique_anchors(self):
        units=answer_units('标题\n第一条。第二条！\n建议再查证。')
        self.assertEqual(len(units),4)
        self.assertEqual(len({u['id'] for u in units}),4)
        self.assertEqual(normalize('**GPA 3.0**'),'gpa3.0')

    def test_support_requires_real_quote_and_no_fact_is_not_perfect(self):
        q={'id':'q001','level':'L1','question':'上限是多少？'}
        doc={'page_content':'学分上限10。','metadata':{'chunk_id':'c1','title':'规则','category':'政策','version':'固定','page':None,'content_url':'https://example.test'}}
        class Fake:
            def __init__(self,claims):self.claims=claims
            def call(self,*args):
                return json.dumps({'units':{'s001':{'claims':self.claims,'non_factual_reason':''}}},ensure_ascii=False),{'truncated':False}
        # format_document adds seven metadata/body lines before the final policy sentence.
        claim={'claim':'学分上限10','score':5,'verdict':'supported','evidence_sentence_ids':['b1_s007']}
        supported=judge_faithfulness(Fake([claim]),q,'上限10。',[doc])
        self.assertEqual(supported['supported_fact_count'],1)
        wrong={**claim,'evidence_sentence_ids':[]}
        unsupported=judge_faithfulness(Fake([wrong]),q,'上限10。',[doc])
        self.assertEqual(unsupported['hallucination_rate'],1)
        self.assertEqual(unsupported['faithfulness_1_5'],3)
        empty=judge_faithfulness(Fake([]),q,'资料不足。',[doc])
        self.assertIsNone(empty['faithfulness_1_5'])
        self.assertIsNone(empty['hallucination_rate'])


if __name__ == '__main__':unittest.main()
