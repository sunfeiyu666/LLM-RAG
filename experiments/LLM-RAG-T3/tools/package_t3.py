"""Validate all finished T3 outputs and package source/data/results, no model weights."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from t3_core import read_jsonl, ENDPOINT
from run_t3 import ANSWER_METHODS


def main():
    out = ROOT/'results/t3'
    qs = read_jsonl(ROOT/'data/eval/qa_set.jsonl')
    ids = {q['id'] for q in qs}
    answers_list = read_jsonl(out/'answers.jsonl')
    answers = {(a['id'], a['method']): a for a in answers_list}
    judgments_list = read_jsonl(out/'judgments.jsonl')
    judgments = {j['id']: j for j in judgments_list}
    ret = {r['id']: r for r in read_jsonl(out/'retrieval.jsonl')}
    assert len(qs) == 20 and len(ids) == 20
    assert len(answers_list) == 180 and len(answers) == 180
    assert set(answers) == {(qid, m) for qid in ids for m in ANSWER_METHODS}
    assert len(judgments_list) == 20 and set(judgments) == ids
    assert set(ret) == ids
    for qid in ids:
        assert set(judgments[qid]['scores']) == set(ANSWER_METHODS)
        assert answers[(qid,'baseline')]['answer'] == answers[(qid,'rrf_1')]['answer']
        assert judgments[qid]['scores']['baseline']['score'] == judgments[qid]['scores']['rrf_1']['score']
        assert [d['chunk_id'] for d in ret[qid]['methods']['baseline']['top']] == [d['chunk_id'] for d in ret[qid]['methods']['rrf_1']['top']]
        for m in ANSWER_METHODS:
            a = answers[(qid,m)]
            assert a['answer'].strip() and len(a['top_chunk_ids']) <= 5
            assert len(a['top_chunk_ids']) == len(set(a['top_chunk_ids']))
            assert 0 <= judgments[qid]['scores'][m]['score'] <= 5
    manifest = json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'data/eval/qa_set.jsonl').read_bytes()).hexdigest() == manifest['binding']['dataset_sha256']
    for f in ['T3实验报告.md','T3案例分析.md','README_T3.md','results/t3/summary.json']:
        assert (ROOT/f).is_file(), f
    assert len(list((out/'figures').glob('*.png'))) == 4
    assert len(read_jsonl(out/'challenge_answers.jsonl')) == 20
    assert len(read_jsonl(out/'challenge_judgments.jsonl')) == 5
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'packages': {p: importlib.metadata.version(p) for p in ['langchain-core','langchain-community','faiss-cpu','numpy','scikit-learn','matplotlib']}}
    try:
        with urlopen(ENDPOINT+'/api/version', timeout=10) as r:
            environment['ollama'] = json.load(r)
        with urlopen(ENDPOINT+'/api/ps', timeout=10) as r:
            environment['model_runtime_status'] = json.load(r)
    except Exception as exc:
        environment['runtime_status_unavailable'] = str(exc)
    (out/'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    verification = {'questions': 20, 'answer_conditions': 9, 'logical_answers': 180, 'blinded_judgments': 20,
                    'diagnostic_questions': 5, 'diagnostic_answers': 20, 'figures': 4,
                    'dense_endpoint_equals_baseline': True, 'identical_endpoint_answers_have_equal_scores': True,
                    'dataset_binding_matches': True, 'unit_tests_passed': 12,
                    'note': 'Structural checks passed; model-based scoring is not independent human validation.'}
    (out/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    excluded_dirs = {'__pycache__','.venv','.runtime','index_langchain','.git','.mplconfig'}
    excluded_names = {'pilot.log','pilot_original_plan.jsonl','pilot_manifest.json','challenge_judgments_pilot.jsonl',
                      'local_deployment_validation.json'}
    deliverable = ROOT.parent/'T3_代码实验与报告.zip'
    with zipfile.ZipFile(deliverable, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or any(part in excluded_dirs for part in p.relative_to(ROOT).parts) or p.name in excluded_names:
                continue
            archive.write(p, str(Path(ROOT.name)/p.relative_to(ROOT)).replace('\\','/'))
    with zipfile.ZipFile(deliverable) as archive:
        assert archive.testzip() is None
    print(json.dumps({'zip': str(deliverable), 'size_bytes': deliverable.stat().st_size, 'validation': verification}, ensure_ascii=False))


if __name__ == '__main__':
    main()
