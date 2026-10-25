"""Validate actual T1 inputs/results while separately reporting non-input documentation drift."""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]

def main():
    output = ROOT / "results"
    metadata = json.loads((output / "t1_run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/eval/qa_manifest.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "data/eval/qa_set.jsonl") == metadata["binding"]["qa_sha256"]
    assert sha(ROOT / "src/rag_chain.py") == metadata["binding"]["rag_chain_sha256"]
    assert sha(ROOT / "src/run_t1_experiment.py") == metadata["binding"]["runner_sha256"]
    assert sha(ROOT / "index_langchain/index_info.json") == metadata["binding"]["index_info_sha256"]
    for relative, digest in manifest["bound_file_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    for name in ["qa_set.jsonl", "qa_evidence.jsonl"]:
        assert sha(ROOT / "data/eval" / name) == manifest["artifact_sha256"][name]
    dataset = read(ROOT / "data/eval/qa_set.jsonl")
    assert len(dataset) == 20
    assert Counter(q['level'] for q in dataset) == {f"L{i}": 5 for i in range(1, 5)}
    for name, arm in [("t1_baseline.jsonl", "rag"), ("t1_no_retrieval.jsonl", "bare")]:
        rows = read(output / name)
        assert len(rows) == len(dataset) and len({r['id'] for r in rows}) == len(dataset)
        for q, r in zip(dataset, rows):
            assert r['id'] == q['id'] and r['question'] == q['question'] and r['gold_answer'] == q['gold_answer']
            assert r['condition'] == arm and r['status'] == 'ok' and r['answer'].strip()
            assert len(r['llm_calls']) == 1
            assert r['llm_calls'][0]['response_metadata']['model'] == metadata['binding']['model_parameters']['model']
            assert len(r['retrieved']) == (5 if arm == 'rag' else 0)
            if arm == 'bare':
                assert r['context'] == ''
    hashes = json.loads((output / "t1_artifact_sha256.json").read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        assert sha(output / name) == digest, name
    drift = [name for name, digest in manifest['artifact_sha256'].items()
             if sha(ROOT / 'data/eval' / name) != digest]
    report = {"status": "passed_for_experimental_inputs_and_outputs", "paired_calls": 40,
              "question_count": 20, "raw_responses_match_questions": True,
              "dataset_evidence_corpus_code_unchanged": True,
              "artifact_hashes_verified": len(hashes),
              "documentation_drift": drift,
              "note": "可读说明发生编号/顺序调整；保留其改动，不修改原金标与manifest。本次题号始终以qa_set.jsonl为准。D3全量校验仍会报告该说明哈希变化；本校验单独验证实际实验输入和输出，不能解释为所有文档未变。" if drift else "No documentation drift"}
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
