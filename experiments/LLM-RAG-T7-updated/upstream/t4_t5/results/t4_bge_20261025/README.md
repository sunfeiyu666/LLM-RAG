# T4：本地 BGE 中文 Embedding 对比实验

运行日期：2026-10-25。使用 `data/eval/qa_set.jsonl` 的 20 道题和同一批 202 个文本块。稠密检索使用 `BAAI/bge-small-zh-v1.5` 与 512 维 FAISS 索引；模型版本和实验设置见 `run_metadata.json`。仓库保存代码与实验结果，模型权重和索引需在复现时准备。

## 指标

| 检索方法 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.80 | 1.00 | 1.00 | 1.00 | 0.875 |
| BGE + FAISS | 0.45 | 0.80 | 0.95 | 1.00 | 0.638 |
| BM25 + BGE（RRF） | 0.60 | 0.95 | 1.00 | 1.00 | 0.754 |

按作业口径，Recall@k 表示前 k 块中至少出现一个标注证据块的题目比例。BGE 在这组题的首次证据命中名次上，只比 BM25 更好 1 题（q016）；BM25 更好 8 题，其余 11 题持平。top-5 完整覆盖所有标注证据块的题数依次为 BM25 12/20、BGE 10/20、混合 13/20。单题样例不能外推为模型的总体优劣。

正式 BGE 对照的 6 条稠密占优、5 条 BM25 占优案例见 [`t4_bge_cases.md`](t4_bge_cases.md)。其中 BGE 占优的 5 条是额外构造的诊断题，另 1 条是正式题 q016；BM25 占优的 5 条均为正式题。五条构造题均由 BGE 在 top-3 命中标注证据，BM25 top-3 则未命中。诊断题不参加上表指标计算，全部 45 条候选及其排序见 `t4_bge_candidate_probe.json`，避免只报告挑出的优势案例。

按 Recall@1、Recall@3 和 MRR@10 综合判断，BM25 是本次正式评测最优的 T4 检索器，T5 应接在 BM25 的 top-5 候选池上。混合 RRF 的主指标未超过 BM25，尽管其 top-5 完整证据覆盖率从 12/20 提高到 13/20。

## 复现与文件

安装 `requirements-t4-t5.txt` 后，在仓库根目录准备 BGE 模型并建立索引，再运行评测。模型权重和索引不纳入 Git；严格复现应使用 `run_metadata.json` 中记录的模型版本。

```powershell
python -B src/neural_retrieval.py --download --build
python -B src/t4_evaluate.py --qa data/eval/qa_set.jsonl --out results/t4_reproduce --dense-backend bge
```

`t4_rankings.jsonl` 保存逐题 top-10；`t4_metrics.csv` 保存作业指标；`t4_diagnostics.json` 保存完整证据覆盖与胜负题号；`run_metadata.json` 保存数据与模型版本；`t4_bge_metrics.png` 是报告用 PNG 图；`t4_bge_cases.md` 保存十一条案例。复现命令使用新目录，不会覆盖正式实验记录。

## 评测集限制

本实验固定使用正式 20 题；五条额外诊断题仅用于展示稠密检索占优的具体情形，不纳入指标。样本规模较小，课程评测说明也未给出独立人工复核声明。
