# T1真实运行结果

2026-10-25对固定D3评测集20题分别运行基础RAG与无检索Qwen，共40次独立调用。

- [T1_基线与对照实验分析.md](T1_基线与对照实验分析.md)：实验方法、统计、至少2个成功案例、至少2个完整失败案例及额外生成缺陷案例。
- [t1_baseline.jsonl](t1_baseline.jsonl)：20条RAG原始输出，含实际召回、Context、耗时与服务端元数据。
- [t1_no_retrieval.jsonl](t1_no_retrieval.jsonl)：20条同模型无检索原始输出。
- [t1_逐题答案对照.md](t1_逐题答案对照.md)：全20题的参考答案、两组原始答案与逐题评语。
- [t1_comparison.csv](t1_comparison.csv)、[t1_summary.json](t1_summary.json)：逐题及汇总数据。
- [t1_answer_reviews.jsonl](t1_answer_reviews.jsonl)：助手评分和评分依据原句，可供人工复核，不是独立人工评测。
- [t1_案例打印.log](t1_案例打印.log)：实际打印的成功、失败和生成缺陷案例及分析。
- [t1_run_metadata.json](t1_run_metadata.json)、[t1_artifact_sha256.json](t1_artifact_sha256.json)：实验设置、绑定关系、完成状态及结果校验值。

本次评分不是EM或自动准确率，也没有把未回答一概当作幻觉。没有对答案做人工润色后冒充模型输出。原始输出保留；T1只涉及基础版与同模型的无检索对照，不含增强版或T2—T5实验。

## T4 检索策略对比（朱相多）

BM25、BGE + FAISS 与 RRF 混合检索的指标、图、逐题排序和十条查询分析见 [T4 结果](t4_bge_20261025/README.md)。对应代码为 `src/retrieval_compare.py`、`src/neural_retrieval.py` 和 `src/t4_evaluate.py`。

## T5 重排与压缩（朱相多）

基于 T4 最优的 BM25 top-5，对比直接截取、规则重排、Cross-Encoder 重排和压缩前后结果，见 [T5 结果](t5_final_20261026/README.md)。对应代码为 `src/rerank.py` 和 `src/t5_evaluate.py`。两项实验可用 `python results/verify_t4_t5.py` 核对。

