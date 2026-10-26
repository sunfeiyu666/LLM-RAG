#RAG 项目

本项目保留一条基础LangChain链路：**加载 → 切分 → 向量化 → 建索引 → 检索top-k → 拼Context → LLM生成**。模型是本地Ollama的 `qwen3:4b-instruct`，23篇知识库Markdown保持不变，回答支持逐字流式显示及引用来源。

## 1 创建 Conda 环境

Windows安装Anaconda

```cmd
conda deactivate
conda env create -f environment.yml
conda activate rag-langchain
python --version
python -m pip check
python tools\verify_environment.py
```

配置使用Python **3.12**，通过pip安装 `requirements.txt`。原 `LLM-homework` 环境的Python 3.10.22不满足当前NumPy固定版本要求，建议使用新的独立环境。若已有Python 3.12环境，激活后执行 `python -m pip install -r requirements.txt` 即可。详细排错见 [Conda复现说明.md](Conda复现说明.md)。

## 2 确认 Ollama 与模型

打开Ollama桌面程序，或在另一个终端运行 `ollama serve` 并保持窗口开启。然后执行：

```cmd
ollama list
```

需要存在原模型标签 `qwen3:4b-instruct`。压缩包不包含模型权重；新电脑应按原来源准备或导入同名模型。自定义标签不保证能在公共模型库直接下载。观察到的原模型摘要见 `metadata/package_sources.json`。

## 3 重建索引并运行

```cmd
python src\index.py
python src\chat_ollama.py
```

输入 `exit`、`quit` 或 `/bye` 退出。

## T2 分块策略实验

T2 独立运行 4 种分块方法与 3 种 `chunk_size` 的 12 组对比，不会覆盖基础索引：

```cmd
python src\experiment_t2.py
```

正式评测集应保存为 `data/eval/qa_set.jsonl`。在该文件尚未提供时，脚本会使用现有 20 条开发定位题检查流程；这部分结果不能代替最终实验结果。Ollama 可用时执行 `python src\experiment_t2.py --generate`，同时生成答案并计算答案正确性。完整说明见 [T2分块策略实验说明.md](docs/T2分块策略实验说明.md)。


## 4 目录与实现

```text
LLM-RAG/
├── README.md
├── Conda复现说明.md
├── environment.yml
├── requirements.txt             直接依赖固定版本
├── requirements-lock.txt        已验证环境的完整pip快照
├── 启动问答.cmd
├── 重建索引.cmd
├── 文档目录.md
├── data/corpus/                 23篇实际读取的Markdown
├── data/originals/              可用原始HTML、PDF、OCR文字
├── data/eval/                    D3评测集、证据及校验脚本
├── metadata/                    来源、语料校验值和打包说明
├── src/
│   ├── index.py                 索引构建入口
│   ├── kb.py                    加载、切分、向量化、FAISS
│   ├── rag_core.py              文本块结构与原向量化预处理
│   ├── retrieve.py              向量top-k检索入口
│   ├── search.py                检索查看入口
│   ├── rag_chain.py             LCEL拼Context和生成链
│   ├── generate.py              生成链导出接口
│   ├── chat_ollama.py           问答交互及流式输出
│   ├── evaluate.py              开发定位检查入口
│   ├── check_retrieval.py       开发定位题和检查
│   └── test_*.py                基础链路程序检查
├── tools/verify_environment.py   只读环境检查
├── docs/资料与实现说明.md
└── results/README.md
```


## D3 评测集

20条问答对，L1—L4各5条，文件为 [data/eval/qa_set.jsonl](data/eval/qa_set.jsonl)。逐题证据、评分与复核边界见 [评测集说明](data/eval/README.md)。当前代码中的旧开发定位检查不等于本评测集的完整评估。

```cmd
python data\eval\verify_eval_set.py
```

T4/T5 使用同一份 20 题评测内容，复核脚本会统一不同平台的文本换行符。评测集与清单仍由团队原目录维护。


## T1 基础RAG与无检索对照

已对D3全部20题真实运行两组实验。结果与说明见 [results/README.md](results/README.md)。日常问答仍仅保留基础版。

```cmd
python src\run_t1_experiment.py --output-dir results\t1_rerun
```

两组原始答案、实际检索片段和助手评分分别保存，未改变知识库、向量化器或生成模型。

## T4 检索策略对比（朱相多）

运行 T4/T5 前，除基础 `requirements.txt` 外，还需安装 [`requirements-t4-t5.txt`](requirements-t4-t5.txt)。大型模型权重、向量索引与虚拟环境不纳入 Git。

正式 20 题统一使用 `1024/128` 分块得到的 147 个知识库文本块，对比 BM25、BGE 中文 Embedding + FAISS，以及两路 RRF。Recall@1/3/5/10、MRR@10、PNG 图和 5+5 条排序案例均保存在 [T4 结果](results/t4_1024_128_20261026/README.md)。按正式检索指标，BM25 是本次最优方法。

## T5 重排与压缩（朱相多）

T5 结果保存在独立的 [`results/t5_1024_128_20261026/`](results/t5_1024_128_20261026/README.md)，以 T4 最优的 BM25 top-5 为共同候选池，比较直接 top-3、规则重排 top-3、本地 BGE Cross-Encoder 模型重排 top-3，以及按当前 147 个文本块估计词项自信息量的句子压缩。20 题×6 组的原始答案、Context token、延迟、逐题质量评分和权衡图已核对。T4 与 T5 的代码分别在 `src/retrieval_compare.py`、`src/t4_evaluate.py` 和 `src/rerank.py`、`src/t5_evaluate.py`。
