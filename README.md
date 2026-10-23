#RAG 项目

本项目保留一条基础LangChain链路：**加载 → 切分 → 向量化 → 建索引 → 检索top-k → 拼Context → LLM生成**。模型是本地Ollama的 `qwen3:4b-instruct`，23篇知识库Markdown保持不变，回答支持逐字流式显示及引用来源。

## 1 创建 Conda 环境

Windows安装Anaconda

```cmd
cd /d D:\Desktop\学习资料\LLM-RAG
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
RAG项目_Conda复现版/
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
├── data/eval/README.md           正式评测集目录说明
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
