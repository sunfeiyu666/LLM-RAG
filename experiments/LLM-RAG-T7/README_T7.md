# T7 复现说明

## 固定对象

用户指定本地T3的BM25（rrf_0，α=0），基础仓库commit为`53c4c7bc43da35af3034d839eabe02340c038738`。源项目为sunfeiyu666/LLM-RAG。23篇文档、202片段；650字符切分、120重叠；BM25中文单字/双字与完整数字英文分词，k1=1.5、b=0.75。最终Context为top-5，候选池20。20道问题，L1至L4各5题。

这是对已保存真实回答及其实际Context的离线评估，不重新生成、不修饰原回答。q016、q018、q019、q020触及768 token生成上限。引用的课程政策属于固定课程资料快照，不当作当前学生正式资格审核。

## 最短复现：不需要启动模型

当前机器已部署环境，在PowerShell先进入本目录，然后运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\reproduce_t7.ps1
```

脚本优先使用相邻`LLM-RAG-local\.venv\Scripts\python.exe`，也可以通过`-PythonExe`指定已安装依赖的Python。它复算固定评分记录的汇总、四张图及报告，同时检查原始回答、真实Context、全库BM25排序、事实引用、20题和5案例。不会重新调用模型，也不会改原始回答。

其他机器可用Python3.12安装`requirements-t7.txt`，再执行：

```powershell
python -m pip install -r requirements-t7.txt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\reproduce_t7.ps1 -PythonExe python
```

`requirements.txt`与`requirements-lock.txt`是T3原环境留档；T7仅需numpy和matplotlib。图表默认查找Windows微软雅黑，其他系统没有中文字体时需另配字体。复核表是已写明的判断，不是再次执行LLM；数据或回答变更后必须重新核验，不能照搬本表。

## 重新执行本地模型评估

已部署Ollama地址为`http://127.0.0.1:11434`，模型`qwen3:4b-instruct`，固定digest见run_manifest.json。模型权重、运行时和虚拟环境未打包。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ..\LLM-RAG-local\tools\start_local.ps1 -Mode Check
& '..\LLM-RAG-local\.venv\Scripts\python.exe' -X utf8 src\run_t7.py --stage all
& '..\LLM-RAG-local\.venv\Scripts\python.exe' -X utf8 tools\repair_evidence_links.py
& '..\LLM-RAG-local\.venv\Scripts\python.exe' -X utf8 tools\judge_correctness.py
```

完整记录存在时上述脚本会断点续跑或复用缓存。要做独立新实验，应另复制目录，保留原结果，再归档新目录的旧评分记录和cache后运行；不要把缓存复用称为新模型推理实验。输入校验不符会停止，必须重新建立对应版本的评测，不可绕过校验。

新忠实度调用默认上下文12288、温度0、seed42，输出较长截断时有16384上下文的重试。被测答案原配置8192/768保持不变。模型原始提取存在把标题扩写成事实、错放原句和把题设当幻觉等问题，故最终还做了全部20题逐句助手复核。重新跑模型不等于自动完成复核。

## 三层评分，避免混用

- `data/t3_judgments.jsonl`：T3旧盲评，BM25均分4.35；仅审计对照。
- `results/t7/correctness_new.jsonl`：T7新20次单题模型评分，均分3.85；仍发现不合理扣分，完整保留。
- `results/t7/correctness_reviewed.json`：助手逐题5分量表，最终均分4.30。每项含满分、实得分和原因。量表在查看输出后制定，不是预注册量表，也不是独立人类评分。

忠实度原始模型提取222项，初判95项未支持；17项漏证据链接重定位后82项未支持。全部回答逐句复核后保留169个事实，其中26项未完整支持。最终微平均为26/169=15.38%。这一变化来自评估纠错与拆分分母变化，不是系统回答改善。

最终事实表逐项可读，构建器`tools/build_reviewed_facts.py`公开每个原句编号、声明、评分、证据句和原因；只引用实际召回Context，不引用gold来证明忠实度。所有实际回答单元都有“事实”或“排除”记录。纯建议、题设、标题、引用标号不计事实；建议中夹带的政策断言仍单独计入。混合事实拆分有主观性，提交者应复核。

## 指标和文件对应

|内容|文件|
|---|---|
|问题和金标|data/eval/qa_set.jsonl|
|完整202片段与元数据|data/frozen_chunks.json|
|20条实际BM25回答与Context|data/system_answers.jsonl|
|输入模型绑定摘要|results/t7/run_manifest.json|
|检索、原计时、原token记录|results/t7/per_question_metrics.jsonl|
|最终20题和全部k值|results/t7/per_question.json|
|原始忠实度模型评分|results/t7/faithfulness.jsonl|
|漏填证据链接复核|results/t7/evidence_link_repairs.jsonl|
|最终忠实度及每句排除理由|results/t7/faithfulness_reviewed.jsonl|
|最终扁平事实和证据|results/t7/claim_evaluations.json|
|最终正确性子项量表|results/t7/correctness_reviewed.json|
|最差五题排序、归因|results/t7/worst_five.json、failure_reviews.json|
|真实调用请求和返回|results/t7/cache/（仅打包本次正式调用关联缓存）|
|汇总、图、校验|results/t7/summary.json、figures/、verification.json|

作业Recall@k采用“任一gold命中”的二值口径。标准全部gold覆盖另报。MRR/MAP给出全排名和@10两种截断。Precision只把已标注gold编号当相关，重叠等价片段可能被低估。

忠实度1–5：5直接支持、4可推导、3部分支持、2无依据、1矛盾。每题事实均分后再按题平均；幻觉率按作业严格定义，将部分支持也算未支持，以全体事实数作分母。没有事实则N/A。最终评分与失败案例都属于助手辅助分析，保留可复核证据。

本次没有新增独立端到端延迟或多模态性能实验；旧缓存计时不能充作新速度基准。实验结果不能外推到其他版本或生产系统。
