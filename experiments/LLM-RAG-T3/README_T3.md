# T3 查询分解与融合实验

本目录是在课程小组LLM-RAG基础项目上新增的T3实现。起点仓库为 https://github.com/sunfeiyu666/LLM-RAG ，使用已核对的提交53c4c7bc43da35af3034d839eabe02340c038738。这里完成T3，不代替T7完整系统评估。

## 完成范围

- 用本地大模型做查询改写；生成3–5条互补多查询；将复合问题分解为不同子问题。
- 多路检索并行执行，用RRF去重融合，固定返回top-5；子查询先形成短答，再综合。
- 对向量/BM25的加权RRF扫描α=0、0.3、0.5、0.7、1，另做归一化分数加权的检索检查，明确区分两种方法。
- 正式20题上比较Recall@5和答案质量，保存所有查询、排名、答案、盲评、汇总和PNG曲线。
- 口语化诊断案例单独保存，不混入正式20题的统计。案例报告展示真实查询、中间短答、召回片段和回答。

## 本机最快使用方式

本目录与此前部署的`LLM-RAG-local`并列存放，可直接复用其Python环境和模型服务。打开PowerShell，进入本目录后执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "..\LLM-RAG-local\tools\start_local.ps1" -Mode Check
& "..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 src\run_t3.py --stage all
& "..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 tools\run_t3_challenges.py
& "..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 tools\build_t3_cases.py
& "..\LLM-RAG-local\.venv\Scripts\python.exe" -X utf8 tools\summarize_t3.py
```

结果存在时会继续断点，不重复已完成的条件；完整请求相同的生成输出会复用缓存。只重新生成图表和报告时，运行最后一条即可，不会再调用模型。

若要从零开展一轮独立复跑，先将整个`results/t3`目录改名保留，再运行以上命令；不能将不同数据、模型或参数的结果混在同一目录。原仓库`data/eval/qa_manifest.json`和部分metadata保留为历史来源记录，本次T3的实际数据绑定以`results/t3/run_manifest.json`为准。

不要在正在运行时同时启动第二份相同的主实验；需要退出时中断进程，之后重新运行即可。

## 在其他电脑或组内仓库复现

使用Python 3.12，进入项目根目录执行：

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-t3.txt
ollama pull qwen3:4b-instruct
.venv\Scripts\python.exe src\index.py
.venv\Scripts\python.exe src\run_t3.py --stage all
.venv\Scripts\python.exe tools\run_t3_challenges.py
.venv\Scripts\python.exe tools\build_t3_cases.py
.venv\Scripts\python.exe tools\summarize_t3.py
```

另开终端启动`ollama serve`，或使用已运行的Ollama。服务默认地址是127.0.0.1:11434。模型清单摘要应与run_manifest.json一致。

本任务固定202块、650字符、120重叠；如果其他组员改变分块或语料，请重新映射gold片段，另开结果目录，不要混用旧结果。模型、参数、数据和片段内容绑定不一致时，程序会停止。

## 主要文件

| 文件 | 用途 |
| --- | --- |
| src/t3_core.py | 本地LLM调用、查询规划、BM25、RRF及分数融合 |
| src/run_t3.py | 20题主实验，支持plan/retrieve/generate/judge分阶段与断点 |
| src/test_t3.py | 加权端点、去重、指标及年份等单元测试 |
| tools/run_t3_challenges.py | 第一批4道预先写定的口语化诊断题 |
| tools/build_t3_cases.py | 追加探索题、诊断题回答、评分及案例报告 |
| tools/summarize_t3.py | 完整性核对、汇总、绘图、实验报告 |
| results/t3/query_plans.jsonl | 模型真实生成的中间查询 |
| results/t3/retrieval.jsonl | 所有分路排名、融合结果及逐题检索指标 |
| results/t3/answers.jsonl | 9个条件×20题，共180条逻辑回答 |
| results/t3/judgments.jsonl | 隐藏策略名、打乱顺序的答案正确性模型评分 |
| results/t3/summary.json | 策略、权重和分层汇总 |
| results/t3/figures/ | 策略对比和融合权重PNG图 |
| T3实验报告.md | 方法、指标、结果、曲线及局限 |
| T3案例分析.md | 查询失败与优化后的完整诊断案例 |

## 与同学合并

优先合并新增的t3_core.py、run_t3.py、test_t3.py、相关tools脚本、本README和requirements-t3.txt，以及独立的results/t3目录。不要覆盖组内T1、T2、T4、T5文件，不要上传`.venv`、Ollama模型或本机索引缓存。

当前向量路沿用原项目TF-IDF/LSA，不是神经嵌入模型。若T4负责人更换成BGE等模型，需要对接接口、重新冻结配置并复跑，不能把现有曲线直接标成神经稠密检索结果。

## 评分和诚信说明

Recall@5按本作业FAQ的“任一gold片段命中”计算，另报gold覆盖。正确性为同一Qwen的0–5分盲评，完全相同的答案只评一次并共享分数，仍存在自评偏差；字符ROUGE-L仅作辅助。部分长回答可能触及768 token的统一预算，原文和截断标志均保留。这是固定输出预算下的比较，更高预算可能改变复杂问题的评分。

α在同一20题上探索，没有独立留出集，不声称显著性或唯一最优。口语化案例是诊断示例，含事后挑选展示和一题追加探索，不能冒充独立泛化测试。

基础项目、方法文献、Ollama及大模型辅助编程来源已在报告列出。所有模型输出来自本地实际调用；提交者应自行核验评测集、图表及分析，并按作业要求注明辅助来源。报告不会把助手核验冒称为独立人工审核。
