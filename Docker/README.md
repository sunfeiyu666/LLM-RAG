# T8-4 工程化：Docker + 一键启动 + Streamlit

## 快速使用（Windows）

前提：Docker Desktop 已启动并使用 Linux 容器；Windows 上的 Ollama 已启动，且 `ollama list` 中存在 `qwen3:4b-instruct`。本机已有该模型，不需要重新下载。

1. 双击本文件夹中的 `start.cmd`。
2. 等待首次镜像构建和知识库索引建立，脚本会自动打开 http://localhost:8501。
3. 在网页输入问题，回答会逐步输出；回答下方展示引用来源，展开检索详情可查看实际候选片段。
4. 双击 `stop.cmd` 停止。停止操作保留索引，后续可直接再次启动。

Docker 内自带 Python 3.12 和依赖；此启动方式无需激活 Conda。原来的 `conda activate rag-langchain`、`python src\chat_ollama.py` 命令行流程仍可使用。

## 命令行 / WSL 使用

Windows PowerShell（位于项目根目录）：

```powershell
docker compose -f Docker/compose.yaml up -d --build --wait --wait-timeout 240
docker compose -f Docker/compose.yaml ps
docker compose -f Docker/compose.yaml logs --tail 60
docker compose -f Docker/compose.yaml down
```

WSL / Linux shell（你的 Docker 截图来自 Linux shell）：

```bash
cd '/mnt/d/Desktop/学习资料/LLM-RAG/LLM-RAG/Docker'
sh start.sh
# 浏览器打开 http://localhost:8501
docker compose down
```

## 架构与保留的实验设置

浏览器 → Docker 内 Streamlit → 原 `src/rag_chain.py` 的 LangChain LCEL → FAISS top-k → Context → Windows Ollama → 原 Qwen 模型。

- 知识库：原有 23 篇 Markdown，切分长度 650、重叠 120，共 202 个片段。
- 向量化：原字符 TF-IDF + LSA 64 维；向量索引为 FAISS。默认 top-k=5，网页可调整。
- 原提示词、temperature=0.2、模型 `qwen3:4b-instruct` 均保持原设置。没有加入重排或额外路由。
- `runtime.py` 仅在网页进程中配置模型地址及索引目录，不修改原 `src` 文件。
- 历史问答保存在当前浏览器会话中，仅用于展示，不传入模型；仍是单轮基础 RAG。刷新/重连可能丢失会话记录。
- 网页流式显示模型返回的文本块，不是人为延时打字；模型加载、检索和首字生成仍需等待。
- 仅列出回答实际标注且编号有效的引用；候选片段另行展开。无效编号会告警。编号合法不等于内容正确，仍需核对证据。
- 部分生成后中断会显示“未完成”，保留部分文字，不能当作完整答案。
- 同一服务一次处理一个生成请求，避免多个浏览器同时占用本地模型。
- 原 T1 评测集及 `results` 文件不改写。

## 文件职责

| 文件 | 作用 |
|---|---|
| Dockerfile / Dockerfile.dockerignore | Python 镜像构建、限制构建上下文 |
| compose.yaml | 网页端口、宿主机 Ollama、索引持久化卷 |
| requirements-ui.txt | 引用原依赖，额外固定 Streamlit 版本 |
| start.cmd / start.ps1 | Windows 检查 Docker、构建、启动、等待健康状态并打开浏览器 |
| stop.cmd / start.sh | Windows 停止 / WSL 启动 |
| preflight.py | 容器内检查 Ollama 和模型是否存在 |
| entrypoint.py | 启动前检查、建立/加载索引，然后启动网页 |
| runtime.py / web_app.py | 原 RAG 适配和 Streamlit 界面 |
| smoke_check.py | 真正调用现有模型的流式问答自检 |
| .env.example | 可选的配置模板；默认不需要复制 |

## 配置与数据持久化

如需改端口或地址，将 `.env.example` 复制为本文件夹中的 `.env` 后修改：

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:4b-instruct
WEB_PORT=8501
PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

容器内的 `localhost` 指向容器本身，因此使用 `host.docker.internal` 访问 Windows Ollama。本机部署时已实际验证容器访问成功，不需要修改 Ollama 监听配置。

依赖默认从清华 PyPI 镜像下载，并使用 BuildKit 下载缓存加速重建。需要官方源时，在 `.env` 设置 `PYPI_INDEX_URL=https://pypi.org/simple` 后重新构建。

索引保存在 Compose 命名卷 `ynu-rag-t8_rag-index`，首次按同一语料在 Linux 环境建立；后续按原程序指纹逻辑复用或自动重建。Windows 的 `index_langchain` 保持原样，模型权重也继续由宿主机 Ollama 管理。镜像仅打包运行所需的代码、语料和来源清单，不打包虚拟环境、原始大文件、旧评测结果或模型权重。

`docker compose down` 保留索引。不要随意添加 `-v`，该选项会删除本部署的索引卷。更新语料或代码后重新运行 `start.cmd` 即可重建镜像。

## 自检与演示

在项目根目录运行：

```powershell
# 确认容器健康
docker compose -f Docker/compose.yaml ps
# 独立执行一次真实流式问答；不会改写 T1 结果
docker compose -f Docker/compose.yaml exec -T rag-web python /app/Docker/smoke_check.py
# 运行已有的 7 个核心测试
docker compose -f Docker/compose.yaml exec -T -w /app/src rag-web python -m unittest test_kb test_langchain test_streaming -v
```

推荐网页测试问题：

1. 云南大学数据库概论的先修课是什么？（检查“数据结构”及原文引用。）
2. 软件工程普通班与卓越班的个性课程平台学分分别是多少，相差多少？（检查版本和跨文档比较。）
3. 我想申请明年的推免，请直接告诉我一定能不能通过。（检查资料不足提示及避免保证性结论。）

这些问题不保证每次模型回答正确；工程化不改变基础 RAG 的召回和推理局限。展示时应核对检索内容和引用。

录屏可按“启动与容器健康 → 网页问答及流式输出 → 展开证据与引用 → 重启复用索引”进行，若作业要求提交演示视频，请自行录制实际操作。

## 常见问题

- **Docker 引擎不可用**：启动 Docker Desktop，确认切换到 Linux containers；WSL 中还需要打开 Docker Desktop 对该发行版的集成。
- **首次构建很慢**：首次需下载基础镜像和 Python 包。保留终端观察进度，不要反复关闭重建。
- **Ollama 连接失败**：先确认 Windows Ollama 在运行，`http://localhost:11434/api/tags` 能访问。查看容器日志。换电脑后若容器无法访问宿主机，参照 Ollama 官方 FAQ 设置 `OLLAMA_HOST`（例如 `0.0.0.0:11434`）后重启 Ollama；按需限制防火墙范围。
- **模型不存在**：使用 `ollama list` 核对完整标签；不要假定本机已有的标签一定能从公共仓库重新 pull。迁移电脑需要同时准备同一个模型。
- **端口占用**：在 `.env` 修改 `WEB_PORT` 后重新运行启动脚本。
- **网页已启动但后续问答失败**：健康检查验证网页服务存活，无法保证之后 Ollama 一直可用；可点击侧栏“检查模型连接”。
- **等待首字较久**：与模型冷启动、上下文长度和硬件有关；流式显示不等于推理加速。
- **需要完全重建索引**：先停止服务，再只删除本部署的命名卷（会丢弃容器索引缓存），下次启动自动重建。原始知识库和 Windows 索引不受影响。

官方资料：
- https://docs.docker.com/desktop/features/networking/networking-how-tos/
- https://docs.ollama.com/faq
- https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream
