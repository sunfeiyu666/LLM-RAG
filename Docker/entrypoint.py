import os
import sys

from preflight import check_ollama
from runtime import kb

try:
    print("Checking Ollama:", check_ollama(), flush=True)
    knowledge_base = kb.load_index()
    print(f"RAG ready: {len(knowledge_base.chunks)} chunks. Index: {kb.INDEX}", flush=True)
except Exception as exc:
    raise SystemExit(f"Startup failed: {exc}")

os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run",
    "/app/Docker/web_app.py", "--server.address=0.0.0.0", "--server.port=8501",
    "--server.headless=true", "--browser.gatherUsageStats=false"])
