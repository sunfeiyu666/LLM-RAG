"""Standard-library-only check, run inside the container before Streamlit starts."""
import json
import os
import urllib.error
import urllib.request


def check_ollama():
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=8) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise RuntimeError(
            f"无法连接 Ollama：{base}。请启动宿主机 Ollama；容器访问宿主机时使用 "
            "host.docker.internal，并检查 OLLAMA_HOST 监听地址。详见 Docker/README.md。"
        ) from exc
    names = {m.get("name", m.get("model", "")) for m in payload.get("models", [])}
    if model not in names and (":" in model or model + ":latest" not in names):
        raise RuntimeError(f"Ollama 中没有模型 {model}；实际模型：{', '.join(sorted(names))}。请核对配置。")
    return {"ollama_url": base, "model": model, "available": True}


if __name__ == "__main__":
    try:
        print(json.dumps(check_ollama(), ensure_ascii=False), flush=True)
    except Exception as exc:
        raise SystemExit(str(exc))
