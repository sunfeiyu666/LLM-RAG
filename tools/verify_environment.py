"""只读检查当前Python、依赖以及可选的Ollama，不安装或修改环境。"""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true", help="同时检查Ollama服务和现有千问模型")
    args = parser.parse_args()
    problems = []
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    if sys.version_info < (3, 12):
        problems.append("本项目固定依赖要求Python 3.12及以上，建议按environment.yml新建rag-langchain环境。")
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.strip().split("==", 1)
        try:
            actual = importlib.metadata.version(name)
            print(f"{name}: {actual}")
            if actual != expected:
                problems.append(f"{name}版本应为{expected}，实际为{actual}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"缺少{name}")
    if args.ollama:
        try:
            with urlopen("http://localhost:11434/api/tags", timeout=5) as response:
                names = [m["name"] for m in json.load(response).get("models", [])]
            if "qwen3:4b-instruct" not in names:
                problems.append("Ollama中缺少现有模型标签qwen3:4b-instruct，请按原来源导入或准备同名模型。")
            else:
                print("Ollama model: qwen3:4b-instruct")
        except Exception as exc:
            problems.append(f"Ollama连接失败，请先启动Ollama：{exc}")
    if problems:
        print("\n需要处理：")
        for problem in problems:
            print("-", problem)
        return 1
    print("环境检查通过。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
