FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app/Docker STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY Docker/requirements-ui.txt /app/Docker/requirements-ui.txt
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url "$PYPI_INDEX_URL" -r /app/Docker/requirements-ui.txt
COPY src/ /app/src/
COPY metadata/ /app/metadata/
COPY data/corpus/ /app/data/corpus/
COPY Docker/ /app/Docker/
RUN mkdir -p /app/index_langchain
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
ENTRYPOINT ["python", "/app/Docker/entrypoint.py"]
