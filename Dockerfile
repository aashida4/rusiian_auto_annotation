# syntax=docker/dockerfile:1.7
# GPU サーバー向け Docker イメージ。
# - ベースは python:3.12-slim（Ollama が自前で CUDA ライブラリを同梱するため、
#   ホスト側に NVIDIA Container Toolkit があれば `--gpus all` で GPU を利用できる）
# - 依存関係は uv.lock に従って `uv sync --frozen` で再現する
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_MODELS=/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    PATH=/opt/app/.venv/bin:/usr/local/bin:/usr/bin:/bin

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        ffmpeg \
        ca-certificates \
        curl \
        procps \
        tini \
        zstd \
 && rm -rf /var/lib/apt/lists/*

# Ollama 本体 (ホストの NVIDIA ドライバ越しに GPU を掴む)
RUN curl -fsSL https://ollama.com/install.sh | sh

# uv をインストール
RUN pip install --no-cache-dir uv

WORKDIR /opt/app

# 依存解決を先に行うことでビルドキャッシュを効かせる
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# アプリ本体
COPY src/ ./src/
COPY docker/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh \
 && mkdir -p /models /work \
 && chmod -R a+rX /opt/app

WORKDIR /work

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/app/entrypoint.sh"]
CMD ["main", "--help"]
