#!/bin/bash
# Docker コンテナのエントリポイント。
# 第1引数（または環境変数 APP）でサブコマンドを切り替える:
#   main                 -> src/detection/main.py
#   drowsiness           -> src/detection/main_drowsiness.py
#   evaluate             -> src/evaluation/evaluate.py
#   evaluate_drowsiness  -> src/evaluation/evaluate_drowsiness.py
#   visualize            -> src/evaluation/visualize.py
#   eval_icce            -> src/eval_icce/evaluate.py
#   ollama               -> ollama CLI を直接実行
#   serve                -> ollama serve を 0.0.0.0 で待ち受け
#   bash | sh            -> シェル起動 (デバッグ用)
#
# 環境変数:
#   USE_EXTERNAL_OLLAMA=1  -> entrypoint 内での ollama serve 起動をスキップ
#                              (--num-gpus >= 2 の場合は Python 側が GPU 毎に
#                               ollama serve を起動するので 1 を推奨)
#   OLLAMA_MODELS          -> モデル格納先 (デフォルト /models)
#   OLLAMA_HOST            -> Ollama API のホスト (デフォルト 127.0.0.1:11434)

set -eu

: "${OLLAMA_MODELS:=/models}"
: "${OLLAMA_HOST:=127.0.0.1:11434}"
export OLLAMA_MODELS OLLAMA_HOST

APP="${APP:-}"
if [ -z "$APP" ] && [ $# -gt 0 ]; then
    case "$1" in
        main|drowsiness|evaluate|evaluate_drowsiness|visualize|eval_icce|ollama|serve|bash|sh)
            APP="$1"; shift ;;
    esac
fi
APP="${APP:-main}"

case "$APP" in
    main)                SCRIPT="src/detection/main.py" ;;
    drowsiness)          SCRIPT="src/detection/main_drowsiness.py" ;;
    evaluate)            SCRIPT="src/evaluation/evaluate.py" ;;
    evaluate_drowsiness) SCRIPT="src/evaluation/evaluate_drowsiness.py" ;;
    visualize)           SCRIPT="src/evaluation/visualize.py" ;;
    eval_icce)           SCRIPT="src/eval_icce/evaluate.py" ;;
    ollama)              exec ollama "$@" ;;
    serve)
        export OLLAMA_HOST="${OLLAMA_HOST_SERVE:-0.0.0.0:11434}"
        exec ollama serve
        ;;
    bash|sh)             exec "$APP" "$@" ;;
    *)
        echo "Unknown APP: $APP" >&2
        echo "Available: main, drowsiness, evaluate, evaluate_drowsiness, visualize, eval_icce, ollama, serve, bash" >&2
        exit 2
        ;;
esac

# 外部 Ollama を利用する場合は内蔵 ollama serve を起動しない
if [ "${USE_EXTERNAL_OLLAMA:-0}" = "1" ]; then
    exec python "/opt/app/${SCRIPT}" "$@"
fi

ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 60); do
    if curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
    echo "Ollama サーバの起動に失敗しました。ログ:" >&2
    cat /tmp/ollama.log >&2
    exit 1
fi

exec python "/opt/app/${SCRIPT}" "$@"
