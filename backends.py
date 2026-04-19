"""推論バックエンド抽象レイヤ

Ollama と llama.cpp (llama-server) の両方を、ollama.Client.generate() と同等の
インタフェースで扱えるようにするためのモジュール。
"""

import atexit
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


LLAMA_BASE_PORT = 11600


class LlamaCppClient:
    """ollama.Client.generate() と互換な形で llama-server の OpenAI 互換 API を叩くクライアント"""

    def __init__(self, host: str, request_timeout: float = 600.0):
        self.host = host.rstrip("/")
        self.request_timeout = request_timeout

    def list(self):
        """疎通確認用（ollama.Client.list の互換）"""
        req = urllib.request.Request(f"{self.host}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
        return True

    def generate(self, model=None, prompt="", images=None, stream=False, options=None, **_):
        del stream, options  # num_ctx は llama-server 起動時に指定するため未使用
        images = images or []

        content = [{"type": "text", "text": prompt}]
        for img in images:
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(img)[1].lower().lstrip(".") or "png"
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64}"},
            })

        payload = {
            "model": model or "local",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
            data = json.loads(r.read().decode("utf-8"))

        text = data["choices"][0]["message"]["content"]
        return {"response": text}


def wait_llama_ready(host: str, timeout: float = 300.0):
    client = LlamaCppClient(host)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.list()
            return client
        except Exception:
            time.sleep(0.5)
    return None


def _register_shutdown(procs):
    def shutdown():
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    atexit.register(shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sig)

        def handler(signum, frame, _prev=prev):
            shutdown()
            if callable(_prev):
                _prev(signum, frame)
            else:
                sys.exit(130)

        signal.signal(sig, handler)


def spawn_llama_servers(
    num_gpus: int,
    model_path: str,
    mmproj_path: str,
    binary: str = "llama-server",
    base_port: int = LLAMA_BASE_PORT,
    ctx_size: int = 8192,
    ngl: int = 999,
    extra_args=None,
    ready_timeout: float = 300.0,
):
    """各 GPU に 1 インスタンスずつ llama-server を立てて LlamaCppClient のリストを返す"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF モデルが見つかりません: {model_path}")
    if not os.path.isfile(mmproj_path):
        raise FileNotFoundError(f"mmproj ファイルが見つかりません: {mmproj_path}")

    procs = []
    clients = []
    _register_shutdown(procs)

    for i in range(num_gpus):
        port = base_port + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [
            binary,
            "--model", model_path,
            "--mmproj", mmproj_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(ngl),
            "--jinja",
        ]
        if extra_args:
            cmd.extend(extra_args)
        print(f"[info] llama-server 起動中: GPU {i} on 127.0.0.1:{port}")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proc)

    for i in range(num_gpus):
        host = f"http://127.0.0.1:{base_port + i}"
        client = wait_llama_ready(host, timeout=ready_timeout)
        if client is None:
            raise RuntimeError(f"llama-server 起動タイムアウト: {host}")
        clients.append(client)
        print(f"[info] llama-server 準備完了: {host} (GPU {i})")

    return clients


def add_backend_cli_args(parser):
    """main.py / main_drowsiness.py 共通のバックエンド関連 CLI 引数を追加"""
    parser.add_argument(
        "--backend", choices=["ollama", "llama.cpp"],
        default=os.environ.get("BACKEND", "ollama"),
        help="推論バックエンド (default: ollama, env: BACKEND)",
    )
    parser.add_argument(
        "--llama-model", default=os.environ.get("LLAMA_MODEL"),
        help="llama.cpp 用 GGUF モデルのパス (env: LLAMA_MODEL)",
    )
    parser.add_argument(
        "--llama-mmproj", default=os.environ.get("LLAMA_MMPROJ"),
        help="llama.cpp 用 mmproj GGUF のパス (env: LLAMA_MMPROJ)",
    )
    parser.add_argument(
        "--llama-server-bin", default=os.environ.get("LLAMA_SERVER_BIN", "llama-server"),
        help="llama-server バイナリ (default: llama-server, env: LLAMA_SERVER_BIN)",
    )
    parser.add_argument(
        "--llama-host", default=os.environ.get("LLAMA_HOST"),
        help="既存 llama-server の URL。指定時は起動をスキップ (env: LLAMA_HOST)",
    )
    parser.add_argument(
        "--llama-ngl", type=int,
        default=int(os.environ.get("LLAMA_NGL", "999")),
        help="GPU にオフロードするレイヤ数 (default: 999=全て, env: LLAMA_NGL)",
    )


def create_llama_clients(args, num_gpus: int, ctx_size: int):
    """args (add_backend_cli_args の結果) と num_gpus から LlamaCppClient のリストを作る"""
    if args.llama_host:
        return [LlamaCppClient(args.llama_host)]
    if not args.llama_model or not args.llama_mmproj:
        raise SystemExit(
            "Error: --backend llama.cpp 使用時は --llama-model と --llama-mmproj が必要です "
            "(もしくは既存サーバを使うなら --llama-host)"
        )
    return spawn_llama_servers(
        num_gpus=max(1, num_gpus),
        model_path=args.llama_model,
        mmproj_path=args.llama_mmproj,
        binary=args.llama_server_bin,
        ctx_size=ctx_size,
        ngl=args.llama_ngl,
    )
