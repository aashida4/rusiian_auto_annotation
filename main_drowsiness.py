import ollama
import json
import re
import argparse
import os
import sys
import cv2
import tempfile
import threading
import queue
import subprocess
import time
import atexit
import signal
import urllib.request
import urllib.error

# Windows環境でのUnicodeEncodeErrorを防止
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- 設定項目 ---
DEFAULT_MODEL = "gemma4:31b"
NUM_CTX = 8192
BASE_PORT = 11500
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
SYSTEM_OLLAMA_MODELS = "/usr/share/ollama/.ollama/models"

# --- プロンプトの作成 ---
prompt = """You are an annotator for images of learners watching e-learning lecture videos.
Given an input image, assign labels according to the following procedure and criteria.

Task:
Annotate the image with:
1. drowsiness
2. engagement

Important procedure:
Step 1. First determine drowsiness.
Step 2. Then determine engagement.

Label definitions:

drowsiness:
- 1 = Extremely Drowsy
- 2 = Slightly Drowsy
- 3 = Alert
- x = unclear

engagement:
- 1 = Not Engaged
- 2 = Engaged
- x = unclear

Decision rules:

1. First determine drowsiness
- 3 Alert:
  No visible signs of drowsiness. Normal head pose, eyelids, and blinking. No visible difficulty in continuing lecture viewing.
- 2 Slightly Drowsy:
  Visible signs of drowsiness, such as increased blinking, yawning, drooping eyelids, rubbing eyes, or shifting posture.
  However, the learner still appears to continue learning.
- 1 Extremely Drowsy:
  Fatigue appears severe enough that the learner is no longer able to properly continue watching the lecture.
  Examples include eyes closed for extended periods, falling asleep, head dropping, or suddenly waking up.
- x unclear:
  No person is visible, the face/state is not visible enough, or a single image does not provide enough evidence.

2. Then determine engagement
- If drowsiness is 1 or 2, set engagement = 1 (Not Engaged).
- Only when drowsiness is 3, determine engagement independently.
- 2 Engaged:
  The learner is awake and appears focused on and engaged in the lecture.
- 1 Not Engaged:
  The learner shows drowsiness, or appears awake but is doing something unrelated to the lecture.
- x unclear:
  There is not enough visual evidence to judge engagement.

Output format:
Return JSON only.

{
  "drowsiness": {
    "label": "1 or 2 or 3 or x",
    "reason": "brief visual evidence from the image"
  },
  "engagement": {
    "label": "1 or 2 or x",
    "reason": "brief visual evidence from the image"
  }
}

Additional rules:
- Base the decision only on observable visual evidence in the image.
- Do not infer unobservable mental state beyond what is visually supported.
- If the evidence is weak or insufficient, use x.
- Keep reasons brief and concrete.
"""

def extract_json(text):
    """レスポンスからJSON部分を抽出してパースする（ネストJSON対応）"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ``` で囲われているケース
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # 最初の '{' から対応する '}' までを抽出
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError(f"JSONを抽出できませんでした: {text[:200]}")

def analyze_learning_scene(image_path, model_name, client=None):
    try:
        gen = client.generate if client is not None else ollama.generate
        response = gen(
            model=model_name,
            prompt=prompt,
            images=[image_path],
            stream=False,
            options={"num_ctx": NUM_CTX},
        )
        result = extract_json(response['response'])
        return result
    except Exception as e:
        return {"error": str(e)}

def detect_gpu_count():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        n = len([l for l in out.strip().splitlines() if l.strip()])
        return max(1, n)
    except Exception:
        return 1

def try_unload_default(model_name):
    """既存のデフォルト Ollama (11434) が掴んでいるモデルを解放させる (best-effort)"""
    try:
        req = urllib.request.Request(
            f"{DEFAULT_OLLAMA_HOST}/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
        print(f"[info] 既存 Ollama ({DEFAULT_OLLAMA_HOST}) のモデル {model_name} をアンロードしました")
        time.sleep(3)
    except Exception as e:
        print(f"[warn] 既存 Ollama アンロード要求失敗（続行）: {e}")

def wait_ready(host, timeout=120):
    client = ollama.Client(host=host)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.list()
            return client
        except Exception:
            time.sleep(0.5)
    return None

def spawn_ollama_servers(num_gpus, model_name):
    """各 GPU に 1 インスタンスずつ ollama serve を立てて Client のリストを返す"""
    try_unload_default(model_name)
    procs = []
    clients = []

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

    for i in range(num_gpus):
        port = BASE_PORT + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        if os.path.isdir(SYSTEM_OLLAMA_MODELS):
            env["OLLAMA_MODELS"] = SYSTEM_OLLAMA_MODELS
        print(f"[info] Ollama 起動中: GPU {i} on 127.0.0.1:{port}")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proc)

    for i in range(num_gpus):
        host = f"http://127.0.0.1:{BASE_PORT + i}"
        client = wait_ready(host, timeout=120)
        if client is None:
            raise RuntimeError(f"Ollama サーバ起動タイムアウト: {host}")
        clients.append(client)
        print(f"[info] Ollama 準備完了: {host} (GPU {i})")
    return clients

def _format_result(output):
    drowsiness = output.get("drowsiness") or {}
    engagement = output.get("engagement") or {}
    return (
        f"  drowsiness: {drowsiness.get('label', '?')} - {drowsiness.get('reason', '(no reason)')}\n"
        f"  engagement: {engagement.get('label', '?')} - {engagement.get('reason', '(no reason)')}"
    )

def analyze_image(image_path, model_name):
    """単一画像を解析して結果を表示する"""
    print(f"Analyzing {image_path}...")
    output = analyze_learning_scene(image_path, model_name)
    if "error" in output:
        print(f"Error: {output['error']}")
    else:
        print(_format_result(output))

def analyze_video(video_path, interval, num_gpus, model_name):
    """動画からインターバルごとにフレームを抽出して解析する"""
    import datetime
    start_time = datetime.datetime.now().isoformat()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 動画を開けませんでした: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"動画: {video_path}")
    print(f"  モデル: {model_name}")
    print(f"  FPS: {fps:.1f}, 総フレーム数: {total_frames}, 長さ: {duration:.1f}秒")
    print(f"  解析間隔: {interval}秒")
    print(f"  並列 Ollama インスタンス数: {num_gpus}")
    print()

    if num_gpus >= 2:
        clients = spawn_ollama_servers(num_gpus, model_name)
    else:
        clients = [ollama.Client()]

    results = []
    results_lock = threading.Lock()
    frame_interval = max(1, int(fps * interval))

    with tempfile.TemporaryDirectory() as tmpdir:
        q: "queue.Queue" = queue.Queue(maxsize=max(2, num_gpus * 2))

        def producer():
            idx = 0
            try:
                while idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ts = idx / fps
                    path = os.path.join(tmpdir, f"frame_{idx:06d}.png")
                    cv2.imwrite(path, frame)
                    q.put((idx, ts, path))
                    idx += frame_interval
            finally:
                for _ in range(len(clients)):
                    q.put(None)

        def consumer(client, worker_id):
            while True:
                item = q.get()
                if item is None:
                    break
                frame_idx, timestamp, tmp_path = item
                output = analyze_learning_scene(tmp_path, model_name, client=client)
                with results_lock:
                    tag = f"[GPU{worker_id}] {timestamp:.1f}秒 (フレーム {frame_idx})"
                    if "error" in output:
                        print(f"{tag} Error: {output['error']}", flush=True)
                        results.append({"time": round(timestamp, 1), "frame": frame_idx, "error": output['error']})
                    else:
                        print(f"{tag}", flush=True)
                        print(_format_result(output), flush=True)
                        results.append({"time": round(timestamp, 1), "frame": frame_idx, **output})

        prod = threading.Thread(target=producer, daemon=True)
        cons = [threading.Thread(target=consumer, args=(c, i), daemon=True) for i, c in enumerate(clients)]
        prod.start()
        for t in cons:
            t.start()
        prod.join()
        for t in cons:
            t.join()

    cap.release()

    end_time = datetime.datetime.now().isoformat()

    results.sort(key=lambda r: r["frame"])

    # 結果をJSONファイルに保存
    base = os.path.splitext(video_path)[0]
    output_path = base + "_drowsiness_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {output_path}")

    # メタデータを保存
    frame_interval = max(1, int(fps * interval))
    metadata = {
        "model": model_name,
        "num_ctx": NUM_CTX,
        "video": os.path.basename(video_path),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration, 1),
        "interval_sec": interval,
        "frame_interval": frame_interval,
        "analyzed_frames": len(results),
        "num_gpus": num_gpus,
        "start_time": start_time,
        "end_time": end_time,
    }
    meta_path = base + "_drowsiness_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"メタデータを保存しました: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="e-learning受講者のdrowsiness/engagementを判定する")
    parser.add_argument("input", help="画像ファイルまたは動画ファイルのパス")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"使用する Ollama モデル名。デフォルト: {DEFAULT_MODEL}")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="動画モード時のフレーム抽出間隔（秒）。デフォルト: 1.0")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="並列に立てる Ollama インスタンス数。省略時は nvidia-smi で自動検出")
    args = parser.parse_args()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(args.input)[1].lower()

    if ext in video_exts:
        num_gpus = args.num_gpus if args.num_gpus is not None else detect_gpu_count()
        analyze_video(args.input, args.interval, num_gpus, args.model)
    else:
        analyze_image(args.input, args.model)
