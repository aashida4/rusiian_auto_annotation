import ollama
import json
import re
import argparse
import os
import sys
import cv2
import tempfile

# Windows環境でのUnicodeEncodeErrorを防止
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- 設定項目 ---
MODEL_NAME = "qwen3-vl:latest"

# 判定に使用するカテゴリ定義（プロンプトに挿入）
class_definitions = [
    {"class": "Dictionary", "description": "辞書（紙・電子）を注視している"},
    {"class": "Paper", "description": "罫線のみの作文用ミニッツペーパーを注視している（書き込みの可能性あり）"},
    {"class": "Task", "description": "ロシア語が印字された問題用紙を注視している"},
    {"class": "Memo", "description": "罫線のないメモ用紙を注視している（書き込みがある可能性あり）"},
    {"class": "Others", "description": "その他（手、机、コンピュータ，不明瞭な対象）"}
]

# 判定ルール
# - ロシア語の本文が印刷されていれば 'Task'
# - 罫線のみでロシア語の本文がなければ 'Paper'
# - 視線の先を優先して判定してください。

# --- プロンプトの作成 ---
prompt = f"""
これはロシア語学習における、読解と作文作業の一人称視点画像です。
画像内の「赤色の点と緑色の円」が示す学習者の注視点が、以下カテゴリ定義に基づいてどのカテゴリに該当するか判定してください。

# カテゴリ定義
{json.dumps(class_definitions, ensure_ascii=False, indent=2)}

# 出力形式
必ず以下のJSON形式のみで回答してください。
{{
  "prediction": "Dictionary | Paper | Task | Memo | Others",
  "reasoning": "判断理由"
}}
"""

def extract_json(text):
    """レスポンスからJSON部分を抽出してパースする"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"JSONを抽出できませんでした: {text[:200]}")

def analyze_learning_scene(image_path):
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            images=[image_path],
            stream=False,
        )
        result = extract_json(response['response'])
        return result
    except Exception as e:
        return {"error": str(e)}

def analyze_image(image_path):
    """単一画像を解析して結果を表示する"""
    print(f"Analyzing {image_path}...")
    output = analyze_learning_scene(image_path)
    if "error" in output:
        print(f"Error: {output['error']}")
    else:
        print(f"【判定結果】: {output.get('prediction', '不明')}")
        print(f"【判断根拠】: {output.get('reasoning', '(根拠なし)')}")

def analyze_video(video_path, interval):
    """動画からインターバルごとにフレームを抽出して解析する"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 動画を開けませんでした: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"動画: {video_path}")
    print(f"  FPS: {fps:.1f}, 総フレーム数: {total_frames}, 長さ: {duration:.1f}秒")
    print(f"  解析間隔: {interval}秒")
    print()

    results = []
    frame_interval = int(fps * interval)
    if frame_interval < 1:
        frame_interval = 1

    frame_idx = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            tmp_path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(tmp_path, frame)

            print(f"--- {timestamp:.1f}秒 (フレーム {frame_idx}) ---")
            output = analyze_learning_scene(tmp_path)

            if "error" in output:
                print(f"  Error: {output['error']}")
                results.append({"time": round(timestamp, 1), "frame": frame_idx, "error": output['error']})
            else:
                print(f"  【判定結果】: {output.get('prediction', '不明')}")
                print(f"  【判断根拠】: {output.get('reasoning', '(根拠なし)')}")
                results.append({"time": round(timestamp, 1), "frame": frame_idx, **output})

            frame_idx += frame_interval

    cap.release()

    # 結果をJSONファイルに保存
    output_path = os.path.splitext(video_path)[0] + "_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ロシア語学習シーンの注視対象を判定する")
    parser.add_argument("input", help="画像ファイルまたは動画ファイルのパス")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="動画モード時のフレーム抽出間隔（秒）。デフォルト: 1.0")
    args = parser.parse_args()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(args.input)[1].lower()

    if ext in video_exts:
        analyze_video(args.input, args.interval)
    else:
        analyze_image(args.input)
