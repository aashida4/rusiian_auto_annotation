import json
import argparse
import os
import cv2
import numpy as np


# カテゴリごとの表示色 (BGR)
CATEGORY_COLORS = {
    "Dictionary": (255, 165, 0),   # オレンジ
    "Paper":      (0, 200, 0),     # 緑
    "Task":       (0, 100, 255),   # 赤系
    "Others":     (128, 128, 128), # グレー
}

DEFAULT_COLOR = (255, 255, 255)


def draw_label(frame, prediction, timestamp, frame_idx, reasoning=None):
    """フレームに判定結果を重畳描画する"""
    h, w = frame.shape[:2]
    color = CATEGORY_COLORS.get(prediction, DEFAULT_COLOR)

    # 半透明の背景バー
    overlay = frame.copy()
    bar_height = 80 if reasoning else 60
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 時刻とフレーム番号
    time_text = f"Time: {timestamp:.1f}s  Frame: {frame_idx}"
    cv2.putText(frame, time_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 判定結果ラベル（カテゴリ色で表示）
    label_text = f"Prediction: {prediction}"
    cv2.putText(frame, label_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # カテゴリ色の帯を左端に表示
    cv2.rectangle(frame, (0, 0), (6, bar_height), color, -1)

    return frame


def visualize_results(video_path, json_path, output_dir, max_frames=None):
    """結果JSONに基づいて動画フレームに判定結果を重畳した画像を出力する"""
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 動画を開けませんでした: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画: {video_path}")
    print(f"  FPS: {fps:.1f}, 総フレーム数: {total_frames}")
    print(f"  結果エントリ数: {len(results)}")

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for entry in results:
        if max_frames is not None and count >= max_frames:
            break

        frame_idx = entry["frame"]
        timestamp = entry.get("time", frame_idx / fps)
        prediction = entry.get("prediction", "Error")
        reasoning = entry.get("reasoning")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: フレーム {frame_idx} を読み取れませんでした。スキップします。")
            continue

        frame = draw_label(frame, prediction, timestamp, frame_idx, reasoning)

        out_path = os.path.join(output_dir, f"{timestamp:08.1f}s_{prediction}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        count += 1

        if count % 20 == 0:
            print(f"  {count}/{len(results)} 枚出力済み...")

    cap.release()
    print(f"\n完了: {count} 枚の画像を {output_dir} に保存しました。")


def create_summary_grid(output_dir, json_path, grid_cols=6, thumb_width=320):
    """全フレームのサムネイルをグリッド状にまとめた一覧画像を生成する"""
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    images = []
    for entry in results:
        timestamp = entry.get("time", 0)
        prediction = entry.get("prediction", "Error")
        pattern = f"{timestamp:08.1f}s_{prediction}.jpg"
        path = os.path.join(output_dir, pattern)
        if os.path.exists(path):
            images.append(path)

    if not images:
        print("サマリー用の画像が見つかりませんでした。")
        return

    # サムネイルサイズを計算
    sample = cv2.imread(images[0])
    if sample is None:
        return
    orig_h, orig_w = sample.shape[:2]
    thumb_height = int(thumb_width * orig_h / orig_w)

    # グリッド配置
    grid_rows = (len(images) + grid_cols - 1) // grid_cols
    grid_w = thumb_width * grid_cols
    grid_h = thumb_height * grid_rows
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_width, thumb_height))
        row = i // grid_cols
        col = i % grid_cols
        y = row * thumb_height
        x = col * thumb_width
        grid_img[y:y + thumb_height, x:x + thumb_width] = thumb

    summary_path = os.path.join(output_dir, "summary_grid.jpg")
    cv2.imwrite(summary_path, grid_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"サマリー画像を保存しました: {summary_path} ({grid_cols}x{grid_rows}, {len(images)}枚)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="判定結果JSONに基づいて動画フレームに結果を重畳表示した画像を出力する"
    )
    parser.add_argument("video", help="入力動画ファイルのパス")
    parser.add_argument("--json", help="結果JSONファイルのパス（省略時は <動画名>_results.json）")
    parser.add_argument("--output", help="出力ディレクトリ（省略時は <動画名>_vis/）")
    parser.add_argument("--max", type=int, default=None, help="出力する最大フレーム数")
    parser.add_argument("--grid", action="store_true", help="サマリーグリッド画像も生成する")
    parser.add_argument("--grid-cols", type=int, default=6, help="グリッドの列数（デフォルト: 6）")
    args = parser.parse_args()

    video_path = args.video
    json_path = args.json or (os.path.splitext(video_path)[0] + "_results.json")
    output_dir = args.output or (os.path.splitext(video_path)[0] + "_vis")

    if not os.path.exists(json_path):
        print(f"Error: 結果JSONが見つかりません: {json_path}")
        exit(1)

    visualize_results(video_path, json_path, output_dir, max_frames=args.max)

    if args.grid:
        create_summary_grid(output_dir, json_path, grid_cols=args.grid_cols)
