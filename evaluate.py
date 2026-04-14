import json
import argparse
import os
import bisect


SKIP_LABELS = {"x", "unknown"}


def parse_time(time_str):
    """HH:MM:SS.mmm 形式の時刻を秒に変換する"""
    parts = time_str.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def load_annotations(annotation_path):
    """アノテーションファイルを読み込み、(start_sec, end_sec, label) のリストを返す"""
    intervals = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            # 列: ツール名, (空), 開始時刻, 終了時刻, 持続時間, カテゴリ
            start = parse_time(cols[2])
            end = parse_time(cols[3])
            label = cols[5].strip()
            intervals.append((start, end, label))
    return intervals


def lookup_label(intervals, starts, timestamp):
    """タイムスタンプが該当するアノテーション区間のラベルを返す"""
    idx = bisect.bisect_right(starts, timestamp) - 1
    if idx < 0:
        return None
    start, end, label = intervals[idx]
    if start <= timestamp < end:
        return label
    return None


def compute_metrics(pairs):
    """(ground_truth, prediction) のペアリストから精度指標を計算する"""
    labels = sorted(set(gt for gt, _ in pairs) | set(pred for _, pred in pairs))
    # 混同行列
    matrix = {gt: {pred: 0 for pred in labels} for gt in labels}
    correct = 0
    for gt, pred in pairs:
        matrix[gt][pred] += 1
        if gt == pred:
            correct += 1

    accuracy = correct / len(pairs) if pairs else 0

    # カテゴリ別 Precision / Recall / F1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gt][label] for gt in labels if gt != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return accuracy, per_class, matrix, labels


def print_results(accuracy, per_class, matrix, labels, mismatches, skipped, total):
    """結果をコンソールに表示する"""
    print(f"\n{'='*60}")
    print(f"  評価結果")
    print(f"{'='*60}")
    print(f"  比較対象フレーム数: {total - skipped}")
    print(f"  スキップ数 (x/unknown): {skipped}")
    print(f"  全体正解率 (Accuracy): {accuracy:.1%}")
    print()

    # カテゴリ別指標
    print(f"  {'カテゴリ':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for label in labels:
        m = per_class[label]
        print(f"  {label:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")
    print()

    # 混同行列
    print(f"  混同行列 (行=正解, 列=VLM予測)")
    col_width = max(len(l) for l in labels) + 2
    header = "  " + " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
    print(header)
    for gt in labels:
        row = f"  {gt:<{col_width}}" + "".join(f"{matrix[gt][pred]:>{col_width}}" for pred in labels)
        print(row)
    print()

    # 不一致リスト
    if mismatches:
        print(f"  不一致一覧 ({len(mismatches)}件)")
        print(f"  {'時刻(秒)':>10}  {'正解':<12} {'VLM予測':<12}")
        print(f"  {'-'*36}")
        for time, gt, pred in mismatches:
            print(f"  {time:>10.1f}  {gt:<12} {pred:<12}")
    print()


def save_csv(pairs_with_time, output_path):
    """比較結果をCSVに保存する"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("time,ground_truth,prediction,correct\n")
        for time, gt, pred in pairs_with_time:
            f.write(f"{time},{gt},{pred},{1 if gt == pred else 0}\n")
    print(f"CSVを保存しました: {output_path}")


def evaluate(json_path, annotation_path, output_path=None, offset=0.0):
    """VLM結果とアノテーションを比較する。offsetはVLM時刻に加算してアノテーション時刻に合わせる。"""
    intervals = load_annotations(annotation_path)
    starts = [s for s, _, _ in intervals]

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if offset != 0.0:
        print(f"  時刻オフセット: {offset:+.1f}秒 (VLM時刻にこの値を加算してアノテーションと照合)")

    pairs = []
    mismatches = []
    pairs_with_time = []
    skipped = 0

    for entry in results:
        timestamp = entry["time"]
        prediction = entry.get("prediction")
        if not prediction:
            skipped += 1
            continue

        gt_label = lookup_label(intervals, starts, timestamp + offset)
        if gt_label is None or gt_label in SKIP_LABELS:
            skipped += 1
            continue

        pairs.append((gt_label, prediction))
        pairs_with_time.append((timestamp, gt_label, prediction))
        if gt_label != prediction:
            mismatches.append((timestamp, gt_label, prediction))

    if not pairs:
        print("比較可能なフレームがありませんでした。")
        return

    accuracy, per_class, matrix, labels = compute_metrics(pairs)
    print_results(accuracy, per_class, matrix, labels, mismatches, skipped, len(results))

    if output_path:
        save_csv(pairs_with_time, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM判定結果と人間アノテーションを比較して精度を評価する"
    )
    parser.add_argument("video", help="入力動画ファイルのパス（JSONパスの自動推定に使用）")
    parser.add_argument("--json", help="結果JSONファイルのパス（省略時は <動画名>_results.json）")
    parser.add_argument("--annotation", required=True, help="人間アノテーションファイルのパス")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="VLM時刻に加算する時刻オフセット（秒）。アノテーションが動画より遅れている場合は正の値")
    parser.add_argument("--output", help="比較結果CSVの出力パス（オプション）")
    args = parser.parse_args()

    json_path = args.json or (os.path.splitext(args.video)[0] + "_results.json")

    if not os.path.exists(json_path):
        print(f"Error: 結果JSONが見つかりません: {json_path}")
        exit(1)
    if not os.path.exists(args.annotation):
        print(f"Error: アノテーションファイルが見つかりません: {args.annotation}")
        exit(1)

    evaluate(json_path, args.annotation, args.output, offset=args.offset)
