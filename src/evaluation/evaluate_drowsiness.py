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
    """アノテーションファイルを読み込み、(start_sec, end_sec, label) のリストを返す。
    5列 (ツール名, 空, 開始, 終了, ラベル) と
    6列 (ツール名, 空, 開始, 終了, 持続時間, ラベル) の両方に対応。"""
    intervals = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) >= 6:
                start = parse_time(cols[2])
                end = parse_time(cols[3])
                label = cols[5].strip()
            elif len(cols) >= 5:
                start = parse_time(cols[2])
                end = parse_time(cols[3])
                label = cols[4].strip()
            else:
                continue
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
    matrix = {gt: {pred: 0 for pred in labels} for gt in labels}
    correct = 0
    for gt, pred in pairs:
        matrix[gt][pred] += 1
        if gt == pred:
            correct += 1

    accuracy = correct / len(pairs) if pairs else 0

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


def print_results(task_name, accuracy, per_class, matrix, labels, mismatches, compared, skipped):
    """結果をコンソールに表示する"""
    print(f"\n{'='*60}")
    print(f"  {task_name} 評価結果")
    print(f"{'='*60}")
    print(f"  比較対象フレーム数: {compared}")
    print(f"  スキップ数 (x/unknown/欠損): {skipped}")
    print(f"  全体正解率 (Accuracy): {accuracy:.1%}")
    print()

    print(f"  {'ラベル':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for label in labels:
        m = per_class[label]
        print(f"  {label:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")
    print()

    print(f"  混同行列 (行=正解, 列=VLM予測)")
    col_width = max(len(l) for l in labels) + 2
    header = "  " + " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
    print(header)
    for gt in labels:
        row = f"  {gt:<{col_width}}" + "".join(f"{matrix[gt][pred]:>{col_width}}" for pred in labels)
        print(row)
    print()

    if mismatches:
        print(f"  不一致一覧 ({len(mismatches)}件)")
        print(f"  {'時刻(秒)':>10}  {'正解':<12} {'VLM予測':<12}")
        print(f"  {'-'*36}")
        for time_val, gt, pred in mismatches:
            print(f"  {time_val:>10.1f}  {gt:<12} {pred:<12}")
    print()


def evaluate_task(task_name, results, annotation_path, offset):
    """1つのタスク (drowsiness or engagement) を評価する"""
    intervals = load_annotations(annotation_path)
    starts = [s for s, _, _ in intervals]

    pairs = []
    mismatches = []
    skipped = 0

    for entry in results:
        timestamp = entry["time"]
        task_data = entry.get(task_name)
        if not task_data or "label" not in task_data:
            skipped += 1
            continue

        prediction = str(task_data["label"]).strip()
        if prediction in SKIP_LABELS:
            skipped += 1
            continue

        gt_label = lookup_label(intervals, starts, timestamp + offset)
        if gt_label is None or gt_label in SKIP_LABELS:
            skipped += 1
            continue

        pairs.append((gt_label, prediction))
        if gt_label != prediction:
            mismatches.append((timestamp, gt_label, prediction))

    if not pairs:
        print(f"\n{task_name}: 比較可能なフレームがありませんでした。")
        return None

    accuracy, per_class, matrix, labels = compute_metrics(pairs)
    print_results(task_name, accuracy, per_class, matrix, labels, mismatches, len(pairs), skipped)
    return pairs


def save_csv(results, drowsiness_intervals, engagement_intervals, d_starts, e_starts, offset, output_path):
    """比較結果をCSVに保存する"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("time,drowsiness_gt,drowsiness_pred,drowsiness_correct,engagement_gt,engagement_pred,engagement_correct\n")
        for entry in results:
            timestamp = entry["time"]
            d_data = entry.get("drowsiness") or {}
            e_data = entry.get("engagement") or {}
            d_pred = str(d_data.get("label", "")).strip()
            e_pred = str(e_data.get("label", "")).strip()
            d_gt = lookup_label(drowsiness_intervals, d_starts, timestamp + offset) or ""
            e_gt = lookup_label(engagement_intervals, e_starts, timestamp + offset) or ""
            d_correct = 1 if d_gt and d_pred and d_gt == d_pred else 0
            e_correct = 1 if e_gt and e_pred and e_gt == e_pred else 0
            f.write(f"{timestamp},{d_gt},{d_pred},{d_correct},{e_gt},{e_pred},{e_correct}\n")
    print(f"CSVを保存しました: {output_path}")


def evaluate(json_path, drowsiness_annotation, engagement_annotation, output_path=None, offset=0.0):
    """VLM結果とアノテーションを比較する"""
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if offset != 0.0:
        print(f"  時刻オフセット: {offset:+.1f}秒 (VLM時刻にこの値を加算してアノテーションと照合)")

    evaluate_task("drowsiness", results, drowsiness_annotation, offset)
    evaluate_task("engagement", results, engagement_annotation, offset)

    if output_path:
        d_intervals = load_annotations(drowsiness_annotation)
        e_intervals = load_annotations(engagement_annotation)
        d_starts = [s for s, _, _ in d_intervals]
        e_starts = [s for s, _, _ in e_intervals]
        save_csv(results, d_intervals, e_intervals, d_starts, e_starts, offset, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drowsiness/Engagement VLM判定結果と人間アノテーションを比較して精度を評価する"
    )
    parser.add_argument("video", help="入力動画ファイルのパス（JSONパスの自動推定に使用）")
    parser.add_argument("--json", help="結果JSONファイルのパス（省略時は <動画名>_drowsiness_results.json）")
    parser.add_argument("--drowsiness", required=True, help="drowsinessアノテーションTSVのパス")
    parser.add_argument("--engagement", required=True, help="engagementアノテーションTSVのパス")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="VLM時刻に加算する時刻オフセット（秒）")
    parser.add_argument("--output", help="比較結果CSVの出力パス（オプション）")
    args = parser.parse_args()

    json_path = args.json or (os.path.splitext(args.video)[0] + "_drowsiness_results.json")

    if not os.path.exists(json_path):
        print(f"Error: 結果JSONが見つかりません: {json_path}")
        exit(1)
    for path, name in [(args.drowsiness, "drowsiness"), (args.engagement, "engagement")]:
        if not os.path.exists(path):
            print(f"Error: {name}アノテーションファイルが見つかりません: {path}")
            exit(1)

    evaluate(json_path, args.drowsiness, args.engagement, args.output, offset=args.offset)
