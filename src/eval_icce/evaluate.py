"""ICCE 評価エントリポイント。

Usage:
    uv run python -m src.eval_icce.evaluate \
        --manifest src/eval_icce/data.csv --output out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .align import AlignmentResult, build_alignment
from .io import load_gt_intervals, load_manifest, load_vlm_log
from .metrics import compute_subject_metrics
from .report import (
    aggregate_metrics,
    aggregate_pairs,
    write_aligned_pairs,
    write_confusion,
    write_per_class_overall,
    write_per_subject_metrics,
    write_summary_markdown,
)

COVERAGE_WARN_THRESHOLD = 0.5

logger = logging.getLogger("eval_icce")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="マニフェスト CSV のパス")
    parser.add_argument("--output", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument(
        "--coverage-warn",
        type=float,
        default=COVERAGE_WARN_THRESHOLD,
        help=f"カバー率がこの値未満なら警告 (default: {COVERAGE_WARN_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    aligned_dir = args.output / "aligned_pairs"
    per_subject_cm_dir = args.output / "per_subject_confusion"

    manifest = load_manifest(args.manifest)
    if not manifest:
        logger.error("マニフェストが空です: %s", args.manifest)
        return 2
    logger.info("被験者数: %d", len(manifest))

    alignments: list[tuple[str, AlignmentResult]] = []
    subject_metrics_list = []

    for entry in manifest:
        logger.info(
            "[%s] log=%s gt=%s offset=%+0.3fs",
            entry.subject_id, entry.log_path, entry.gt_path, entry.offset_sec,
        )
        vlm = load_vlm_log(entry.log_path)
        gt = load_gt_intervals(entry.gt_path)
        alignment = build_alignment(vlm, gt, entry.offset_sec)
        write_aligned_pairs(aligned_dir, entry.subject_id, alignment)

        if alignment.coverage_rate < args.coverage_warn:
            logger.warning(
                "[%s] coverage=%.3f < %.2f — オフセット値がずれている可能性があります",
                entry.subject_id, alignment.coverage_rate, args.coverage_warn,
            )
        if alignment.parse_error_count:
            logger.warning(
                "[%s] parse_error %d 件 (VLM 出力の想定外ラベル)",
                entry.subject_id, alignment.parse_error_count,
            )

        m = compute_subject_metrics(entry.subject_id, alignment)
        subject_metrics_list.append(m)
        alignments.append((entry.subject_id, alignment))

    # 被験者ごとの集計テーブル
    write_per_subject_metrics(args.output / "per_subject_metrics.csv", subject_metrics_list)

    # 全被験者結合
    gt_e, pred_e, gt_i, pred_i = aggregate_pairs(alignments)
    overall = aggregate_metrics(alignments)

    overall_per_class_df = write_per_class_overall(
        args.output / "per_class_metrics_overall.csv", gt_e, pred_e
    )
    cm_overall = write_confusion(
        args.output / "confusion_matrix_overall.csv",
        args.output / "confusion_matrix_overall.png",
        gt_e, pred_e,
        title="Overall confusion (raw counts, parse_error excluded)",
    )
    cm_overall_norm = write_confusion(
        args.output / "confusion_matrix_overall_normalized.csv",
        args.output / "confusion_matrix_overall_normalized.png",
        gt_e, pred_e,
        title="Overall confusion (row-normalized, parse_error excluded)",
        normalize="true",
    )

    # 被験者ごとの混同行列
    per_subject_cm_dir.mkdir(parents=True, exist_ok=True)
    for subject_id, alignment in alignments:
        from .metrics import split_pairs
        gt_es, pred_es, _, _ = split_pairs(alignment.rows)
        if not gt_es:
            continue
        write_confusion(
            per_subject_cm_dir / f"{subject_id}.csv",
            per_subject_cm_dir / f"{subject_id}.png",
            gt_es, pred_es,
            title=f"Subject {subject_id} (raw counts, parse_error excluded)",
        )
        write_confusion(
            per_subject_cm_dir / f"{subject_id}_normalized.csv",
            per_subject_cm_dir / f"{subject_id}_normalized.png",
            gt_es, pred_es,
            title=f"Subject {subject_id} (row-normalized, parse_error excluded)",
            normalize="true",
        )

    write_summary_markdown(
        args.output / "summary.md",
        manifest,
        subject_metrics_list,
        overall_per_class_df,
        overall,
        cm_overall,
        cm_overall_norm,
    )

    logger.info("完了: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
