"""CSV / PNG / Markdown レポートの書き出し。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .align import AlignmentResult
from .io import ManifestEntry
from .metrics import LABELS, SubjectMetrics, confusion, per_class_report, split_pairs


def write_aligned_pairs(out_dir: Path, subject_id: str, alignment: AlignmentResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "gt_label": r.gt_label,
                "vlm_label": r.vlm_label,
                "vlm_raw_label": r.vlm_raw_label,
                "in_labeled_interval": r.in_labeled_interval,
            }
            for r in alignment.rows
        ]
    )
    path = out_dir / f"{subject_id}.csv"
    df.to_csv(path, index=False)
    return path


def write_per_subject_metrics(out_path: Path, metrics_list: list[SubjectMetrics]) -> None:
    rows = [
        {
            "subject_id": m.subject_id,
            "n_total": m.n_total,
            "n_parse_error": m.n_parse_error,
            "n_used": m.n_used_excluding_pe,
            "coverage_rate": m.coverage_rate,
            "accuracy_excl_pe": m.accuracy_excl_pe,
            "macro_f1_excl_pe": m.macro_f1_excl_pe,
            "weighted_f1_excl_pe": m.weighted_f1_excl_pe,
            "cohen_kappa_excl_pe": m.cohen_kappa_excl_pe,
            "accuracy_incl_pe": m.accuracy_incl_pe,
            "macro_f1_incl_pe": m.macro_f1_incl_pe,
            "weighted_f1_incl_pe": m.weighted_f1_incl_pe,
            "cohen_kappa_incl_pe": m.cohen_kappa_incl_pe,
        }
        for m in metrics_list
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_per_class_overall(out_path: Path, gt: list[str], pred: list[str]) -> pd.DataFrame:
    report = per_class_report(gt, pred)
    rows = []
    for label in LABELS:
        m = report.get(label, {})
        rows.append(
            {
                "class": label,
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1-score", 0.0),
                "support": int(m.get("support", 0)),
            }
        )
    for agg in ("macro avg", "weighted avg"):
        m = report.get(agg, {})
        rows.append(
            {
                "class": agg,
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1-score", 0.0),
                "support": int(m.get("support", 0)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def write_confusion(
    csv_path: Path,
    png_path: Path,
    gt: list[str],
    pred: list[str],
    title: str,
    normalize: str | None = None,
) -> np.ndarray:
    cm = confusion(gt, pred, normalize=normalize)
    df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    df.to_csv(csv_path, float_format="%.4f" if normalize else "%g")

    fig, ax = plt.subplots(figsize=(6, 5))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        df,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        ax=ax,
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )
    ax.set_xlabel("VLM prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return cm


def write_summary_markdown(
    out_path: Path,
    manifest: list[ManifestEntry],
    metrics_list: list[SubjectMetrics],
    overall_per_class: pd.DataFrame,
    overall_metrics: SubjectMetrics,
    cm_overall: np.ndarray,
    cm_overall_norm: np.ndarray,
) -> None:
    lines: list[str] = []
    lines.append("# 注視対象 VLM アノテーション評価レポート\n")
    lines.append("## マニフェスト\n")
    lines.append("| subject_id | log_path | gt_path | offset_sec |")
    lines.append("|---|---|---|---:|")
    for e in manifest:
        lines.append(
            f"| {e.subject_id} | `{e.log_path}` | `{e.gt_path}` | {e.offset_sec:+.3f} |"
        )
    lines.append("\nオフセット定義: `aligned_gt_time = gt_time + offset_sec`\n")

    lines.append("## 被験者ごとの集計\n")
    lines.append("| subject | n_total | n_pe | n_used | coverage | acc(excl) | macroF1(excl) | wF1(excl) | kappa(excl) | acc(incl) | kappa(incl) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics_list:
        lines.append(
            f"| {m.subject_id} | {m.n_total} | {m.n_parse_error} | {m.n_used_excluding_pe} "
            f"| {m.coverage_rate:.3f} | {m.accuracy_excl_pe:.3f} | {m.macro_f1_excl_pe:.3f} "
            f"| {m.weighted_f1_excl_pe:.3f} | {m.cohen_kappa_excl_pe:.3f} "
            f"| {m.accuracy_incl_pe:.3f} | {m.cohen_kappa_incl_pe:.3f} |"
        )
    lines.append("")
    lines.append("- excl = parse_error 除外集計, incl = parse_error 含む集計")
    lines.append("- coverage = VLM サンプル時刻のうち有効ラベル区間 (Task/Paper/Dictionary/Memo) に入った割合\n")

    lines.append("## 全被験者結合 (parse_error 除外)\n")
    lines.append(
        f"- n_used = {overall_metrics.n_used_excluding_pe} / n_total = {overall_metrics.n_total} "
        f"(parse_error = {overall_metrics.n_parse_error})"
    )
    lines.append(f"- Accuracy: **{overall_metrics.accuracy_excl_pe:.3f}**")
    lines.append(f"- Macro F1: **{overall_metrics.macro_f1_excl_pe:.3f}**")
    lines.append(f"- Weighted F1: **{overall_metrics.weighted_f1_excl_pe:.3f}**")
    lines.append(f"- Cohen's kappa: **{overall_metrics.cohen_kappa_excl_pe:.3f}**\n")

    lines.append("### Per-class P/R/F1\n")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in overall_per_class.iterrows():
        lines.append(
            f"| {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['f1']:.3f} | {int(row['support'])} |"
        )
    lines.append("")

    lines.append("### Confusion matrix (raw counts)\n")
    lines.append(_md_matrix(cm_overall, fmt="d"))
    lines.append("\n### Confusion matrix (row-normalized)\n")
    lines.append(_md_matrix(cm_overall_norm, fmt=".3f"))
    lines.append("")

    lines.append("## 全被験者結合 (parse_error 含む)\n")
    lines.append(f"- Accuracy: **{overall_metrics.accuracy_incl_pe:.3f}**")
    lines.append(f"- Macro F1: **{overall_metrics.macro_f1_incl_pe:.3f}**")
    lines.append(f"- Weighted F1: **{overall_metrics.weighted_f1_incl_pe:.3f}**")
    lines.append(f"- Cohen's kappa: **{overall_metrics.cohen_kappa_incl_pe:.3f}**\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _md_matrix(cm: np.ndarray, fmt: str) -> str:
    header = "| GT \\\\ pred | " + " | ".join(LABELS) + " |"
    sep = "|---" + "|---:" * len(LABELS) + "|"
    lines = [header, sep]
    for i, label in enumerate(LABELS):
        cells = " | ".join(f"{cm[i, j]:{fmt}}" for j in range(len(LABELS)))
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def aggregate_pairs(alignments: list[tuple[str, AlignmentResult]]):
    """全被験者を結合した (gt_excl, pred_excl, gt_incl, pred_incl) を返す。"""
    gt_e, pred_e, gt_i, pred_i = [], [], [], []
    for _, alignment in alignments:
        a, b, c, d = split_pairs(alignment.rows)
        gt_e.extend(a)
        pred_e.extend(b)
        gt_i.extend(c)
        pred_i.extend(d)
    return gt_e, pred_e, gt_i, pred_i


def aggregate_metrics(alignments: list[tuple[str, AlignmentResult]]) -> SubjectMetrics:
    """全被験者を結合した SubjectMetrics (subject_id="overall") を返す。"""
    from .metrics import compute_subject_metrics
    from .align import AlignmentResult as _AR

    all_rows = []
    parse_error = 0
    in_valid = 0
    for _, a in alignments:
        all_rows.extend(a.rows)
        parse_error += a.parse_error_count
        in_valid += sum(1 for r in a.rows if r.in_labeled_interval)
    coverage = (in_valid / len(all_rows)) if all_rows else 0.0
    merged = _AR(rows=all_rows, coverage_rate=coverage, parse_error_count=parse_error)
    return compute_subject_metrics("overall", merged)
