"""注視対象コードの時間ビン積み上げ棒グラフ。

VLM 出力 JSON のサンプル列を一定時間幅のビンに区切り、各ビン内のコード割合
(Task/Paper/Dictionary/Memo/Others, optionally parse_error) を積み上げ棒で
可視化する。横軸=時間, 縦軸=各コードの割合 (または件数)。

Usage:
    # input/world_results.json を 60 秒ビンで積み上げ
    uv run python -m src.eval_icce.stacked_bar \\
        --input input/world_results.json \\
        --output out/stacked_bar/ \\
        --bin 60

    # parse_error も別カテゴリとして残す
    uv run python -m src.eval_icce.stacked_bar \\
        --input input/world_results.json \\
        --output out/stacked_bar/ \\
        --bin 60 --include-parse-error

    # マニフェスト (被験者ごとに出力)
    uv run python -m src.eval_icce.stacked_bar \\
        --manifest src/eval_icce/data.csv \\
        --output out/stacked_bar/ --bin 60
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import CLASS_LABELS, PARSE_ERROR, VlmSample, load_manifest, load_vlm_log

logger = logging.getLogger("eval_icce.stacked_bar")


# 積み上げ順 (下から)。Others を一番上に置いて、Task/Paper/Dictionary/Memo を視認しやすく
PLOT_LABELS: tuple[str, ...] = CLASS_LABELS  # ("Task", "Paper", "Dictionary", "Memo", "Others")

COLOR_MAP: dict[str, str] = {
    "Task": "#1f77b4",
    "Paper": "#2ca02c",
    "Dictionary": "#ff7f0e",
    "Memo": "#9467bd",
    "Others": "#7f7f7f",
    PARSE_ERROR: "#d62728",
}


@dataclass(frozen=True)
class BinResult:
    unit_id: str
    bin_width_sec: float
    bin_edges: np.ndarray     # shape (n_bins + 1,)
    bin_centers: np.ndarray   # shape (n_bins,)
    labels: tuple[str, ...]   # 列の意味 (積み上げ順)
    counts: np.ndarray        # shape (n_bins, n_labels)
    ratios: np.ndarray        # shape (n_bins, n_labels) 行和=1 (空ビンは全 0)
    n_samples_total: int
    n_parse_error: int


# ---------------- core computation ----------------


def compute_bins(
    unit_id: str,
    samples: list[VlmSample],
    bin_width: float,
    *,
    include_parse_error: bool,
) -> BinResult:
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")

    labels = list(PLOT_LABELS)
    if include_parse_error:
        labels.append(PARSE_ERROR)
    label_to_col = {lab: i for i, lab in enumerate(labels)}

    if not samples:
        return BinResult(
            unit_id=unit_id,
            bin_width_sec=bin_width,
            bin_edges=np.array([0.0, bin_width]),
            bin_centers=np.array([bin_width / 2.0]),
            labels=tuple(labels),
            counts=np.zeros((1, len(labels))),
            ratios=np.zeros((1, len(labels))),
            n_samples_total=0,
            n_parse_error=0,
        )

    t_max = max(s.time_sec for s in samples)
    n_bins = int(math.floor(t_max / bin_width)) + 1
    edges = np.arange(n_bins + 1, dtype=float) * bin_width
    centers = (edges[:-1] + edges[1:]) / 2.0

    counts = np.zeros((n_bins, len(labels)), dtype=float)
    n_parse_error = 0
    for s in samples:
        if s.normalized_label == PARSE_ERROR:
            n_parse_error += 1
        col = label_to_col.get(s.normalized_label)
        if col is None:
            continue
        bin_idx = int(math.floor(s.time_sec / bin_width))
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        counts[bin_idx, col] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    ratios = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    return BinResult(
        unit_id=unit_id,
        bin_width_sec=bin_width,
        bin_edges=edges,
        bin_centers=centers,
        labels=tuple(labels),
        counts=counts,
        ratios=ratios,
        n_samples_total=len(samples),
        n_parse_error=n_parse_error,
    )


# ---------------- IO ----------------


def write_bin_csv(out_path: Path, result: BinResult, *, mode: str) -> None:
    """mode='ratio' なら割合, 'count' なら生件数で書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = result.ratios if mode == "ratio" else result.counts
    df = pd.DataFrame(data, columns=list(result.labels))
    df.insert(0, "bin_end_sec", result.bin_edges[1:])
    df.insert(0, "bin_start_sec", result.bin_edges[:-1])
    df.insert(0, "bin_index", np.arange(len(df)))
    fmt = "%.4f" if mode == "ratio" else "%g"
    df.to_csv(out_path, index=False, float_format=fmt)


# ---------------- visualization ----------------


def plot_stacked_bar(
    out_path: Path,
    result: BinResult,
    title: str,
    *,
    mode: str = "ratio",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if result.counts.shape[0] == 0:
        logger.warning("空のビン: %s", title)
        return

    data = result.ratios if mode == "ratio" else result.counts
    n_bins = data.shape[0]
    bar_width = result.bin_width_sec * 0.92

    fig_w = min(24.0, max(8.0, n_bins * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    bottom = np.zeros(n_bins)
    for j, lab in enumerate(result.labels):
        ax.bar(
            result.bin_centers,
            data[:, j],
            width=bar_width,
            bottom=bottom,
            label=lab,
            color=COLOR_MAP.get(lab),
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += data[:, j]

    ax.set_xlabel("Time (sec)")
    if mode == "ratio":
        ax.set_ylabel("Proportion")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.set_xlim(result.bin_edges[0], result.bin_edges[-1])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_line(
    out_path: Path,
    result: BinResult,
    title: str,
    *,
    mode: str = "ratio",
) -> None:
    """同じビン集計を、コードごとの折れ線として描く。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if result.counts.shape[0] == 0:
        logger.warning("空のビン: %s", title)
        return

    data = result.ratios if mode == "ratio" else result.counts
    n_bins = data.shape[0]

    fig_w = min(24.0, max(8.0, n_bins * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    for j, lab in enumerate(result.labels):
        ax.plot(
            result.bin_centers,
            data[:, j],
            label=lab,
            color=COLOR_MAP.get(lab),
            marker="o",
            markersize=4,
            linewidth=1.8,
        )

    ax.set_xlabel("Time (sec)")
    if mode == "ratio":
        ax.set_ylabel("Proportion")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.set_xlim(result.bin_edges[0], result.bin_edges[-1])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------- CLI ----------------


def _run_unit(
    unit_id: str,
    samples: list[VlmSample],
    bin_width: float,
    include_parse_error: bool,
    unit_dir: Path,
) -> BinResult:
    result = compute_bins(
        unit_id, samples, bin_width, include_parse_error=include_parse_error
    )
    logger.info(
        "[%s] n_total=%d n_parse_error=%d n_bins=%d",
        unit_id, result.n_samples_total, result.n_parse_error, len(result.bin_centers),
    )

    write_bin_csv(unit_dir / "bin_ratios.csv", result, mode="ratio")
    write_bin_csv(unit_dir / "bin_counts.csv", result, mode="count")
    plot_stacked_bar(
        unit_dir / "stacked_bar_ratio.png", result,
        title=f"{unit_id} — code ratio (bin={bin_width:g}s)",
        mode="ratio",
    )
    plot_stacked_bar(
        unit_dir / "stacked_bar_count.png", result,
        title=f"{unit_id} — code count (bin={bin_width:g}s)",
        mode="count",
    )
    plot_line(
        unit_dir / "line_ratio.png", result,
        title=f"{unit_id} — code ratio over time (bin={bin_width:g}s)",
        mode="ratio",
    )
    plot_line(
        unit_dir / "line_count.png", result,
        title=f"{unit_id} — code count over time (bin={bin_width:g}s)",
        mode="count",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="VLM 出力 JSON のパス (単一ファイル)")
    src.add_argument("--manifest", type=Path, help="マニフェスト CSV のパス (複数被験者)")
    parser.add_argument("--output", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument(
        "--bin", dest="bin_width", type=float, default=60.0,
        help="ビンの時間幅 (秒, default: 60)",
    )
    parser.add_argument(
        "--include-parse-error", action="store_true",
        help="parse_error も独立カテゴリとして積み上げる (default: 除外)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.bin_width <= 0:
        logger.error("--bin は 0 より大きい値を指定してください")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        samples = load_vlm_log(args.input)
        _run_unit(
            args.input.stem, samples, args.bin_width,
            args.include_parse_error, args.output,
        )
    else:
        manifest = load_manifest(args.manifest)
        if not manifest:
            logger.error("マニフェストが空です: %s", args.manifest)
            return 2
        logger.info("被験者数: %d", len(manifest))
        per_subject_dir = args.output / "per_subject"
        for entry in manifest:
            samples = load_vlm_log(entry.log_path)
            _run_unit(
                entry.subject_id, samples, args.bin_width,
                args.include_parse_error, per_subject_dir / entry.subject_id,
            )

    logger.info("完了: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
