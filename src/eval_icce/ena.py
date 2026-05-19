"""ENA (Epistemic Network Analysis) — 注視対象コードの共起ネットワーク分析。

VLM が推定したフレームごとの注視対象ラベル系列に対し、moving stanza window で
共起をカウントし、ノード=ラベル, エッジ=共起回数のネットワークを描画する。

Usage:
    # 単一ログ (input/world_results.json を ENA にかける)
    uv run python -m src.eval_icce.ena \\
        --input input/world_results.json \\
        --output out/ena/ \\
        --window 5

    # 時間方向に 4 等分してセグメント別ネットワークも出す
    uv run python -m src.eval_icce.ena \\
        --input input/world_results.json \\
        --output out/ena/ \\
        --window 5 --segments 4

    # マニフェスト CSV から被験者ごと + overall を一括出力
    uv run python -m src.eval_icce.ena \\
        --manifest src/eval_icce/data.csv \\
        --output out/ena/ --window 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import CLASS_LABELS, PARSE_ERROR, VlmSample, load_manifest, load_vlm_log

logger = logging.getLogger("eval_icce.ena")

# ENA で扱うコード集合 (parse_error はコード化対象外)。
CODES: tuple[str, ...] = CLASS_LABELS  # ("Task", "Paper", "Dictionary", "Memo", "Others")
_CODE_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CODES)}


@dataclass(frozen=True)
class EnaResult:
    unit_id: str
    adjacency: np.ndarray   # shape (n_codes, n_codes), 対称, 対角 0
    code_counts: np.ndarray  # shape (n_codes,) 出現回数 (parse_error 除外後)
    n_samples_total: int     # 元の VLM サンプル数
    n_parse_error: int
    n_samples_used: int      # parse_error 除外後の系列長


# ---------------- core computation ----------------


def _codes_from_samples(samples: list[VlmSample]) -> list[str]:
    """parse_error を除外し、5 クラスのみのコード列に変換する。"""
    return [s.normalized_label for s in samples if s.normalized_label != PARSE_ERROR]


def compute_adjacency(codes: list[str], window: int) -> np.ndarray:
    """Moving stanza window でコード共起を集計する。

    各位置 i (現在行) について、過去 [i-window+1, i-1] の各コードと現在コードの
    ペアを 1 つの共起としてカウントする。同一コードの自己ループは含めない。
    対称行列で返す (adj[a,b] == adj[b,a])。
    """
    n = len(CODES)
    adj = np.zeros((n, n), dtype=float)
    if window < 2 or not codes:
        return adj
    for i, c_i in enumerate(codes):
        a = _CODE_TO_IDX.get(c_i)
        if a is None:
            continue
        start = max(0, i - window + 1)
        for j in range(start, i):
            c_j = codes[j]
            b = _CODE_TO_IDX.get(c_j)
            if b is None or b == a:
                continue
            adj[a, b] += 1
            adj[b, a] += 1
    return adj


def compute_code_counts(codes: list[str]) -> np.ndarray:
    """各コードの出現回数 (parse_error 除外後)。"""
    counts = np.zeros(len(CODES), dtype=float)
    for c in codes:
        idx = _CODE_TO_IDX.get(c)
        if idx is not None:
            counts[idx] += 1
    return counts


def build_ena_result(unit_id: str, samples: list[VlmSample], window: int) -> EnaResult:
    codes = _codes_from_samples(samples)
    adj = compute_adjacency(codes, window)
    counts = compute_code_counts(codes)
    n_pe = sum(1 for s in samples if s.normalized_label == PARSE_ERROR)
    return EnaResult(
        unit_id=unit_id,
        adjacency=adj,
        code_counts=counts,
        n_samples_total=len(samples),
        n_parse_error=n_pe,
        n_samples_used=len(codes),
    )


def split_into_segments(samples: list[VlmSample], n: int) -> list[list[VlmSample]]:
    """サンプル列を時間順に n 等分する (端数は末尾に寄せる)。"""
    if n <= 1 or not samples:
        return [samples]
    size = len(samples) // n
    if size == 0:
        return [samples]
    chunks: list[list[VlmSample]] = []
    for k in range(n):
        start = k * size
        end = (k + 1) * size if k < n - 1 else len(samples)
        chunks.append(samples[start:end])
    return chunks


# ---------------- IO ----------------


def _adj_dataframe(adj: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(adj, index=list(CODES), columns=list(CODES))


def write_adjacency(out_path: Path, adj: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _adj_dataframe(adj).to_csv(out_path, float_format="%g")


def write_code_counts(out_path: Path, counts: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"code": list(CODES), "count": counts.astype(int)})
    df.to_csv(out_path, index=False)


# ---------------- visualization ----------------


def plot_network(
    out_path: Path,
    adj: np.ndarray,
    code_counts: np.ndarray,
    title: str,
    *,
    edge_max: float | None = None,
    count_max: float | None = None,
) -> None:
    """5 ノードを円周上に配置し、共起をエッジ太さで可視化する。

    edge_max / count_max を与えると、複数プロットで太さ・サイズスケールを揃えられる。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(CODES)

    # 上が Task になるように π/2 から時計回り (見やすさのため)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")

    e_max = edge_max if edge_max and edge_max > 0 else (adj.max() if adj.size else 0)
    if not e_max:
        e_max = 1.0
    edge_scale = 9.0 / e_max  # 1 ペアの最大本数 → 線幅 9pt

    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w <= 0:
                continue
            ax.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                color="steelblue",
                linewidth=w * edge_scale,
                alpha=0.55,
                solid_capstyle="round",
                zorder=1,
            )
            mx, my = (xs[i] + xs[j]) / 2.0, (ys[i] + ys[j]) / 2.0
            ax.text(
                mx, my, f"{int(w)}",
                fontsize=9, color="dimgray",
                ha="center", va="center", zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )

    c_max = count_max if count_max and count_max > 0 else (code_counts.max() if code_counts.size else 0)
    if not c_max:
        c_max = 1.0
    node_sizes = 300 + (code_counts / c_max) * 1600

    ax.scatter(
        xs, ys, s=node_sizes,
        color="tomato", edgecolors="black", linewidths=1.2, zorder=3,
    )
    for i, label in enumerate(CODES):
        ax.text(
            xs[i] * 1.22, ys[i] * 1.22, label,
            ha="center", va="center", fontsize=12, fontweight="bold",
        )
        ax.text(
            xs[i] * 1.22, ys[i] * 1.22 - 0.08,
            f"n={int(code_counts[i])}",
            ha="center", va="center", fontsize=9, color="dimgray",
        )

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------- summary ----------------


def write_summary(
    out_path: Path,
    results: list[EnaResult],
    window: int,
    segments: int,
    overall: EnaResult | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# ENA レポート (注視対象コード共起ネットワーク)\n")
    lines.append(f"- moving stanza window: **{window}** サンプル")
    lines.append(f"- セグメント分割数: **{segments}**")
    lines.append(f"- コード集合: {', '.join(CODES)}")
    lines.append("- parse_error はコード化対象外 (集計から除外)\n")

    lines.append("## ユニット別サンプル数\n")
    lines.append("| unit | n_total | n_parse_error | n_used |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.unit_id} | {r.n_samples_total} | {r.n_parse_error} | {r.n_samples_used} |"
        )
    if overall is not None:
        lines.append(
            f"| **overall** | {overall.n_samples_total} | {overall.n_parse_error} | {overall.n_samples_used} |"
        )
    lines.append("")

    for r in results:
        lines.append(f"## {r.unit_id} — 共起カウント\n")
        lines.append(_md_matrix(r.adjacency))
        lines.append("")
    if overall is not None:
        lines.append("## overall — 共起カウント\n")
        lines.append(_md_matrix(overall.adjacency))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _md_matrix(adj: np.ndarray) -> str:
    header = "| code | " + " | ".join(CODES) + " |"
    sep = "|---" + "|---:" * len(CODES) + "|"
    lines = [header, sep]
    for i, label in enumerate(CODES):
        cells = " | ".join(f"{int(adj[i, j])}" for j in range(len(CODES)))
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


# ---------------- CLI ----------------


def _run_unit(
    unit_id: str,
    samples: list[VlmSample],
    window: int,
    segments: int,
    unit_dir: Path,
) -> EnaResult:
    """1 ユニット (= 1 ログファイル) を処理し、結果ファイルを書き出す。"""
    result = build_ena_result(unit_id, samples, window)
    logger.info(
        "[%s] n_total=%d n_parse_error=%d n_used=%d",
        unit_id, result.n_samples_total, result.n_parse_error, result.n_samples_used,
    )

    write_adjacency(unit_dir / "adjacency.csv", result.adjacency)
    write_code_counts(unit_dir / "code_counts.csv", result.code_counts)
    plot_network(
        unit_dir / "network.png",
        result.adjacency,
        result.code_counts,
        title=f"ENA network — {unit_id} (window={window})",
    )

    if segments > 1:
        seg_dir = unit_dir / "segments"
        chunks = split_into_segments(samples, segments)
        seg_results = [
            build_ena_result(f"{unit_id}_seg{k+1}", ch, window)
            for k, ch in enumerate(chunks)
        ]
        edge_max = max((r.adjacency.max() for r in seg_results), default=0.0)
        count_max = max((r.code_counts.max() for r in seg_results), default=0.0)
        for k, r in enumerate(seg_results, start=1):
            write_adjacency(seg_dir / f"segment_{k}_adjacency.csv", r.adjacency)
            write_code_counts(seg_dir / f"segment_{k}_code_counts.csv", r.code_counts)
            plot_network(
                seg_dir / f"segment_{k}_network.png",
                r.adjacency,
                r.code_counts,
                title=f"{unit_id} segment {k}/{segments} (window={window})",
                edge_max=edge_max,
                count_max=count_max,
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
        "--window", type=int, default=5,
        help="moving stanza window のサンプル幅 (default: 5)",
    )
    parser.add_argument(
        "--segments", type=int, default=1,
        help="系列を N 等分して各セグメント別ネットワークも出す (default: 1)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.window < 2:
        logger.error("--window は 2 以上を指定してください")
        return 2
    if args.segments < 1:
        logger.error("--segments は 1 以上を指定してください")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    results: list[EnaResult] = []
    overall: EnaResult | None = None

    if args.input is not None:
        samples = load_vlm_log(args.input)
        unit_id = args.input.stem
        results.append(_run_unit(unit_id, samples, args.window, args.segments, args.output))
    else:
        manifest = load_manifest(args.manifest)
        if not manifest:
            logger.error("マニフェストが空です: %s", args.manifest)
            return 2
        logger.info("被験者数: %d", len(manifest))
        per_subject_dir = args.output / "per_subject"
        all_samples: list[VlmSample] = []
        for entry in manifest:
            samples = load_vlm_log(entry.log_path)
            all_samples.extend(samples)
            results.append(
                _run_unit(
                    entry.subject_id, samples, args.window, args.segments,
                    per_subject_dir / entry.subject_id,
                )
            )
        overall = _run_unit(
            "overall", all_samples, args.window, args.segments, args.output / "overall"
        )

    write_summary(
        args.output / "summary.md", results, args.window, args.segments, overall=overall,
    )
    logger.info("完了: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
