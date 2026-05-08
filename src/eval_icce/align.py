"""オフセット適用と VLM サンプル時刻 ↔ GT 区間の突き合わせ。"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from .io import GtInterval, VlmSample


@dataclass(frozen=True)
class AlignedRow:
    timestamp: float          # VLM サンプル時刻 (秒)
    gt_label: str             # CLASS_LABELS のいずれか
    vlm_label: str            # CLASS_LABELS のいずれか or "parse_error"
    vlm_raw_label: str | None
    in_labeled_interval: bool  # VLM 時刻が "有効ラベル区間" に入ったか


@dataclass(frozen=True)
class AlignmentResult:
    rows: list[AlignedRow]
    coverage_rate: float       # 有効ラベル区間に入った VLM サンプルの割合
    parse_error_count: int     # VLM parse_error 件数


# 有効 GT 区間とみなすクラス (Step 2 で coverage 計算に使用)。
# x/unknown/空 由来の "Others" は coverage に含めない。
_VALID_GT_CLASSES: frozenset[str] = frozenset({"Task", "Paper", "Dictionary", "Memo"})


def apply_offset(intervals: list[GtInterval], offset_sec: float) -> list[GtInterval]:
    """GT 区間の start/end に offset_sec を加算する。

    オフセット解釈: aligned_gt_time = gt_time + offset_sec
    つまり offset_sec > 0 のとき GT 区間は時間軸の正方向 (後ろ) にずれる。
    その後 VLM 側のサンプル時刻 t と aligned 区間を直接突き合わせる。
    """
    if offset_sec == 0.0:
        return list(intervals)
    return [
        GtInterval(
            start_sec=iv.start_sec + offset_sec,
            end_sec=iv.end_sec + offset_sec,
            raw_label=iv.raw_label,
            normalized_label=iv.normalized_label,
        )
        for iv in intervals
    ]


def _lookup(intervals: list[GtInterval], starts: list[float], t: float) -> GtInterval | None:
    """t を含む区間 [start, end) を bisect で検索。重なりは start 昇順の最初に該当するもの。"""
    idx = bisect.bisect_right(starts, t) - 1
    while idx >= 0:
        iv = intervals[idx]
        if iv.start_sec <= t < iv.end_sec:
            return iv
        # GT 区間は通常重ならないが、念のため 1 つ前まで戻る
        if iv.end_sec <= t:
            return None
        idx -= 1
    return None


def build_alignment(
    vlm_samples: list[VlmSample],
    gt_intervals: list[GtInterval],
    offset_sec: float,
) -> AlignmentResult:
    """VLM サンプルと GT 区間を突き合わせて AlignedRow のリストを作る。

    GT に該当区間がない / 該当区間が x/unknown/空 → gt_label = "Others"
    """
    aligned_intervals = apply_offset(gt_intervals, offset_sec)
    starts = [iv.start_sec for iv in aligned_intervals]

    rows: list[AlignedRow] = []
    in_valid_count = 0
    parse_error = 0
    for s in vlm_samples:
        iv = _lookup(aligned_intervals, starts, s.time_sec)
        if iv is None:
            gt_label = "Others"
            in_valid = False
        else:
            gt_label = iv.normalized_label
            in_valid = iv.normalized_label in _VALID_GT_CLASSES
        if in_valid:
            in_valid_count += 1
        if s.normalized_label == "parse_error":
            parse_error += 1
        rows.append(
            AlignedRow(
                timestamp=s.time_sec,
                gt_label=gt_label,
                vlm_label=s.normalized_label,
                vlm_raw_label=s.raw_label,
                in_labeled_interval=in_valid,
            )
        )

    coverage = (in_valid_count / len(rows)) if rows else 0.0
    return AlignmentResult(rows=rows, coverage_rate=coverage, parse_error_count=parse_error)
