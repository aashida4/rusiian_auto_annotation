"""分類性能と Cohen's kappa の計算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from .io import CLASS_LABELS, PARSE_ERROR

# sklearn に渡すラベル順 (5 クラス固定)
LABELS: list[str] = list(CLASS_LABELS)


@dataclass(frozen=True)
class SubjectMetrics:
    subject_id: str
    n_total: int                 # 元の VLM サンプル数
    n_parse_error: int
    n_used_excluding_pe: int     # parse_error を除いた評価対象数
    coverage_rate: float         # GT 区間カバー率 (有効ラベル区間に入った割合)
    accuracy_excl_pe: float
    macro_f1_excl_pe: float
    weighted_f1_excl_pe: float
    cohen_kappa_excl_pe: float
    accuracy_incl_pe: float      # parse_error も "誤り" として含めた集計
    macro_f1_incl_pe: float
    weighted_f1_incl_pe: float
    cohen_kappa_incl_pe: float


def _split_pairs(rows) -> tuple[list[str], list[str], list[str], list[str]]:
    """AlignedRow のリストから (gt_excl, pred_excl, gt_incl, pred_incl) を返す。

    excl: parse_error を完全に除外
    incl: parse_error をそのままラベルとして残す (accuracy 等は誤りに寄与)
    """
    gt_excl, pred_excl, gt_incl, pred_incl = [], [], [], []
    for r in rows:
        gt_incl.append(r.gt_label)
        pred_incl.append(r.vlm_label)
        if r.vlm_label != PARSE_ERROR:
            gt_excl.append(r.gt_label)
            pred_excl.append(r.vlm_label)
    return gt_excl, pred_excl, gt_incl, pred_incl


def compute_subject_metrics(subject_id: str, alignment) -> SubjectMetrics:
    rows = alignment.rows
    gt_e, pred_e, gt_i, pred_i = _split_pairs(rows)
    n_total = len(rows)
    n_pe = alignment.parse_error_count
    n_used = len(gt_e)

    if n_used > 0:
        acc_e = accuracy_score(gt_e, pred_e)
        macro_e = f1_score(gt_e, pred_e, labels=LABELS, average="macro", zero_division=0)
        weighted_e = f1_score(gt_e, pred_e, labels=LABELS, average="weighted", zero_division=0)
        kappa_e = cohen_kappa_score(gt_e, pred_e, labels=LABELS)
    else:
        acc_e = macro_e = weighted_e = kappa_e = float("nan")

    if n_total > 0:
        acc_i = accuracy_score(gt_i, pred_i)
        labels_incl = LABELS + [PARSE_ERROR]
        macro_i = f1_score(gt_i, pred_i, labels=labels_incl, average="macro", zero_division=0)
        weighted_i = f1_score(gt_i, pred_i, labels=labels_incl, average="weighted", zero_division=0)
        kappa_i = cohen_kappa_score(gt_i, pred_i, labels=labels_incl)
    else:
        acc_i = macro_i = weighted_i = kappa_i = float("nan")

    return SubjectMetrics(
        subject_id=subject_id,
        n_total=n_total,
        n_parse_error=n_pe,
        n_used_excluding_pe=n_used,
        coverage_rate=alignment.coverage_rate,
        accuracy_excl_pe=acc_e,
        macro_f1_excl_pe=macro_e,
        weighted_f1_excl_pe=weighted_e,
        cohen_kappa_excl_pe=kappa_e,
        accuracy_incl_pe=acc_i,
        macro_f1_incl_pe=macro_i,
        weighted_f1_incl_pe=weighted_i,
        cohen_kappa_incl_pe=kappa_i,
    )


def per_class_report(gt: list[str], pred: list[str]) -> dict:
    """sklearn classification_report の dict 形式を返す (5 クラス固定)。"""
    return classification_report(
        gt, pred, labels=LABELS, output_dict=True, zero_division=0
    )


def confusion(gt: list[str], pred: list[str], normalize: str | None = None) -> np.ndarray:
    """5x5 混同行列 (行=GT, 列=予測)。normalize="true" で行正規化。"""
    return confusion_matrix(gt, pred, labels=LABELS, normalize=normalize)


def split_pairs(rows):
    """report.py からも使うので公開する。"""
    return _split_pairs(rows)
