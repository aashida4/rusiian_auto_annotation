"""マニフェスト / VLM ログ / GT アノテーションのロード。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

CLASS_LABELS: tuple[str, ...] = ("Task", "Paper", "Dictionary", "Memo", "Others")
PARSE_ERROR = "parse_error"
GT_NULL_LABELS: frozenset[str] = frozenset({"x", "unknown", ""})


@dataclass(frozen=True)
class ManifestEntry:
    subject_id: str
    log_path: Path
    gt_path: Path
    offset_sec: float


@dataclass(frozen=True)
class VlmSample:
    time_sec: float
    raw_label: str | None
    normalized_label: str  # CLASS_LABELS のいずれか or PARSE_ERROR


@dataclass(frozen=True)
class GtInterval:
    start_sec: float
    end_sec: float
    raw_label: str
    normalized_label: str  # CLASS_LABELS のいずれか (x/unknown/空 → "Others")


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """マニフェスト CSV を読み込む。

    log_path / gt_path はマニフェスト CSV があるディレクトリからの相対パスとして
    解決する (絶対パスはそのまま使う)。
    """
    manifest_path = manifest_path.resolve()
    base_dir = manifest_path.parent
    entries: list[ManifestEntry] = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"subject_id", "log_path", "gt_path", "offset_sec"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest に必須列がありません: {sorted(missing)}")
        for row in reader:
            log_path = _resolve(base_dir, row["log_path"])
            gt_path = _resolve(base_dir, row["gt_path"])
            entries.append(
                ManifestEntry(
                    subject_id=str(row["subject_id"]).strip(),
                    log_path=log_path,
                    gt_path=gt_path,
                    offset_sec=float(row["offset_sec"]),
                )
            )
    return entries


def _resolve(base_dir: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_vlm_log(log_path: Path) -> list[VlmSample]:
    """VLM 出力 JSON を読み込み、5 クラスに正規化する。

    複合ラベル (例: "Paper, Task")、未知ラベル、null は parse_error 扱い。
    """
    with log_path.open(encoding="utf-8") as f:
        data = json.load(f)
    samples: list[VlmSample] = []
    for entry in data:
        time_sec = float(entry["time"])
        raw = entry.get("prediction")
        normalized = _normalize_vlm_label(raw)
        samples.append(VlmSample(time_sec=time_sec, raw_label=raw, normalized_label=normalized))
    return samples


def _normalize_vlm_label(raw: object) -> str:
    if not isinstance(raw, str):
        return PARSE_ERROR
    cleaned = raw.strip()
    if cleaned in CLASS_LABELS:
        return cleaned
    return PARSE_ERROR


def load_gt_intervals(gt_path: Path) -> list[GtInterval]:
    """ELAN エクスポート風 TSV を読み込む。

    列構造: tier_name \\t (空) \\t start[HH:MM:SS.fff] \\t end[HH:MM:SS.fff]
            \\t duration[HH:MM:SS.fff] \\t label

    label 正規化:
        - strip() 適用 (例: "Paper " → "Paper")
        - "x" / "unknown" / 空文字は "Others" に統合
        - 既知 4 クラス (Task/Paper/Dictionary/Memo) はそのまま
        - それ以外 → "Others" (想定外)
    """
    intervals: list[GtInterval] = []
    with gt_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            try:
                start = _parse_hms(cols[2])
                end = _parse_hms(cols[3])
            except (ValueError, IndexError):
                continue
            raw_label = cols[5].strip()
            normalized = _normalize_gt_label(raw_label)
            intervals.append(
                GtInterval(
                    start_sec=start,
                    end_sec=end,
                    raw_label=raw_label,
                    normalized_label=normalized,
                )
            )
    intervals.sort(key=lambda iv: iv.start_sec)
    return intervals


def _normalize_gt_label(raw: str) -> str:
    if raw in GT_NULL_LABELS:
        return "Others"
    if raw in CLASS_LABELS:
        return raw
    return "Others"


def _parse_hms(s: str) -> float:
    """HH:MM:SS.fff (または MM:SS.fff / SS.fff) を秒に変換する。"""
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    return float(s)
