# eval_icce — 注視対象 VLM アノテーションの精度評価

学習者の一人称視点動画 + 視線情報から VLM (例: gemma3:4b) が推定した注視対象クラスと、人手で付与した GT (連続時間区間アノテーション) を突き合わせ、被験者ごと・全被験者結合の分類性能とアノテータ間一致度 (Cohen's kappa) をレポートする。論文 Results 節にそのまま貼れる粒度の `summary.md` と各種 CSV / 混同行列 PNG を出力する。

検出側 (`src/detection/main.py`) には依存せず、その出力 JSON を入力とするオフライン評価。

## クラス定義 (5 値分類)

| ラベル | 意味 |
|---|---|
| `Task` | ロシア語の問題が書かれた課題用紙 |
| `Paper` | 解答用紙 |
| `Dictionary` | 辞書 (紙または電子) |
| `Memo` | 白紙のメモ用紙 |
| `Others` | 上記以外 |

## マニフェスト CSV

被験者ごとに VLM ログと GT のペア・時刻オフセットを 1 行ずつ記述する。サンプル: `src/eval_icce/data.csv`

```csv
subject_id,log_path,gt_path,offset_sec
1,../../input/honban/0620/1/world_results.json,../../input/honban/0620/1/Task.txt,0
2,../../input/honban/0620/2/world_results.json,../../input/honban/0620/2/Task.txt,0
3,../../input/honban/0620/3/world_results.json,../../input/honban/0620/3/synchronized_video.txt,0
4,../../input/honban/0620/4/world_results.json,../../input/honban/0620/4/synchronized_video.txt,0
0711,../../input/honban/0711/world_results.json,../../input/honban/0711/Task.txt,0
```

| 列 | 説明 |
|---|---|
| `subject_id` | 被験者 ID。出力ディレクトリ・ファイル名にこの値が使われる |
| `log_path` | VLM 出力 JSON のパス。**マニフェスト CSV があるディレクトリからの相対パス**として解決される (絶対パスはそのまま) |
| `gt_path` | GT アノテーション TSV のパス。同上 |
| `offset_sec` | GT 時刻に加算する秒数 (小数可)。**意味**: `aligned_gt_time = gt_time + offset_sec` として揃えてから VLM サンプル時刻と突き合わせる。GT が VLM より遅れているなら正、早ければ負 |

### 入力ファイルの形式

- **VLM ログ (`world_results.json`)**: `[{time: 秒, frame: int, prediction: ラベル文字列, reasoning: ...}, ...]`。`prediction` が 5 クラス以外 (複合ラベル `"Paper, Task"` 等を含む) や null の場合は `parse_error` 扱いで集計から除外（含めた集計も別途出る）
- **GT (`Task.txt` / `synchronized_video.txt` 等)**: ELAN エクスポート風 TSV。列: `tier_name \t (空) \t start[HH:MM:SS.fff] \t end[HH:MM:SS.fff] \t duration[HH:MM:SS.fff] \t label`。`label` が `x` / `unknown` / 空 の区間は **GT 上で `Others` として明示注釈された区間**として扱う (どのGT区間にも入らないVLMサンプルも同じく `Others`)

## 実行

```bash
# サンプルマニフェストで動かす
uv run python -m src.eval_icce.evaluate \
  --manifest src/eval_icce/data.csv \
  --output out/eval_icce/

# 別マニフェストを使う
uv run python -m src.eval_icce.evaluate \
  --manifest /path/to/your_manifest.csv \
  --output /path/to/output_dir/

# カバー率の警告閾値を変更 (デフォルト 0.5)
uv run python -m src.eval_icce.evaluate \
  --manifest src/eval_icce/data.csv \
  --output out/ \
  --coverage-warn 0.7
```

| 引数 | 説明 |
|---|---|
| `--manifest` | マニフェスト CSV のパス (必須) |
| `--output` | 出力ディレクトリ。なければ作成 (必須) |
| `--coverage-warn` | カバー率がこの値未満の被験者は警告ログを出す (default: 0.5) |

## 出力

```
<output>/
├── aligned_pairs/
│   └── {subject_id}.csv               … オフセット適用・グリッド整列済みの (timestamp, gt_label, vlm_label, vlm_raw_label, in_labeled_interval)
├── per_subject_metrics.csv            … 被験者ごとの accuracy / macro-F1 / weighted-F1 / Cohen's kappa / coverage_rate (excl/incl 両方)
├── per_class_metrics_overall.csv      … 全被験者結合での per-class P/R/F1/support
├── confusion_matrix_overall.csv       … 全体混同行列 (生件数, 行=GT 列=VLM)
├── confusion_matrix_overall.png       … 同 (seaborn heatmap)
├── confusion_matrix_overall_normalized.csv / .png  … 行正規化版
├── per_subject_confusion/
│   └── {subject_id}.csv / .png / _normalized.csv / _normalized.png
└── summary.md                         … 全結果を Markdown 表でまとめたレポート
```

### 集計の 2 系統 (excl / incl)

| 系統 | 内容 |
|---|---|
| `excl_pe` | VLM の `parse_error` サンプルを完全に除外して算出。論文の本数値はこちらを使う想定 |
| `incl_pe` | `parse_error` を 1 つの予測ラベルとして残して算出。除外による母数変動と乖離を確認する用 |

`per_subject_metrics.csv` には両系統の accuracy / macro-F1 / weighted-F1 / Cohen's kappa が並ぶ。

### カバー率 (`coverage_rate`)

VLM サンプル時刻のうち、有効ラベル区間 (`Task` / `Paper` / `Dictionary` / `Memo`) に入った割合。`x`/`unknown`/空 由来の `Others` 区間に入ったサンプル、およびどの区間にも入らなかったサンプルはカバー外として数える。**マニフェストのオフセット値が大きく狂っているとここが急落**するので、`--coverage-warn` 閾値未満で警告が出る。

## 評価の仕様詳細

- 5 クラスは `["Task", "Paper", "Dictionary", "Memo", "Others"]` に固定し、`sklearn` の各メトリクスに `labels=` 明示で渡す。`zero_division=0`
- Cohen's kappa は `cohen_kappa_score(labels=...)` を明示
- 区間照合は半開区間 `[start, end)`、`bisect` で O(log N)
- VLM 5 クラスに完全一致しないラベルは `parse_error` (例: `"Paper, Task"`, null, 末尾空白付き未知ラベル)。VLM ラベル側は前後空白だけ `strip()` で許容
- GT ラベルは `strip()` 後に `x`/`unknown`/空 → `Others`、5 クラスのいずれか → そのまま、それ以外 → `Others`
- 時刻はすべて秒の浮動小数に統一 (`HH:MM:SS.fff`, `MM:SS.fff`, `SS.fff` をサポート)
- オフセットは加算 (`aligned_gt_time = gt_time + offset_sec`)。`align.py:apply_offset` のコメントにも明記
- 乱数なし

## モジュール構成

```
src/eval_icce/
├── __init__.py
├── evaluate.py     … CLI エントリポイント
├── io.py           … マニフェスト / VLM ログ / GT のロードとラベル正規化
├── align.py        … オフセット適用・bisect でのサンプル ↔ 区間突き合わせ
├── metrics.py      … sklearn ベースの分類性能 / Cohen's kappa
├── report.py       … CSV / PNG / Markdown レポートの書き出し
├── data.csv        … スキーマ確認用サンプルマニフェスト (5 被験者分)
└── README.md       … このファイル
```

## 依存

リポジトリ全体の `pyproject.toml` に追加済み:

- `pandas` (>=3.0)
- `scikit-learn` (>=1.8)
- `matplotlib` (>=3.10)
- `seaborn` (>=0.13)

`uv sync` で揃う。
