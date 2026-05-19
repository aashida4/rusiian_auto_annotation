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

## ENA (Epistemic Network Analysis)

VLM が推定した注視対象コード列に対し、moving stanza window でコード間共起をカウントしてネットワークを描く。`ena.py` 単体で完結し、`evaluate.py` パイプラインからは独立。

```bash
# 単一ログ (input/world_results.json) を ENA にかける
uv run python -m src.eval_icce.ena \
  --input input/world_results.json \
  --output out/ena/ \
  --window 5

# 時間方向に 4 等分してセグメント別ネットワークも出す
uv run python -m src.eval_icce.ena \
  --input input/world_results.json \
  --output out/ena/ \
  --window 5 --segments 4

# マニフェスト CSV から被験者ごと + 全被験者結合 (overall) を一括出力
uv run python -m src.eval_icce.ena \
  --manifest src/eval_icce/data.csv \
  --output out/ena/ --window 5
```

| 引数 | 説明 |
|---|---|
| `--input` | VLM 出力 JSON のパス (単一ファイル)。`--manifest` と排他 |
| `--manifest` | マニフェスト CSV のパス (複数被験者)。`--input` と排他 |
| `--output` | 出力ディレクトリ (必須) |
| `--window` | moving stanza window のサンプル幅 (default: 5)。現在行から過去 `window-1` 行までを 1 スタンザとして共起カウント |
| `--segments` | 系列を時間順に N 等分してセグメント別ネットワークも出す (default: 1) |

### 仕様

- コード集合は 5 クラス固定 `["Task", "Paper", "Dictionary", "Memo", "Others"]`
- `parse_error` (`io.py` のラベル正規化で 5 クラスに該当しなかった VLM 出力) はコード化対象外として系列から除外
- 共起カウント: 現在行 i のコードと、過去 `[i-window+1, i-1]` 範囲の異なるコードのペアごとに対称行列に +1。自己ループ (同一コードの連続) は加算しない
- ネットワーク図: 5 ノードを円周上 (上=Task) に配置、エッジ太さ ∝ 共起回数、ノードサイズ ∝ コード出現回数。エッジ中央に共起回数の生値を表示
- セグメント分割時は、セグメント間でエッジ太さ・ノードサイズのスケールを揃える (相対比較できるように)
- 乱数なし

### 出力

```
<output>/
├── adjacency.csv          … 5x5 共起カウント (対称, 対角 0)
├── code_counts.csv        … 各コードの出現回数 (parse_error 除外後)
├── network.png            … ネットワーク可視化
├── segments/              … --segments > 1 のとき
│   ├── segment_{k}_adjacency.csv
│   ├── segment_{k}_code_counts.csv
│   └── segment_{k}_network.png
└── summary.md             … パラメータ・サンプル数・共起行列を Markdown 表で
```

`--manifest` 指定時はさらに `per_subject/{subject_id}/` と `overall/` 配下に同じ構造で出力する。

## 時間ビン積み上げ棒グラフ

VLM 出力 JSON のサンプル列を一定時間幅のビンに区切り、各ビン内のコード割合 (および件数) を積み上げ棒で可視化する。`stacked_bar.py` 単体で完結。

```bash
# input/world_results.json を 120 秒ビンで可視化
uv run python -m src.eval_icce.stacked_bar \
  --input input/world_results.json \
  --output out/stacked_bar/ \
  --bin 120

# parse_error も独立カテゴリとして積み上げる
uv run python -m src.eval_icce.stacked_bar \
  --input input/world_results.json \
  --output out/stacked_bar/ \
  --bin 60 --include-parse-error

# マニフェスト (被験者ごと)
uv run python -m src.eval_icce.stacked_bar \
  --manifest src/eval_icce/data.csv \
  --output out/stacked_bar/ --bin 60
```

| 引数 | 説明 |
|---|---|
| `--input` | VLM 出力 JSON のパス (単一ファイル)。`--manifest` と排他 |
| `--manifest` | マニフェスト CSV のパス (複数被験者)。`--input` と排他 |
| `--output` | 出力ディレクトリ (必須) |
| `--bin` | ビンの時間幅 (秒, default: 60) |
| `--include-parse-error` | parse_error も独立カテゴリとして積み上げる (default: 除外) |

### 仕様

- 5 クラス固定 `["Task", "Paper", "Dictionary", "Memo", "Others"]` (積み上げ順は左の凡例どおり)。`--include-parse-error` 指定時は末尾に `parse_error` (赤系) を追加
- ビン分割: サンプル `time` (秒) を `bin_width` で割って `floor` した値をビン番号にする。ビン境界は `[0, w, 2w, …]`、最後のビンは最大時刻を含むよう拡張
- 割合は各ビンの行和=1 になるよう正規化 (空ビンは全 0)。件数の生値も別途出力
- 横軸はビン中心の秒数、棒幅は `bin_width × 0.92` (棒どうしを少し離す)

### 出力

```
<output>/
├── bin_ratios.csv         … bin_index, bin_start_sec, bin_end_sec, <labels...> (割合)
├── bin_counts.csv         … 同レイアウトの生件数
├── stacked_bar_ratio.png  … 積み上げ棒 (割合, y∈[0,1])
├── stacked_bar_count.png  … 積み上げ棒 (件数)
├── line_ratio.png         … コードごとの折れ線 (割合, y∈[0,1])
└── line_count.png         … コードごとの折れ線 (件数)
```

積み上げ棒・折れ線とも同じビン集計を共有しており、色も `COLOR_MAP` で統一。折れ線版は「ある 1 コードだけの時間推移」を読み取りやすい。

`--manifest` 指定時は `per_subject/{subject_id}/` 配下に同じ構造で出力する。

## モジュール構成

```
src/eval_icce/
├── __init__.py
├── evaluate.py      … 分類精度評価 CLI エントリポイント
├── ena.py           … ENA 共起ネットワーク分析 CLI エントリポイント
├── stacked_bar.py   … 時間ビン積み上げ棒グラフ CLI エントリポイント
├── io.py            … マニフェスト / VLM ログ / GT のロードとラベル正規化
├── align.py         … オフセット適用・bisect でのサンプル ↔ 区間突き合わせ
├── metrics.py       … sklearn ベースの分類性能 / Cohen's kappa
├── report.py        … CSV / PNG / Markdown レポートの書き出し
├── data.csv         … スキーマ確認用サンプルマニフェスト (5 被験者分)
└── README.md        … このファイル
```

## 依存

リポジトリ全体の `pyproject.toml` に追加済み:

- `pandas` (>=3.0)
- `scikit-learn` (>=1.8)
- `matplotlib` (>=3.10)
- `seaborn` (>=0.13)

`uv sync` で揃う。
