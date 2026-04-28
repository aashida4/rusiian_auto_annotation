# rusiian_auto_annotation

ロシア語学習シーンの注視対象自動判定、および e-learning 受講者の居眠り・集中度判定を行うツール群。
Ollama + Vision LLM (gemma4:31b) を使い、動画フレームを逐次解析して JSON で結果を出力する。

## スクリプト一覧

| スクリプト | 概要 |
|---|---|
| `main.py` | 一人称視点画像から注視対象 (Dictionary / Paper / Task / Memo / Others) を判定 |
| `main_drowsiness.py` | e-learning 受講者の drowsiness (1-3) と engagement (1-2) を判定 |
| `evaluate.py` | `main.py` の出力を人間アノテーション TSV と比較して精度評価 |
| `evaluate_drowsiness.py` | `main_drowsiness.py` の出力を drowsiness / engagement 各 TSV と比較して精度評価 |
| `visualize.py` | `main.py` の判定結果を動画フレームにオーバーレイして可視化 |

## セットアップ

### ローカル実行

```bash
# 依存インストール (uv)
uv sync

# Ollama が起動していることを確認
ollama list
```

### コンテナ (Apptainer / Singularity)

```bash
# SIF ビルド (ローカルで実行し、サーバへ転送)
apptainer build rusiian_auto_annotation.sif apptainer.def
# Singularity の場合
sudo singularity build rusiian_auto_annotation.sif apptainer.def

# サーバへ転送
scp rusiian_auto_annotation.sif <server>:/path/to/
```

## 使い方

### main.py (注視対象判定)

```bash
# 単一画像
uv run python main.py input/sample.png

# 動画 (10秒間隔)
uv run python main.py input/sample.mp4 --interval 10.0

# マルチ GPU (自動検出)
uv run python main.py input/sample.mp4 --interval 1.0

# GPU 数を明示
uv run python main.py input/sample.mp4 --interval 1.0 --num-gpus 2

# モデルを指定
uv run python main.py input/sample.mp4 --model gemma4:31b
```

| 引数 | 説明 | デフォルト |
|---|---|---|
| `input` | 画像または動画ファイルのパス | (必須) |
| `--model` | 使用する Ollama モデル名 | gemma4:31b |
| `--interval` | フレーム抽出間隔 (秒) | 1.0 |
| `--num-gpus` | 並列 Ollama インスタンス数 | nvidia-smi で自動検出 |

出力: `<動画名>_results.json`, `<動画名>_metadata.json`

### main_drowsiness.py (居眠り・集中度判定)

```bash
# 動画
uv run python main_drowsiness.py input/face/sample.mp4 --interval 10.0

# マルチ GPU
uv run python main_drowsiness.py input/face/sample.mp4 --num-gpus 2

# モデルを指定
uv run python main_drowsiness.py input/face/sample.mp4 --model qwen3.6:35b
```

| 引数 | 説明 | デフォルト |
|---|---|---|
| `input` | 画像または動画ファイルのパス | (必須) |
| `--model` | 使用する Ollama モデル名 | gemma4:31b |
| `--interval` | フレーム抽出間隔 (秒) | 1.0 |
| `--num-gpus` | 並列 Ollama インスタンス数 | nvidia-smi で自動検出 |

出力: `<動画名>_drowsiness_results.json`, `<動画名>_drowsiness_metadata.json`

### evaluate.py (注視対象の精度評価)

```bash
uv run python evaluate.py input/sample.mp4 \
  --annotation input/sample_annotation.tsv \
  --output results.csv
```

| 引数 | 説明 | デフォルト |
|---|---|---|
| `video` | 動画ファイルのパス (JSON パス推定用) | (必須) |
| `--json` | 結果 JSON のパス | `<動画名>_results.json` |
| `--annotation` | 人間アノテーション TSV | (必須) |
| `--offset` | 時刻オフセット (秒) | 0.0 |
| `--output` | 比較結果 CSV 出力先 | (なし) |

### evaluate_drowsiness.py (居眠り・集中度の精度評価)

```bash
uv run python evaluate_drowsiness.py input/face/sample.mp4 \
  --drowsiness input/face/sample.drowsiness.tsv \
  --engagement input/face/sample.engagement.tsv \
  --output results.csv
```

| 引数 | 説明 | デフォルト |
|---|---|---|
| `video` | 動画ファイルのパス (JSON パス推定用) | (必須) |
| `--json` | 結果 JSON のパス | `<動画名>_drowsiness_results.json` |
| `--drowsiness` | drowsiness アノテーション TSV | (必須) |
| `--engagement` | engagement アノテーション TSV | (必須) |
| `--offset` | 時刻オフセット (秒) | 0.0 |
| `--output` | 比較結果 CSV 出力先 | (なし) |

### visualize.py (結果の可視化)

```bash
uv run python visualize.py input/sample.mp4 --grid
```

| 引数 | 説明 | デフォルト |
|---|---|---|
| `video` | 動画ファイルのパス | (必須) |
| `--json` | 結果 JSON のパス | `<動画名>_results.json` |
| `--output` | 出力ディレクトリ | `<動画名>_vis/` |
| `--max` | 出力フレーム上限数 | (なし) |
| `--grid` | グリッド画像も生成 | false |
| `--grid-cols` | グリッドの列数 | 6 |

## コンテナでの実行

```bash
# 注視対象判定 (デフォルト: main.py)
singularity run --nv \
  --bind /path/to/models:/models \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation.sif input/sample.mp4 --num-gpus 2

# 居眠り判定 (--app drowsiness)
singularity run --app drowsiness --nv \
  --bind /path/to/models:/models \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation.sif input/sample.mp4 --num-gpus 4

# 評価
singularity run --app evaluate --nv \
  --bind /path/to/models:/models \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation.sif input/sample.mp4 --annotation input/sample.tsv

# 可視化
singularity run --app visualize --nv \
  --bind /path/to/models:/models \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation.sif input/sample.mp4 --grid

# Ollama CLI
singularity run --app ollama --nv \
  --bind /path/to/models:/models \
  rusiian_auto_annotation.sif list
```

### マルチ GPU 時の注意

- `--num-gpus 2` 以上を指定すると、スクリプトが GPU ごとに `ollama serve` を自動起動する
- 既存の Ollama サービスが VRAM を掴んでいる場合は自動でアンロードを試みる
- コンテナ内で `USE_EXTERNAL_OLLAMA=1` を付けると、entrypoint の Ollama 起動をスキップできる (マルチ GPU 時に推奨)

```bash
USE_EXTERNAL_OLLAMA=1 singularity run --app drowsiness --nv \
  --bind /path/to/models:/models \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation.sif input/sample.mp4 --num-gpus 4
```

## モデル設定

- モデル: `--model` フラグで指定可能 (デフォルト: 両スクリプトとも `gemma4:31b`)
- コンテキスト長: `8192` トークン (`NUM_CTX` で変更可)

## メタデータ

動画解析時に `_metadata.json` / `_drowsiness_metadata.json` が自動生成され、以下の情報が記録される:

- `model`: 使用モデル名
- `num_ctx`: コンテキスト長
- `video`: 動画ファイル名
- `fps`: フレームレート
- `total_frames`: 総フレーム数
- `duration_sec`: 動画の長さ (秒)
- `interval_sec`: フレーム抽出間隔 (秒)
- `frame_interval`: フレーム間隔 (フレーム数)
- `analyzed_frames`: 解析したフレーム数
- `num_gpus`: 使用 GPU 数
- `start_time` / `end_time`: 解析の開始・終了時刻
- ホストのモデルディレクトリ:
  - systemd Ollama: `/usr/share/ollama/.ollama/models`
  - ユーザ Ollama: `~/.ollama/models`
