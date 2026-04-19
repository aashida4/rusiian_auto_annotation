# rusiian_auto_annotation

ロシア語学習シーンの注視対象自動判定、および e-learning 受講者の居眠り・集中度判定を行うツール群。
Vision LLM (gemma4:31b など) で動画フレームを逐次解析し、JSON で結果を出力する。
推論バックエンドは **Ollama** (デフォルト) と **llama.cpp** を切り替え可能。

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

2 種類の定義ファイルを用意しています:

| ファイル | バックエンド | 用途 |
|---|---|---|
| `apptainer.def` | Ollama (デフォルト) | 既存の Ollama 運用向け。llama.cpp も外部サーバ接続 or llama-server バイナリのバインドマウントで利用可能 |
| `apptainer_llamacpp.def` | llama.cpp (CUDA 同梱) | llama-server を CUDA 対応でビルドして同梱。Ollama は含まない |

```bash
# Ollama 版
apptainer build rusiian_auto_annotation.sif apptainer.def
sudo singularity build rusiian_auto_annotation.sif apptainer.def

# llama.cpp 版 (CUDA 同梱)
apptainer build rusiian_auto_annotation_llamacpp.sif apptainer_llamacpp.def
sudo singularity build rusiian_auto_annotation_llamacpp.sif apptainer_llamacpp.def

# サーバへ転送
scp rusiian_auto_annotation*.sif <server>:/path/to/
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
```

| 引数 | 説明 | デフォルト |
|---|---|---|
| `input` | 画像または動画ファイルのパス | (必須) |
| `--interval` | フレーム抽出間隔 (秒) | 1.0 |
| `--num-gpus` | 並列サーバインスタンス数 | nvidia-smi で自動検出 |
| `--backend` | 推論バックエンド (`ollama` / `llama.cpp`) | `ollama` |
| `--llama-model` | GGUF モデルのパス (llama.cpp) | (env `LLAMA_MODEL`) |
| `--llama-mmproj` | mmproj GGUF のパス (llama.cpp) | (env `LLAMA_MMPROJ`) |
| `--llama-server-bin` | `llama-server` バイナリのパス | `llama-server` |
| `--llama-host` | 外部 `llama-server` URL (指定時はサーバ起動をスキップ) | (env `LLAMA_HOST`) |
| `--llama-ngl` | GPU にオフロードするレイヤ数 | 999 (全部) |

出力: `<動画名>_results.json`

### main_drowsiness.py (居眠り・集中度判定)

```bash
# 動画
uv run python main_drowsiness.py input/face/sample.mp4 --interval 10.0

# マルチ GPU
uv run python main_drowsiness.py input/face/sample.mp4 --num-gpus 2
```

引数は `main.py` と共通 (`input`, `--interval`, `--num-gpus`, `--backend`, `--llama-*`)。

出力: `<動画名>_drowsiness_results.json`

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

- モデル: `gemma4:31b` (各スクリプトの `MODEL_NAME` で変更可。llama.cpp バックエンド時は無視される)
- コンテキスト長: `8192` トークン (`NUM_CTX` で変更可)
- ホストのモデルディレクトリ (Ollama):
  - systemd Ollama: `/usr/share/ollama/.ollama/models`
  - ユーザ Ollama: `~/.ollama/models`

## バックエンド

### Ollama (デフォルト)

追加の設定は不要。`ollama serve` が起動していて、`MODEL_NAME` のモデルが `ollama pull` 済みであればそのまま動きます。マルチ GPU 時はスクリプトが GPU ごとに `ollama serve` を自動起動します。

### llama.cpp

`llama-server` を使う OpenAI 互換 API 経由のバックエンド。Ollama より細かいチューニングが可能で、Ollama に依存せずに動かせます。

#### 事前準備 (ユーザ作業)

1. **`llama-server` バイナリを用意する**
   - ソースからビルド (CUDA 対応):
     ```bash
     git clone https://github.com/ggml-org/llama.cpp
     cd llama.cpp
     cmake -B build -DGGML_CUDA=ON
     cmake --build build --config Release -j
     # 成果物: build/bin/llama-server
     ```
   - もしくは [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) の CUDA 対応プリビルドバイナリを取得。
   - `PATH` に通すか、`--llama-server-bin /path/to/llama-server` で指定。

2. **GGUF モデルと mmproj ファイルを用意する**
   Vision (マルチモーダル) 対応モデルでは 2 ファイル必要です。Hugging Face から取得するのが早いです。
   例 (Gemma 3 27B, unsloth/ggml-org 提供の GGUF を使う場合):
   ```bash
   mkdir -p ~/gguf && cd ~/gguf
   # 量子化済み本体
   wget https://huggingface.co/ggml-org/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q4_K_M.gguf
   # mmproj (vision エンコーダ)
   wget https://huggingface.co/ggml-org/gemma-3-27b-it-GGUF/resolve/main/mmproj-gemma-3-27b-it-f16.gguf
   ```
   選択するモデルは `llama-server` が対応している Vision 系に限られる点に注意 (Gemma 3 / Qwen2-VL / LLaVA 系など)。

3. **疎通確認 (任意)**
   ```bash
   llama-server \
     --model ~/gguf/gemma-3-27b-it-Q4_K_M.gguf \
     --mmproj ~/gguf/mmproj-gemma-3-27b-it-f16.gguf \
     --ctx-size 8192 --n-gpu-layers 999 --port 11600 --jinja
   # 別ターミナルで:
   curl http://127.0.0.1:11600/v1/models
   ```

#### 実行

```bash
# 内蔵で llama-server を起動して実行 (GPU ごとに 1 プロセス)
uv run python main.py input/sample.mp4 \
  --backend llama.cpp \
  --llama-model ~/gguf/gemma-3-27b-it-Q4_K_M.gguf \
  --llama-mmproj ~/gguf/mmproj-gemma-3-27b-it-f16.gguf \
  --num-gpus 2

# 既に起動済みの llama-server に接続する (外部サーバ)
uv run python main.py input/sample.mp4 \
  --backend llama.cpp \
  --llama-host http://127.0.0.1:11600
```

環境変数でも同等の指定が可能です:
`BACKEND`, `LLAMA_MODEL`, `LLAMA_MMPROJ`, `LLAMA_SERVER_BIN`, `LLAMA_HOST`, `LLAMA_NGL`。

#### コンテナ (Apptainer) での llama.cpp 利用

2 通りの使い方があります。

**1. llama.cpp 同梱イメージ (`apptainer_llamacpp.def`) を使う (推奨)**

`llama-server` を CUDA 対応でビルド・同梱した専用イメージを作成します。Ollama は含みません。

```bash
# ビルド
apptainer build rusiian_auto_annotation_llamacpp.sif apptainer_llamacpp.def

# 実行 (GGUF をバインドマウント)
singularity run --nv \
  --bind ~/gguf:/gguf \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation_llamacpp.sif input/sample.mp4 \
  --llama-model /gguf/gemma-3-27b-it-Q4_K_M.gguf \
  --llama-mmproj /gguf/mmproj-gemma-3-27b-it-f16.gguf \
  --num-gpus 2
```

このイメージは `BACKEND=llama.cpp` がデフォルトで設定済みです。ビルド時間短縮のためには、`apptainer_llamacpp.def` 内の `CMAKE_CUDA_ARCHITECTURES` を対象サーバの GPU アーキ (RTX 30 系: `86`, A100: `80`, H100: `90` 等) に絞ってください。

**2. Ollama 版イメージに `llama-server` をバインドマウント**

既存の `apptainer.def` 由来の `.sif` を使い、ホスト側でビルド/取得した `llama-server` をバインドマウントする方法です。

```bash
BACKEND=llama.cpp singularity run --nv \
  --bind /path/to/llama-server:/usr/local/bin/llama-server \
  --bind ~/gguf:/models \
  --bind $PWD:/work --pwd /work \
  rusiian_auto_annotation.sif input/sample.mp4 \
  --backend llama.cpp \
  --llama-model /models/gemma-3-27b-it-Q4_K_M.gguf \
  --llama-mmproj /models/mmproj-gemma-3-27b-it-f16.gguf \
  --num-gpus 2
```

`BACKEND=llama.cpp` をセットするとエントリスクリプトが Ollama サーバの自動起動をスキップします。
