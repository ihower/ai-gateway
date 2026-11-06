# llama-nemotron-rerank-1b-v2-rocm

[llama-nemotron-rerank-1b-v2-cuda](../llama-nemotron-rerank-1b-v2-cuda) 的 **AMD ROCm 版**：在 AMD GPU 上用 vLLM 架設 [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) 的 reranker API。支援 Radeon 8060S（Strix Halo 內建，gfx1151）和 Radeon RX 9070 XT（gfx1201）。

`/rerank`、`/score`、預設截斷、分數校準 `sigmoid(logit / 4)` 的行為都和 NVIDIA 版相同。API 用法請看 [NVIDIA 版 README](../llama-nemotron-rerank-1b-v2-cuda/README.md#api)，把 port 換成 12363 即可。

## 和 NVIDIA 版的差異

| | NVIDIA 版 | ROCm 版（本專案） |
|---|---|---|
| GPU | RTX 5070 Ti / 5080 | Radeon 8060S（gfx1151，統一記憶體）、RX 9070 XT（gfx1201） |
| 精度 | FP8（線上 12361、12362） | **BF16**，可選 INT8；9070 XT 可選 FP8 |
| Python | 3.12 | **3.14**（AMD vLLM wheel 只有 cp314） |
| torch | PyPI | 2.13.0+rocm10.1.0 |
| vLLM | 0.31.0 | 0.29.1.dev0+rocm10.1.0（AMD 官方 wheel） |
| 分數校準 | `--pooler-config logit_sigma` | `ScoreCalibrationMiddleware`（尚未改用 0.29 的 `logit_sigma`） |
| 預設 port | 12361 | **12363** |
| 選 GPU | `--gpu N` | `--gpu N`（HIP 編號，`--list-gpus` 查詢） |
| 量化 | `start_fp8.sh` | `start_fp8.sh`（僅 gfx1201）、`start_int8.sh` |

## 和線上 FP8 的分數差距

分數**不會完全相同**。和 12361 比對 5 個 query × 18 份文件（共 90 組）：

| | 結果 |
|---|---|
| 分數平均差 | 0.0035 |
| 分數最大差 | 0.0735（出現在中段分數） |
| 第一名一致 | 5/5 |
| 前三名集合一致 | 3/5（不一致的都是第三名，分數約 0.03～0.04 的無關文件） |

以上是 vLLM 0.19.1 時的結果，資料量不大，僅供參考。詳細比較和不跑 FP8 的原因見 [implementation-notes](docs/implementation-notes.md)。

## FP8（`start_fp8.sh`，僅 RX 9070 XT）

```bash
./start_fp8.sh --gpu 0
```

和 12361 一樣用 `--quantization fp8_per_tensor`，並對齊 NVIDIA 版的算法：

- `vllm_rocm_fp8_pad.py`（vLLM general plugin）：ROCm 上 vLLM 預設用 per-tensor 量化 activation，整批共用一個 scale，分數會受同一批文件影響。這個 plugin 改成和 CUDA 相同的逐 token 量化，並把 `score` 層（2048 → 1）補齊到 16 的倍數，讓它也走 FP8。
- `SCORE_CALIBRATION=native`：用 vLLM 原生 `logit_sigma`，分數和 12361 一樣是 BF16 精度。

8060S（gfx1151）不支援 FP8，`start.sh` 會直接報錯。

和 12361 比對 6 個 query × 23 份文件（共 138 組）：

| | 分數平均差 | 最大差 | 第一名 |
|---|---|---|---|
| 9070 XT FP8（`start_fp8.sh`，關閉 CUDA graph） | 0.0027 | 0.052 | 6/6 |
| 9070 XT FP8（開啟 CUDA graph） | 0.0022 | 0.023 | 6/6 |
| 5080 FP8 eager（同一台 NVIDIA、只換模式） | 0.0028 | 0.028 | 6/6 |
| 9070 XT BF16 | 0.0038 | 0.052 | 6/6 |

和 12361 的差距與 NVIDIA 自己 compile 和 eager 之間的差距相當，跨廠牌的 kernel 不可能逐位元相同。權重記憶體 1.41 GiB（BF16 為 2.32 GiB），8 份長文件 0.91 秒（BF16 為 1.06 秒）。

## INT8（`start_int8.sh`）

線性層權重用 INT8，輸入由 vLLM 動態量化。embedding、norm 和 `score` 仍是 BF16。第一次使用要先轉換：

```bash
uv run python quantize_int8.py   # 輸出到 models/llama-nemotron-rerank-1b-v2-int8（約 1.5 GB）
./start_int8.sh
```

離線實測（eager 模式）：

| | BF16 | INT8 |
|---|---|---|
| 權重記憶體 | 2.40 GiB | 1.49 GiB |
| 峰值記憶體 | 2.85 GiB | 1.94 GiB |
| 64 份短文件 | 0.162 秒 | 0.170 秒 |
| 8 份長文件（約 3900 tokens） | 3.86 秒 | 3.55 秒 |

主要收穫是省記憶體，速度差距不大。INT8 的 compile 模式明顯較慢，所以 `start_int8.sh` 固定用 `--enforce-eager`。

分數比對 5 個 query × 18 份文件（共 90 組）：INT8 和 BF16 的第一名、前三名都一致，分數平均差 0.0022、最大差 0.0432。INT8 和 12361 的分數最大差 0.0510，和 BF16 對 12361 的 0.0509 接近。

## 安裝與啟動

```bash
uv sync
./start.sh
```

會從 AMD 的 index 下載 ROCm SDK、torch 和 vLLM，`.venv` 約 19 GB。版本全部釘死在 `pyproject.toml`，彼此 ABI 必須一致，不要單獨升級其中一個。

## 選擇 GPU

這台機器有兩張 AMD GPU，用 HIP 裝置編號選擇（HIP 是 AMD 對應 CUDA 的 GPU API）：

```bash
./start.sh --list-gpus   # 列出編號，例如 0: RX 9070 XT、1: Radeon 8060S
./start.sh --gpu 0       # RX 9070 XT
./start.sh --gpu 1       # Radeon 8060S
```

`--gpu N` 等同設定 `HIP_VISIBLE_DEVICES=N`。都沒指定時用第 0 張。編號依 PCI 順序，增減顯卡後可能改變，請先用 `--list-gpus` 確認。啟動時會印出實際使用的 GPU 型號。

`GPU_MEMORY_UTILIZATION` 未指定時依 GPU 總記憶體自動計算（`GPU_MEMORY_TARGET_GIB / 總記憶體`）：8060S 約 0.054、9070 XT 約 0.377。

`start.sh` 會關閉 CUDA graph（`cudagraph_mode: NONE`），保留 torch.compile。ROCm 的 graph capture 每次啟動要 2～25 分鐘（5080 只要 2 秒），而且沒有快取。graph 只用在 512 tokens 以下的批次，reranker 通常用不到。關掉後約 30 秒就能啟動。

看到 `Application startup complete` 就可以呼叫，預設監聽 `0.0.0.0:12363`。額外參數會原樣傳給 `vllm serve`。

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `MODEL` | `nvidia/llama-nemotron-rerank-1b-v2` | HF repo 或本地路徑（`start_int8.sh` 預設為本地 INT8 checkpoint） |
| `HOST` | `0.0.0.0` | 監聽位址 |
| `PORT` | `12363` | 監聽 port（`--port` 優先） |
| `DTYPE` | `auto` | `auto` 即 BF16 |
| `MAX_MODEL_LEN` | `4096` | 每組 query + 文件的 token 上限（模型官方上限 8192） |
| `KV_CACHE_MEMORY_BYTES` | `128M` | 每 4096 tokens 需要 128M，調高 `MAX_MODEL_LEN` 時要一併調高 |
| `GPU_MEMORY_UTILIZATION` | 自動計算 | 只用於啟動時的剩餘記憶體檢查，指定時優先 |
| `GPU_MEMORY_TARGET_GIB` | `6` | 自動計算 `GPU_MEMORY_UTILIZATION` 時的目標記憶體 |
| `HIP_VISIBLE_DEVICES` | 未設定 | 選 GPU（`--gpu` 優先） |
| `SCORE_CALIBRATION` | `middleware` | 分數校準方式：`middleware` 或 `native`（vLLM `logit_sigma`；`start_fp8.sh` 預設） |
| `CHAT_TEMPLATE` | `nemotron-rerank.jinja` | score template，**必要**，缺少時分數錯誤 |
| `LOGIT_SIGMA` | `4.0` | 分數校準 `sigmoid(logit / LOGIT_SIGMA)`，必須大於 0 |
| `VLLM_PROCESS_NAME_PREFIX` | `nemotron-rerank-rocm` | 程序名稱前綴 |

## 測試

```bash
uv run pytest                                               # middleware 單元測試，不需要 GPU
uv run python test_client.py                                # 對執行中的 server 做整合測試（預設 :12363）
uv run python test_client.py --compare http://localhost:12361   # 另外和 NVIDIA 版比對分數
```

## ROCm 上要注意的地方

`start.sh` 已經處理好以下幾點，自己寫指令時要記得：

- **`vllm_force_rocm.py` 和 `VLLM_PLUGINS=force_rocm`**：這台同時有 NVIDIA GPU，vLLM 內建偵測會報 `Only one platform plugin can be activated`；也順便擋掉不相容的 `amd-quark` plugin。
- **`pyproject.toml` 排除 Python 3.14.1**：torchvision 0.28.0 不支援這個版本。
- **`PYTHONPATH` 要加上 `_rocm_sdk_core/share/amd_smi`**，並設定 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`。
- **會 `unset CUDA_VISIBLE_DEVICES`**：HIP 也會讀這個變數。
- **不需要 `--enforce-eager`**：這個模型在 compile 模式下正常。

## 檔案

| 檔案 | 說明 |
|---|---|
| `start.sh` | 設定 ROCm 環境並啟動 vLLM |
| `start_int8.sh` | 以 INT8 checkpoint 啟動 |
| `start_fp8.sh` | 以 FP8 啟動（僅 gfx1201） |
| `vllm_rocm_fp8_pad.py` | vLLM plugin，讓 ROCm FP8 的算法和 NVIDIA 一致 |
| `quantize_int8.py` | 把 BF16 權重轉成 INT8 checkpoint |
| `vllm_force_rocm.py` | vLLM platform plugin，強制使用 ROCm |
| `rerank_middleware.py` | 預設截斷、分數校準 middleware |
| `nemotron-rerank.jinja` | score template |
| `test_client.py` | 整合測試 |
| `tests/` | 單元測試 |
| `docs/implementation-notes.md` | 實作決策紀錄 |
