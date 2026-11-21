# pplx-decider-flash-cuda

用 [vLLM](https://github.com/vllm-project/vllm) 跑 [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3)（NVFP4），提供與 llama.cpp 相同的 **SystemOne 格式 `POST /v1/systemone`**，可直接取代原本的 `llama serve` Q4_K_M 後端。

- 模型：Qwen3.5-4B backbone + 255 列 decision readout，full attention 為 **noncausal**、GDN（linear attention）為 causal
- vLLM 0.34 內建這種架構（`D3ForSequenceClassification`），所以不需要自訂模型；`prepare_model.py` / `quantize_nvfp4.py` 只把 `decision_config.json` 的 `format_version` 改成 vLLM 接受的 1
- prompt 完全沿用官方 protocol v4（`pplx_decider/protocol.py`，未修改），每題一個 sequence（官方預設的 `separate` mode），marker 在最後一個 token，由 vLLM 的 LAST pooling 取出
- 只支援文字題（`choice` / `noul` / `score`）；`images`、`locate` 回 400（與 llama.cpp 版相同）
- 只有一個 process：middleware 掛在 vLLM 的 API server 上，在 process 內轉呼叫 `/classify`

## 架構

```
client ──HTTP──▶ vLLM API server :12372
                  ├─ SystemOneMiddleware：官方 protocol v4 編碼 → /classify（token ids）→ 溫度 + softmax → SystemOne 回應
                  └─ vLLM engine（D3ForSequenceClassification，NVFP4 CUTLASS GEMM）
```

## 安裝與準備模型

```bash
uv sync --group quantize --group dev
# 下載官方 BF16 checkpoint（約 9 GB）
uvx --from huggingface_hub hf download perplexity-ai/pplx-decider-flash-v1.3 --local-dir /mnt/data/models/pplx/pplx-decider-flash-v1.3
# 量化成 NVFP4（約 1 分鐘，GPU 峰值約 10 GiB）
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python quantize_nvfp4.py \
    --src /mnt/data/models/pplx/pplx-decider-flash-v1.3 \
    --dest /mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4 --keep '*self_attn*'
```

量化設定：ModelOpt `NVFP4_DEFAULT_CFG`（W4A4，max 校準），8 層 full attention 的 q/k/v/o、GDN 的 `in_proj_a/b`、`conv1d`、vision tower 與 readout 保留 BF16。校準時 full attention 為 noncausal（與 serving 相同），校準資料為 `tests/fixtures/calibration-requests.json` 加上 `tests/synthetic.py` 的 calib split（133 個 sequences）。

## 啟動

```bash
./start_nvfp4.sh --gpu 1 --port 12372
```

看到 `Application startup complete` 就可以呼叫。第一次啟動會做 torch.compile 與 CUDA graph capture（之後有快取）。

| 變數 | 預設 | 說明 |
|---|---|---|
| `GPU` / `--gpu` | `1`（RTX 5080） | CUDA 裝置編號 |
| `PORT` / `--port` | `12372` | 監聽 port |
| `HOST` | `0.0.0.0` | 監聽位址 |
| `MODEL` | `/mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4` | 模型目錄 |
| `MAX_MODEL_LEN` | `32768` | 單一 sequence（state + 一題）的 token 上限，超過回 400，不截斷 |
| `GPU_MEMORY_UTILIZATION` | `0.5` | vLLM 可用的 GPU 記憶體比例 |
| `VLLM_PROCESS_NAME_PREFIX` | `pplx-decider-flash` | `ps` 顯示的 process 名稱前綴 |

額外參數原樣傳給 `vllm serve`。改跑 FP8（準確度較好、速度較慢，見下表）：

```bash
MODEL=/mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-bf16 ./start_nvfp4.sh --quantization fp8
```

（BF16 目錄用 `uv run --no-sync python prepare_model.py --src <官方目錄> --dest <輸出目錄>` 建立，全部為 symlink。）

### 注意

- 固定 `--linear-backend cutlass`：auto 會選 FlashInfer 的 NVFP4 GEMM，第一次使用要 JIT 編譯 `fp4_gemm_cutlass_sm120`，耗時很久且平行 nvcc 很吃主機 RAM（曾因此 OOM）。
- 不能開 prefix caching：noncausal attention 下前綴的 hidden state 依賴後文。

## API

`POST /v1/systemone`，格式同 llama.cpp / Jev SystemOne（見 ai-gateway SPEC §6）：

```bash
curl http://localhost:12372/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "My integration keeps failing with a 401 since we rotated our API key.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this request?",
               "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"}
  }
}'
```

回應 `{"model", "answers", "usage": {"input_tokens", "output_tokens", "sequences"}}`。與 llama.cpp 版的差異：`instructions` 可省略（官方預設 `Choose the best matching option.`）；可帶 `temperature` 覆寫校準溫度。

## 準確度與速度（RTX 5080）

評估集：官方 runtime（FP32 CPU）的 27 題 fixtures + 60 題合成 eval（不參與校準），比對選項機率（0–1）。

| 版本 | top answer vs 官方 | |Δp| mean vs 官方 | top answer vs BF16（87 題） | |Δp| mean vs BF16 | 吞吐量 | 單筆延遲（≈3 題） |
|---|---|---|---|---|---|---|
| vLLM BF16 | 27/27 | 0.00045 | — | — | 124 seq/s | 32 ms |
| vLLM FP8（`--quantization fp8`） | 26/27 | 0.0023 | 86/87 | 0.0047 | 205 seq/s | 21 ms |
| **vLLM NVFP4 W4A4（本設定）** | 25/27 | 0.0101 | 84/87 | 0.0143 | **366 seq/s** | **16 ms** |
| llama.cpp Q4_K_M（5070 Ti，參考） | 27/27 | 0.0037 | 85/87 | 0.0075 | — | 55 ms（含 HTTP） |

NVFP4 的機率誤差約為 FP8 的 3 倍（E2M1 的單一權重誤差約為 E4M3 的 4 倍）；不一致的題目都在決策邊界附近。依賴機率值本身（而非 top answer）的用途建議改用 FP8。

## 測試

```bash
uv run --no-sync pytest                                   # middleware 單元測試（不需要 GPU）
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python tests/check_accuracy.py <模型目錄> --baseline tests/fixtures/bf16-answers.json  # 離線準確度
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python tests/bench.py <模型目錄>                                # 離線速度
uv run --no-sync python tests/check_server.py http://127.0.0.1:12372 --baseline tests/fixtures/bf16-answers.json              # 對 server
```

## 授權

模型與 `pplx_decider/`（官方 protocol）為 Apache-2.0，見 `pplx_decider/LICENSE`、`pplx_decider/NOTICE`。`tests/fixtures/` 取自 [fxoz/pplx-decider-flash-v1.3-GGUF](https://huggingface.co/fxoz/pplx-decider-flash-v1.3-GGUF)（Apache-2.0）。
