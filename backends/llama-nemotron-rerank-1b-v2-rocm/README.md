# llama-nemotron-rerank-1b-v2-rocm

[llama-nemotron-rerank-1b-v2-server](../llama-nemotron-rerank-1b-v2-server) 的 **AMD ROCm 版**：在 Radeon 8060S（Strix Halo，gfx1151）上用 vLLM 架設 [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) 的 reranker API。

`/rerank`、`/score`、預設截斷、分數校準 `sigmoid(logit / 4)` 的行為都和 NVIDIA 版相同。API 用法請看 [NVIDIA 版 README](../llama-nemotron-rerank-1b-v2-server/README.md#api)，把 port 換成 12363 即可。

## 和 NVIDIA 版的差異

| | NVIDIA 版 | ROCm 版（本專案） |
|---|---|---|
| GPU | RTX 5070 Ti / 5080 | Radeon 8060S（gfx1151，統一記憶體） |
| 精度 | FP8（線上 12361、12362） | **BF16** |
| Python | 3.12 | **3.13**（AMD wheel 只有 cp313） |
| torch | PyPI | 2.10.0+rocm7.13.0 |
| vLLM | 0.31.0 | 0.19.1+rocm7.13.0（AMD 官方 gfx1151 wheel） |
| 分數校準 | `--pooler-config logit_sigma` | `ScoreCalibrationMiddleware`（0.19.1 沒有 `logit_sigma`） |
| 預設 port | 12361 | **12363** |
| 選 GPU | `--gpu N` | 無 |
| FP8 | `start_fp8.sh` | 不支援 |

## 和線上 FP8 的分數差距

分數**不會完全相同**。和 12361 比對 5 個 query × 18 份文件（共 90 組）：

| | 結果 |
|---|---|
| 分數平均差 | 0.0035 |
| 分數最大差 | 0.0735（出現在中段分數） |
| 第一名一致 | 5/5 |
| 前三名集合一致 | 3/5（不一致的都是第三名，分數約 0.03～0.04 的無關文件） |

資料量不大，僅供參考。詳細比較和不跑 FP8 的原因見 [implementation-notes](docs/implementation-notes.md)。

## 安裝與啟動

```bash
uv sync
./start.sh
```

會從 AMD 的 index 下載 ROCm SDK、torch 和 vLLM，總共數 GB。版本全部釘死在 `pyproject.toml`，彼此 ABI 必須一致，不要單獨升級其中一個。

看到 `Application startup complete` 就可以呼叫，預設監聽 `0.0.0.0:12363`。額外參數會原樣傳給 `vllm serve`。

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `MODEL` | `nvidia/llama-nemotron-rerank-1b-v2` | HF repo 或本地路徑 |
| `HOST` | `0.0.0.0` | 監聽位址 |
| `PORT` | `12363` | 監聽 port（`--port` 優先） |
| `DTYPE` | `auto` | `auto` 即 BF16 |
| `MAX_MODEL_LEN` | `4096` | 每組 query + 文件的 token 上限（模型官方上限 8192） |
| `KV_CACHE_MEMORY_BYTES` | `128M` | 每 4096 tokens 需要 128M，調高 `MAX_MODEL_LEN` 時要一併調高 |
| `GPU_MEMORY_UTILIZATION` | `0.05` | 只用於啟動時的剩餘記憶體檢查 |
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
- **`PYTHONPATH` 要加上 `_rocm_sdk_core/share/amd_smi`**，並設定 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`。
- **會 `unset CUDA_VISIBLE_DEVICES`**：HIP 也會讀這個變數。
- **不需要 `--enforce-eager`**：這個模型在 compile 模式下正常。

## 檔案

| 檔案 | 說明 |
|---|---|
| `start.sh` | 設定 ROCm 環境並啟動 vLLM |
| `vllm_force_rocm.py` | vLLM platform plugin，強制使用 ROCm |
| `rerank_middleware.py` | 預設截斷、分數校準 middleware |
| `nemotron-rerank.jinja` | score template |
| `test_client.py` | 整合測試 |
| `tests/` | 單元測試 |
| `docs/implementation-notes.md` | 實作決策紀錄 |
