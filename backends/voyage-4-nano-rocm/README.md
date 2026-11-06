# voyage-4-nano-rocm

[voyage-4-nano-cuda](../voyage-4-nano-cuda) 的 **AMD ROCm 版**：在 Radeon 8060S（Strix Halo，gfx1151）上用 vLLM 架設 [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) 的 OpenAI 相容 embedding API。

API、`input_type` middleware、`dimensions`、base64 等行為和 NVIDIA 版完全相同（`prompt_middleware.py` 原樣沿用），差別只在執行環境。

## 和 NVIDIA 版的差異

| | NVIDIA 版 | ROCm 版（本專案） |
|---|---|---|
| GPU | RTX 5070 Ti | Radeon 8060S（gfx1151，統一記憶體） |
| Python | 3.12+ | **3.13**（AMD wheel 只有 cp313） |
| torch | 2.9.1+cu128（PyPI） | 2.10.0+rocm7.13.0（`repo.amd.com/rocm/whl/gfx1151`） |
| vLLM | 0.16.0（PyPI） | 0.19.1+rocm7.13.0（AMD 官方 gfx1151 wheel） |
| 預設 port | 12351 | **12353**（兩個可以同時跑） |
| attention backend | FlashAttention | `ROCM_ATTN` |
| 選 GPU | `--gpu N` | 無（只有一張 AMD GPU） |

## 安裝

```bash
uv sync
```

會從 AMD 的 index 下載 ROCm SDK、torch 和 vLLM，總共數 GB。版本全部釘死在 `pyproject.toml`，彼此 ABI 必須一致，不要單獨升級其中一個。

## 啟動

```bash
./serve_vllm.sh
```

看到 `Application startup complete` 就可以呼叫。環境變數（`HOST`、`PORT`、`GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN`、`API_SERVER_COUNT`、`DEFAULT_INPUT_TYPE`、`VLLM_PROCESS_NAME_PREFIX`）和 NVIDIA 版相同，只有 `PORT` 預設改成 `12353`。其他參數會原樣傳給 `vllm serve`。

API 用法請看 [NVIDIA 版 README](../voyage-4-nano-cuda/README.md#api)，把 port 換成 12353 即可。

## 測試

```bash
uv run pytest                    # middleware 單元測試，不需要 GPU
uv run python test_client.py     # 對正在跑的 server 做 HTTP smoke test（預設 :12353）
```

## ROCm 上要注意的地方

`serve_vllm.sh` 已經處理好以下幾點，自己寫指令時要記得：

- **同時有 NVIDIA GPU 時，vLLM 會報 `Only one platform plugin can be activated, but got: ['rocm', 'cuda']`**。vLLM 內建的平台偵測沒辦法用環境變數關掉，所以本專案註冊了一個 out-of-tree platform plugin（`vllm_force_rocm.py`，透過 `pyproject.toml` 的 entry point 生效），強制使用 ROCm。
- **`VLLM_PLUGINS=force_rocm`**：只載入上面那個 plugin。vLLM 依賴的 `amd-quark` 0.13 也會註冊一個 vLLM plugin，但它和這版 vLLM 不相容（`ImportError: cannot import name 'RoutedExperts'`），我們也不需要它的線上量化功能。
- **`PYTHONPATH` 要加上 `_rocm_sdk_core/share/amd_smi`**，並設定 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`：這是 AMD 文件要求的，否則 vLLM 偵測不到 ROCm 平台。
- **還是要加 `--enforce-eager`**：vLLM 0.19.1 仍然有 torch.compile 的 `IndexError: list index out of range`，和 NVIDIA 版的 0.16 相同。
- **會 `unset CUDA_VISIBLE_DEVICES`**：HIP 也會讀這個變數，若外面為了 NVIDIA 設了 `CUDA_VISIBLE_DEVICES=1`，會找不到 AMD GPU。
- **統一記憶體**：GPU 記憶體就是系統 RAM（GPU 可用約 112 GiB）。`GPU_MEMORY_UTILIZATION=0.15` 約 17 GiB，會實際佔用系統記憶體；模型權重本身只有約 0.7 GiB，RAM 吃緊時可以調低。

## 驗證結果

用 offline `LLM(enforce_eager=True)` 對 query 和中英文 document 做 embedding，和 Sentence Transformers（CPU，fp32）比對，4 筆的 cosine 都是 0.99998，輸出 2048 維，已經 L2 normalize。

## 檔案

| 檔案 | 說明 |
|---|---|
| `serve_vllm.sh` | 設定 ROCm 環境並啟動 vLLM |
| `vllm_force_rocm.py` | vLLM platform plugin，強制使用 ROCm |
| `prompt_middleware.py` | ASGI middleware（和 NVIDIA 版相同） |
| `test_client.py` | HTTP smoke test |
| `tests/test_prompt_middleware.py` | middleware 單元測試 |
| `docs/implementation-notes.md` | 設計決策與排錯紀錄 |
