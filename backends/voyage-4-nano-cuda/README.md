# voyage-4-nano-server

用 [vLLM](https://github.com/vllm-project/vllm) 架設 [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) 的 **OpenAI 相容 embedding API**，並用一個 ASGI middleware 支援 Voyage 的 `input_type`（query / document prompt）。

- 輸出 2048 維（可用 `dimensions` 降到 1024 / 512 / 256），向量已 L2 normalize
- 和 OpenAI `/v1/embeddings` 格式相容，支援 `encoding_format: "base64"`
- 只有一個 process：middleware 直接掛在 vLLM 的 API server 上，不需要另外開 proxy

## 架構

```
client ──HTTP──▶ vLLM API server :12351
                  ├─ VoyagePromptMiddleware：依 input_type 在每段文字前加 prompt
                  └─ vLLM engine（VoyageQwen3BidirectionalEmbedModel，mean pooling + 1024→2048 linear head）
```

middleware 只處理 request（解析 JSON、加 prompt），response 原封不動，向量只序列化一次。

## 環境需求

- Linux 與 NVIDIA GPU（已在 RTX 5070 Ti 16GB 測試；預設只用 30% GPU 記憶體）
- [uv](https://docs.astral.sh/uv/)
- 第一次啟動會從 Hugging Face 下載模型（約 670 MB），存在 `~/.cache/huggingface/hub/`

## 安裝

```bash
uv sync
```

## 啟動

```bash
./serve_vllm.sh
```

看到 `Application startup complete` 就可以開始呼叫了。可以用環境變數調整設定：

| 變數 | 預設 | 說明 |
|---|---|---|
| `HOST` | `0.0.0.0` | 監聽位址 |
| `PORT` | `12351` | 監聽 port |
| `GPU_MEMORY_UTILIZATION` | `0.15` | vLLM 可使用的 GPU 記憶體比例 |
| `MAX_MODEL_LEN` | `8192` | 單筆輸入的 token 上限 |
| `API_SERVER_COUNT` | `1` | API server process 數；高併發時 CPU 不夠可以調高 |
| `DEFAULT_INPUT_TYPE` | `document` | 請求沒帶 `input_type` 時用的值 |
| `VLLM_PROCESS_NAME_PREFIX` | `voyage-4-nano` | `ps` 顯示的 process 名稱前綴，例如 `voyage-4-nano::EngineCore` |

用 `--gpu` 指定要用哪張 GPU（等同設定 `CUDA_VISIBLE_DEVICES`），例如 `./serve_vllm.sh --gpu 1`。

其他額外的參數會原樣傳給 `vllm serve`，例如 `./serve_vllm.sh --uvicorn-log-level warning`。

## API

### `POST /v1/embeddings`

| 欄位 | 說明 |
|---|---|
| `input` | 字串或字串陣列 |
| `model` | `voyage-4-nano` |
| `input_type` | **擴充欄位**：`query`、`document`（預設）或 `none` |
| `dimensions` | `2048`（預設）、`1024`、`512`、`256`，其他值回 400 |
| `encoding_format` | `float`（預設）或 `base64` |

```bash
curl http://localhost:12351/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "voyage-4-nano",
  "input": "火星為什麼是紅色的？",
  "input_type": "query",
  "dimensions": 512
}'
```

用 OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:12351/v1", api_key="unused")

docs = client.embeddings.create(
    model="voyage-4-nano",
    input=["火星表面富含氧化鐵，因此呈現紅色。", "金星常被稱為地球的雙胞胎。"],
    dimensions=1024,
)  # 沒帶 input_type 時預設為 document

query = client.embeddings.create(
    model="voyage-4-nano",
    input="哪顆行星是紅色的？",
    dimensions=1024,
    extra_body={"input_type": "query"},
)
```

另外也有 `GET /v1/models` 和 `GET /health`。其他 vLLM 原生端點都照常可用，middleware 只攔截 `POST /v1/embeddings`。

### `input_type` 是什麼

voyage-4-nano 訓練時會在輸入前面加上 prompt，用來區分查詢和被搜尋的文件。middleware 會依 `input_type` 自動加上：

| `input_type` | 加在每段文字前面的 prompt |
|---|---|
| `query` | `Represent the query for retrieving supporting documents: ` |
| `document` | `Represent the document for retrieval: ` |
| `none` | 不加（只有這個模式才能傳 token id 陣列） |

- **建索引**的文件用 `document`，**使用者查詢**用 `query`。用錯的話檢索品質會明顯變差：同一句話加 query 和 document prompt 得到的向量，cosine 只有約 0.68。
- 相似度比對、分群、去除重複這類對稱的任務，兩邊用同一個 `input_type`（一般用 `document`）。
- 一般 OpenAI client 不會帶 `input_type`，所以預設當成 `document`。**查詢時要記得帶 `input_type: "query"`**。

### 選擇維度

模型用 Matryoshka 方式訓練，降維後的檢索品質損失很小。向量越小，儲存和搜尋都越省。**query 和 document 必須用同一個維度**，維度在建索引時就決定了。

## 測試

```bash
uv run pytest                    # middleware 單元測試，不需要 GPU
uv run python test_client.py     # 對正在跑的 server 做 HTTP smoke test
uv run python test_client.py --base-url http://other-host:12351
```

`test_client.py` 只打 HTTP API，不會在本機載入模型，任何機器都能跑。

## 已知事項

- **一定要加 `--enforce-eager`**（`serve_vllm.sh` 已經加了）：vLLM 0.16 的 `VoyageQwen3BidirectionalEmbedModel` 在 torch.compile 或 CUDA graph capture 時會出現 `IndexError: list index out of range`。model card 的範例也用 eager 模式。升級 vLLM 時可以試著拿掉。
- **數值有微小抖動**：bf16 推論時，同樣的輸入在不同請求間逐值差約 1e-4 等級（cosine 約 0.99997），檢索時不會有影響。
- vLLM 的 console script 不會把目前目錄放進 `sys.path`，所以 `serve_vllm.sh` 會設定 `PYTHONPATH`，讓 middleware 找得到。

## 檔案

| 檔案 | 說明 |
|---|---|
| `serve_vllm.sh` | 啟動 vLLM 並掛上 middleware |
| `prompt_middleware.py` | ASGI middleware：處理 `input_type` 和 prompt |
| `test_client.py` | HTTP smoke test |
| `tests/test_prompt_middleware.py` | middleware 單元測試 |
| `docs/implementation-notes.md` | 設計決策與排錯紀錄 |

## 授權

模型 voyage-4-nano 以 Apache 2.0 授權釋出，詳見 [model card](https://huggingface.co/voyageai/voyage-4-nano)。
