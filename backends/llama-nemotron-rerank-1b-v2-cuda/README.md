# llama-nemotron-rerank-1b-v2 server

使用 [vLLM](https://github.com/vllm-project/vllm) 將 [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) 部署為 reranker HTTP API（Cohere / Jina 相容的 `/rerank`）。

- 模型：Llama-3.2-1B 改為雙向注意力的 cross-encoder，mean pooling + 二元分類 head
- 每組 query + 文件上限 **4096 tokens**，超過時**預設自動截斷**（不會回錯誤）
- 授權：OpenMDW-1.1，另適用 Llama 3.2 Community License

## 需求

- Linux + NVIDIA GPU（BF16 權重約 2.3 GB，預設設定總共約需 4 GB 顯示記憶體；FP8 版見 [FP8 量化](#fp8-量化)）
- [uv](https://docs.astral.sh/uv/)
- Python 3.12（已在 `.python-version` 固定；**勿用 Python 3.13.8**，見[注意事項](#注意事項)）

## 啟動

```bash
./start.sh
```

第一次執行會由 uv 自動安裝 vLLM 並下載模型。看到 `Application startup complete` 即可使用，預設監聽 `0.0.0.0:12361`。

### 設定

所有設定都可用環境變數覆寫，額外參數會直接傳給 `vllm serve`：

```bash
PORT=8000 KV_CACHE_MEMORY_BYTES=1G ./start.sh --max-num-batched-tokens 16384
```

也可用 `--gpu` 指定 CUDA 裝置編號（設定 `CUDA_VISIBLE_DEVICES`）、`--port` 指定監聽 port（優先於 `PORT` 環境變數）：

```bash
./start.sh --gpu 1 --port 8001
```

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `MODEL` | `nvidia/llama-nemotron-rerank-1b-v2` | HF repo 或本地路徑 |
| `HOST` | `0.0.0.0` | 監聽位址 |
| `PORT` | `12361` | 監聽 port |
| `DTYPE` | `auto` | `auto` 即 BF16 |
| `MAX_MODEL_LEN` | `4096` | 每組 query + 文件的 token 上限（模型官方上限 8192） |
| `KV_CACHE_MEMORY_BYTES` | `128M` | 固定 KV cache 大小，剛好容納 4096 tokens；調高 `MAX_MODEL_LEN` 時需一併調高（見[記憶體](#gpu-記憶體)） |
| `GPU_MEMORY_UTILIZATION` | `0.2` | 僅作為 vLLM 啟動時的剩餘記憶體檢查門檻 |
| `CHAT_TEMPLATE` | `nemotron-rerank.jinja` | score template，**必要**，缺少時分數錯誤 |
| `POOLER_CONFIG` | `{"logit_sigma": 4.0}` | 傳給 `--pooler-config`，分數校準為 `sigmoid(logit / 4)`（見[分數校準](#分數校準)）；設為空字串回到 vLLM 預設 |
| `VLLM_PROCESS_NAME_PREFIX` | `nemotron-rerank` | 程序名稱前綴，顯示為 `nemotron-rerank::EngineCore` |

## API

### `POST /rerank`

同樣的介面也可用 `/v1/rerank`、`/v2/rerank`。

```bash
curl http://localhost:12361/rerank -H 'Content-Type: application/json' -d '{
  "model": "nvidia/llama-nemotron-rerank-1b-v2",
  "query": "台灣半導體產業的競爭優勢是什麼？",
  "documents": [
    "今天台北天氣晴朗，適合去陽明山爬山。",
    "台積電憑藉先進製程與晶圓代工模式，讓台灣在全球半導體供應鏈中占有關鍵地位。"
  ],
  "top_n": 2
}'
```

回應（依 `relevance_score` 由高到低排序，以下分數僅為示意）：

```json
{
  "id": "rerank-...",
  "model": "nvidia/llama-nemotron-rerank-1b-v2",
  "usage": {"prompt_tokens": 98, "total_tokens": 98},
  "results": [
    {"index": 1, "document": {"text": "台積電憑藉..."}, "relevance_score": 0.97},
    {"index": 0, "document": {"text": "今天台北..."}, "relevance_score": 0.0003}
  ]
}
```

`index` 是文件在請求 `documents` 陣列中的位置。

#### 常用參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `query` | 必填 | 查詢字串 |
| `documents` | 必填 | 文件字串陣列 |
| `top_n` | `0` | 只回傳分數最高的 N 筆，`0` 為全部回傳。**注意是 `top_n`，不是 `top_k`** |
| `truncate_prompt_tokens` | `-1`（由 middleware 補上） | 每組 query + 文件的 token 上限，`-1` 為 `MAX_MODEL_LEN`；傳 `null` 可關閉截斷 |
| `max_tokens_per_doc` | `0`（不限制） | 只限制文件內容的 token 數（Cohere 相容），必須小於 `MAX_MODEL_LEN` |
| `max_tokens_per_query` | `0`（不限制） | 只限制 query 的 token 數 |
| `use_activation` | `true` | `true` 回傳校準後的 0～1 分數 `sigmoid(logit / 4)`；`false` 回傳 `logit / 4`（未經 sigmoid） |

### `POST /score`

同樣的介面也可用 `/v1/score`。不排序，依輸入順序回傳每份文件的分數：

```bash
curl http://localhost:12361/score -H 'Content-Type: application/json' -d '{
  "model": "nvidia/llama-nemotron-rerank-1b-v2",
  "queries": "台灣半導體產業的競爭優勢是什麼？",
  "documents": ["文件一", "文件二"]
}'
```

### 其他

- `GET /health`：健康檢查
- `POST /tokenize`：查看文字的 token 數

### Python 範例

```python
import httpx

r = httpx.post(
    "http://localhost:12361/rerank",
    json={
        "model": "nvidia/llama-nemotron-rerank-1b-v2",
        "query": "台灣半導體產業的競爭優勢是什麼？",
        "documents": docs,
        "top_n": 5,
    },
    timeout=60,
)
for item in r.json()["results"]:
    print(item["index"], item["relevance_score"])
```

## 分數校準

vLLM 預設回傳 `sigmoid(logit)`，在 float32 下 logit 超過約 17 就剛好等於 `1.0`。這個模型對高相關文件的 logit 常在 17～22，實測一組 14 份同主題的研究報告，有 13 份都是 `1.0`，無法排出先後。

因此 `start.sh` 預設以 `--pooler-config '{"logit_sigma": 4.0}'` 啟動，回傳 `sigmoid(logit / 4)`：

- 排序與原始 logit 完全一致，只是分數不再擠在 1.0；要到 logit 約 68 才會飽和
- 中心點不變：logit 0 仍是 0.5
- 分數會往 0.5 靠攏，以 sigmoid(logit) 訂的門檻需要重新校正

| 文件（query：貨櫃航運運價對長榮、陽明、萬海獲利的影響） | logit | sigmoid(logit) | sigmoid(logit / 4) |
|---|---|---|---|
| 天氣、拉麵等無關文件 | -23 ～ -19 | 0.000000 | 0.003 ～ 0.009 |
| 關鍵字相同但主題不同（長榮航空、BDI） | -12 ～ -8 | 0.00001 ～ 0.0003 | 0.05 ～ 0.12 |
| 「貨櫃三雄第三季合計獲利 394 億元」一句話 | -1.0 | 0.26 | 0.44 |
| 高相關研究報告 | 13.9 ～ 21.4 | 0.999999 ～ 1.0 | 0.970 ～ 0.995 |

要調整攤開程度可改 `POOLER_CONFIG`，例如 `POOLER_CONFIG='{"logit_sigma": 3.0}' ./start.sh`（數字越大越往 0.5 集中）；也可加上 `logit_mean` 移動中心點。設為空字串（`POOLER_CONFIG= ./start.sh`）則回到 vLLM 預設。

## 長文件截斷

vLLM 預設遇到超過 `MAX_MODEL_LEN` 的輸入會回 400。本專案的 `rerank_middleware.DefaultTruncationMiddleware` 會在請求沒帶 `truncate_prompt_tokens` 時自動補上 `-1`，改為截斷：

- **每組 query + 文件各自截斷**：一次送 100 份文件，只有超長的那幾份被截掉尾端，其他文件不受影響，也不會被移除
- **從右側截斷**：保留 query 與文件開頭，截掉文件尾端
- **沒有額外成本**：截斷發生在 vLLM 原本就要做的 tokenize 之後，只是切 token list
- 適用路徑：`/rerank`、`/v1/rerank`、`/v2/rerank`、`/score`、`/v1/score`

`truncate_prompt_tokens` 與 `max_tokens_per_doc` 的差異：

| | `truncate_prompt_tokens` | `max_tokens_per_doc` |
|---|---|---|
| 限制範圍 | 整組 prompt（query + 格式字串 + 文件） | 只有文件 |
| 保證不超過上限 | ✅ | ❌ query 很長時仍可能 400 |
| tokenize 次數 | 1 次 | 至少 2 次 |
| 適合 | 防止出錯（middleware 預設） | 固定每份文件最多讀多少 token |

兩者可同時使用。若文件很長且答案可能在後段，建議 client 端先自行切段，再分別 rerank（截斷會遺失後段內容）。

## FP8 量化

NVIDIA 官方只提供 BF16 權重。`start_fp8.sh` 讓 vLLM 在載入時動態量化成 FP8（`--quantization fp8_per_tensor`，不需另外下載模型），需 Ada / Hopper / Blackwell GPU（RTX 40、H100、RTX 50 以上）。參數與環境變數都和 `start.sh` 相同：

```bash
./start_fp8.sh
./start_fp8.sh --gpu 1 --port 8001
```

在 RTX 5070 Ti 上離線比對（vLLM 0.31.0，同一組中英文 query/文件）：

| | BF16 | FP8 |
|---|---|---|
| 權重記憶體 | 2.32 GiB | 1.41 GiB |
| 100 份長文件（每份約 3000 tokens） | 13.6 秒 | 8.9 秒 |
| 分數差距 | — | logit 最大差約 1.4（預設 `sigmoid(logit / 4)` 分數約 0.07） |
| 排序 | — | 高相關文件排序一致；分數都接近 0 的低相關文件順序可能互換 |

需要和 NVIDIA 官方評測結果完全一致時，請用 `start.sh`（BF16）。

## GPU 記憶體

vLLM 啟動時一定會預先配置固定大小的 GPU 記憶體。本專案以 `--kv-cache-memory-bytes` 直接指定 KV cache 大小，而非依 GPU 總量比例配置，方便與其他服務共用 GPU：

- 實際用量約為：權重（FP8 1.4 GB／BF16 2.3 GB）+ KV cache（預設 128M = 4096 tokens × 32 KiB）+ 運算空間
- rerank 的序列做一次 forward 就結束，KV cache 只是 attention 計算的暫存空間，不會跨請求重用
- 每個 step 同時處理的 token 數受 KV cache 容量限制（128M 約 4096 tokens/step），一次送 100 份文件會自動拆成多個 step 依序計算
- `GPU_MEMORY_UTILIZATION` 不影響實際配置，但 vLLM 啟動時仍會檢查「剩餘記憶體 ≥ 總量 × 此比例」，GPU 被其他程式佔用較多時可調低

## 注意事項

- **score template 必要**：`nemotron-rerank.jinja` 將輸入組成 `question:{query} \n \n passage:{document}`，缺少時不會報錯但分數錯誤。
- **Python 3.13.8 無法啟動**：Python 3.13.8 的 `inspect.getsourcelines` 有 bug，造成 vLLM 啟動時 Triton kernel 解析失敗（`'NoneType' object has no attribute 'start'`），因此固定使用 Python 3.12。
- **`top_k` 會被靜默忽略**：vLLM 允許未知欄位，只在 server log 印警告，結果會回傳全部文件。請使用 `top_n`。
- **`MAX_MODEL_LEN` 最多 8192**：模型 config 的 `max_position_embeddings=131072` 只是沿用 Llama 3.2 的 RoPE 設定，模型官方支援上限為 8192。調到 8192 時 `KV_CACHE_MEMORY_BYTES` 需改為 `256M`。
- **token 估算**：中文約 1.5 字元／token，英文約 4 字元／token；4096 tokens 約 6000 個中文字。

## 測試

```bash
# middleware 單元測試（不需要 GPU）
uv run pytest

# 對執行中的 server 做整合測試
uv run python test_client.py                       # 預設 http://localhost:12361
uv run python test_client.py http://HOST:12361
```

## 檔案

| 檔案 | 說明 |
|---|---|
| `start.sh` | 啟動腳本（BF16） |
| `start_fp8.sh` | FP8 啟動腳本（包裝 `start.sh`） |
| `nemotron-rerank.jinja` | score template |
| `rerank_middleware.py` | 預設截斷 middleware |
| `test_client.py` | 整合測試 |
| `tests/` | 單元測試 |
| `docs/implementation-notes.md` | 實作決策紀錄 |
