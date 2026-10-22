# AI Gateway 規格書

以 Rust 實作的高效能 API Gateway，前置三類推論後端：

| 服務 | 對外格式 | 分流策略 |
|------|----------|----------|
| Embedding | OpenAI `/v1/embeddings` | 多筆 input **平分**到各後端，合併回應 |
| Reranker | Cohere `/v1/rerank`、`/v2/rerank` | documents **平分**到各後端，依分數合併排序 |
| Decision | Jev 格式（= Clef SystemOne，`/v1/systemone`） | 整個 request **輪流**分派到單一後端 |

---

## 1. 整體架構

```
            ┌──────────────────────── ai-gateway ────────────────────────┐
client ───▶ │ axum router ─▶ handler ─▶ splitter ─▶ BackendPool (ArcSwap) │ ─▶ backends
            │                    ◀── merger ◀── 並行 fan-out (tokio) ◀──   │
            └────────────────────────────────────────────────────────────┘
```

- Runtime：`tokio` + `axum`（server）+ `reqwest`（HTTP client，keep-alive 連線池）
- 設定：`serde_yaml` 讀取 `config.yaml`
- 後端池：`arc-swap` 保存目前的 `Arc<BackendPool>`，reload 時原子替換
- 日誌：`tracing`，JSON 格式輸出到 stdout；每個 request 一行 access log（見 §9.1）

---

## 2. 設定檔 `config.yaml`

App 啟動時載入；路徑預設 `./config.yaml`，可用 `--config <path>` 或環境變數 `AI_GATEWAY_CONFIG` 指定。

```yaml
server:
  listen: "0.0.0.0:8080"
  admin_listen: "127.0.0.1:8081"     # 管理端點（唯讀查詢），只綁 localhost
  control_socket: "/run/ai-gateway/control.sock"  # CLI reload / status 使用，權限 0600
  shutdown_timeout_secs: 60          # 優雅關閉時等待 in-flight request 的上限

defaults:
  connect_timeout_ms: 1000
  request_timeout_ms: 30000
  max_retries: 1                     # 後端失敗時，改送其他後端的次數
  unhealthy_threshold: 3             # 連續失敗幾次才標記為 unhealthy
  unhealthy_cooldown_secs: 600       # 標記為 unhealthy 後跳過該後端的時間（10 分鐘）

probe:                               # 啟動 / reload 時的探測：對每個後端實際打一次 API（見 §8.2）
  timeout_ms: 10000                  # 單一後端探測逾時（含一次推論時間）

services:
  embedding:
    probe_model: null                # 探測請求帶的 model 欄位；null = 不帶
    backends:
      - url: "http://127.0.0.1:12351"
      - url: "http://127.0.0.1:12352"
    fallback:                        # 選填，三種 service 都可設定；見 §8.6
      url: "http://127.0.0.1:12359"
      request_timeout_ms: 120000     # 覆寫 request_timeout_ms

  reranker:
    probe_model: null
    backends:
      - url: "http://127.0.0.1:12361"
        weight: 1                    # 範例採相同權重，可依後端容量調整
      - url: "http://127.0.0.1:12362"
        weight: 1

  decision:
    probe_model: null
    backends:
      - url: "http://127.0.0.1:12371"
        weight: 1                    # 範例採相同權重，可依後端容量調整
      - url: "http://127.0.0.1:12372"
        weight: 1
```

- 每個 backend 可選填 `weight`（預設 1），用於拆分比例與輪流權重。
- 每個 service 可覆寫 `defaults` 的任一欄位（例如 reranker 的 `request_timeout_ms` 較長）。
- 啟動時驗證：URL 格式正確且為 `http://`（未編入 TLS，不支援 `https://` 後端）、每個 service 至少一個 backend；驗證失敗則拒絕啟動（reload 時則保留舊設定，見 §7）。
- 「後端連不上」**不算**設定錯誤：該後端會被停用（`disabled`），不影響啟動或 reload（見 §8.2）。

---

## 3. 對外 API 端點總覽

### 3.1 業務端點（`server.listen`）

| Method | Path | 說明 |
|--------|------|------|
| POST | `/v1/embeddings` | OpenAI 格式 embedding |
| POST | `/v1/rerank` | Cohere v1 格式 rerank |
| POST | `/v2/rerank` | Cohere v2 格式 rerank |
| POST | `/v1/systemone` | Decision API（Jev / SystemOne 格式，見 §6） |
| GET  | `/healthz` | Liveness，永遠回 `200 {"status":"ok"}`（draining 時回 `503`） |
| GET  | `/readyz` | Readiness，每個 service 至少有一個啟用中的健康後端才回 `200` |

### 3.2 管理端點（`server.admin_listen`，僅 localhost）

| Method | Path | 說明 |
|--------|------|------|
| GET  | `/admin/backends` | 列出各後端狀態：`enabled` / `disabled` / `unhealthy`、停用原因、in-flight 數、累計請求/錯誤數 |
| GET  | `/admin/config` | 回傳目前生效的設定與載入時間、設定版本號 |

管理端點只供查詢；**重新載入一律透過 CLI**（§3.3）。

### 3.3 CLI

同一個 binary 提供以下子指令；除 `serve` 外，都是對執行中的 gateway 或設定檔的一次性操作。

| 指令 | 說明 |
|------|------|
| `ai-gateway serve [--config <path>]` | 啟動 gateway |
| `ai-gateway reload [--json]` | 通知執行中的 gateway 重新載入設定，等待完成後印出後端差異與被停用的後端（§7.1） |
| `ai-gateway check [--config <path>]` | 只驗證設定檔並探測所有後端，印出結果，**不影響**執行中的 gateway；適合在 reload 前先檢查 |
| `ai-gateway status [--json]` | 印出執行中 gateway 的設定版本與各後端狀態 |

- `reload` / `status` 透過 `server.control_socket` 與 gateway 溝通，不經過網路埠。
- 未指定 `--config` 時，`reload` / `status` 讀取 `--config` / `AI_GATEWAY_CONFIG` / `./config.yaml` 中的 `control_socket` 來找到 gateway。
- Exit code：`0` 成功；`1` 設定錯誤或 reload 失敗（舊設定保留）；`2` 連不到執行中的 gateway。
- 有後端被停用時 reload 仍算成功（exit `0`），但會在輸出中標出。

---

## 4. Embedding API — `POST /v1/embeddings`

### 4.1 Request（OpenAI 格式）

```json
{
  "model": "bge-m3",
  "input": ["text a", "text b", "text c", "text d", "text e"],
  "encoding_format": "float",
  "dimensions": 1024,
  "user": "optional"
}
```

`input` 支援 OpenAI 的四種形式：

| 形式 | 範例 | 視為幾筆 |
|------|------|---------|
| 字串 | `"hello"` | 1 |
| 字串陣列 | `["a","b"]` | N |
| token 陣列 | `[123, 456]` | 1（**不可**拆） |
| token 陣列的陣列 | `[[1,2],[3,4]]` | N |

### 4.2 分流規則

設 N = input 筆數，H = 目前健康後端數。

1. **N = 1**：以輪流（weighted round-robin）挑一個後端，透明轉發，回應原樣回傳。
2. **N > 1**：拆成 K = min(H, N) 份，即使只有 2 筆也會拆到兩台
   - 依**連續區段**切分，各份筆數差不超過 1（有 weight 時依比例），例：N=5, K=2 → `[0..3]`、`[3..5]`
   - 起始後端依輪流指標輪替，避免多出來的那筆總是落在同一台
   - 各份**並行**送出，其餘欄位（`model`、`encoding_format`、`dimensions`、`user`）原樣帶入每個子請求
3. **合併**：
   - `data`：依原始順序串接，每筆 `index` 改寫為原始陣列中的位置（`offset + 子回應 index`）
   - `usage` 內的數值欄位（`prompt_tokens`、`total_tokens` 等）：加總
   - 其餘頂層欄位（`model`、`object`，以及後端額外回傳的欄位，例如 vLLM 的 `created`）：取第一個子回應的值
   - `id`：若後端回應含 `id`，改為 gateway 產生的新 UUID

### 4.3 Response（OpenAI 格式）

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.01, -0.02]},
    {"object": "embedding", "index": 1, "embedding": [0.03, 0.04]}
  ],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

`encoding_format: "base64"` 時 `embedding` 為字串，合併邏輯相同（每筆獨立，不需解碼）。

---

## 5. Reranker API — `POST /v1/rerank`、`POST /v2/rerank`

### 5.1 Request（Cohere 格式）

```json
{
  "model": "bge-reranker-v2-m3",
  "query": "What is the capital of France?",
  "documents": ["Paris is ...", "Berlin is ...", "..."],
  "top_n": 3,
  "return_documents": true,
  "max_tokens_per_doc": 4096
}
```

- `documents` 可為字串陣列；v1 亦接受 `{"text": "..."}` 物件陣列，原樣轉送。
- v1 / v2 的路徑原樣轉送到後端（`/v1/rerank` → 後端 `/v1/rerank`）。

### 5.2 分流規則

設 N = documents 數，H = 健康後端數。

1. **N = 1 或只有一個健康後端**（K = 1）：輪流挑一個後端，透明轉發。
2. **否則**：依 §4.2 相同規則把 documents 切成 K 份連續區段並行送出
   - 每個子請求的 `query`、`model` 與其他欄位相同
   - `top_n`：子請求帶 `min(top_n, 該份筆數)`（全域前 n 名必然落在各份的前 n 名之中，可減少回傳量）；未指定則不帶
   - `top_n: 0` 視同未指定，回傳全部（與 vLLM 行為一致）
3. **合併**：
   - 每筆結果的 `index` 改寫為 `offset + 子回應 index`
   - 全部結果依 `relevance_score` **由高到低**排序（分數相同時 index 小者在前，確保結果穩定）
   - 若有 `top_n`，取前 `top_n` 筆
   - `return_documents: true` 時，`document` 欄位隨結果一起保留
   - `id`：若後端回應含 `id`，改為 gateway 產生的新 UUID（各份的 id 已無意義）；後端沒有回 `id` 則不加
   - `meta.billed_units.search_units`：加總（若後端有回傳）；`meta.api_version` 取第一個子回應

> **前提**：所有 reranker 後端跑同一個模型，`relevance_score` 可直接比較。

### 5.3 Response（Cohere 格式）

```json
{
  "id": "6f1c...",
  "results": [
    {"index": 0, "relevance_score": 0.98, "document": {"text": "Paris is ..."}},
    {"index": 7, "relevance_score": 0.41, "document": {"text": "..."}}
  ],
  "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 2}}
}
```

---

## 6. Decision API — `POST /v1/systemone`

Jev（TypeSafe，`https://api.typesafe.ai/v1/systemone`）與 Clef 使用相同的 **SystemOne** 格式；可接入提供 SystemOne 相容 API 的後端，例如 llama.cpp。

### 6.1 Request

```json
{
  "model": "clef-flash",
  "state": "Checkout has been failing for every customer for the last hour.",
  "questions": {
    "urgent":   {"type": "noul",   "instructions": "Is this support request urgent?"},
    "team":     {"type": "choice", "instructions": "Which team should handle this request?",
                 "criteria": {"billing": "Payments", "technical": "Outages"}},
    "severity": {"type": "score",  "instructions": "How severe is the customer impact?",
                 "criteria": ["No impact", "Minor", "Major", "Critical"]}
  }
}
```

| 欄位 | 型別 | 說明 |
|------|------|------|
| `model` | string | 原樣轉送 |
| `state` | string \| object \| array | 要判斷的內容；一個 request 只有一個 state |
| `questions` | object | 題目名稱 → 題目定義，至少一題 |
| `questions.*.type` | `noul` \| `choice` \| `score` | `noul` 為是非題（回傳機率） |
| `questions.*.instructions` | string | 題目敘述 |
| `questions.*.criteria` | object \| array | `choice`：選項名稱 → 說明（≥ 2 個）；`score`：等級陣列（≥ 2 個）；`noul`：可選，只能有 `true` / `false` |

### 6.2 Response

```json
{
  "model": "clef-flash.gguf",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.95},
    "team":   {"type": "choice", "choice": "technical", "confidence": 0.9, "probabilities": {"billing": 0.1, "technical": 0.9}}
  },
  "usage": {"input_tokens": 346, "output_tokens": 0}
}
```

### 6.3 分流規則

- 整個 request 送到**單一後端**，以加權輪流挑選，跳過 `disabled` / `unhealthy` 者。
- **不拆分**：所有 questions 共用同一個 state，同一份 state 只需 prefill 一次，拆開送反而變慢。
- Gateway **不解析也不驗證** body，request/response body、status code、`Content-Type` 原樣轉發（以串流方式搬移，不做 JSON 反序列化）；格式錯誤由後端回 `4xx`。

---

## 7. 優雅重開（Graceful Reload / Restart）

需求：後端可能增加或減少；重開時正在跑的 request 要順利完成，之後的 request 用新後端分流。

### 7.1 熱重載（主要方式，不中斷服務）

用 CLI 觸發（見 §3.3）：

```bash
ai-gateway reload
```

- CLI 透過 `server.control_socket`（Unix domain socket）通知執行中的 gateway 重新載入，**等待結果**後印出差異並以 exit code 回報成功與否，可直接用在 script / systemd `ExecReload`。
- gateway 同時處理 `SIGHUP`（例如 `kill -HUP`），流程相同，但結果只寫進日誌。
- 不提供 HTTP 的 reload 端點。

流程：

1. 重新讀取並驗證 `config.yaml`；**驗證失敗則保留舊設定**，記錄錯誤，CLI 印出錯誤並以 exit code `1` 結束。
2. **並行探測**新設定中的所有後端（含沿用的舊後端），連不上的直接標記為 `disabled`（見 §8.2）。探測期間舊 pool 照常服務，總耗時上限約為 `probe.timeout_ms`。
3. 建立新的 `BackendPool`，以 `ArcSwap::store` 原子替換。
4. **進行中的 request**：在開始時已 `load()` 取得舊 pool 的 `Arc`，整個生命週期（含拆分、重試）都使用舊 pool，因此即使其後端在新設定中被移除，也會照常完成。
5. **新的 request**：一律使用新 pool。
6. 舊 pool 在最後一個參考它的 request 結束後自動釋放（`Arc` 計數歸零）。
7. 後端在新舊設定都存在時（以 URL 為鍵），沿用其統計數字；啟用狀態則以本次探測結果為準。
8. 監聽的 socket 不關閉，不會有連線被拒的空窗。

`ai-gateway reload` 輸出範例：

```
$ ai-gateway reload
reloaded config v3 from /etc/ai-gateway/config.yaml
embedding  + http://127.0.0.1:12353  disabled (connect refused)
reranker   - http://127.0.0.1:12362
decision   (no change)
```

加上 `--json` 時改輸出 JSON：

```json
{
  "version": 3,
  "changes": {
    "embedding": {"added": ["http://127.0.0.1:12353"], "removed": []},
    "reranker":  {"added": [], "removed": ["http://127.0.0.1:12362"]},
    "decision":  {"added": [], "removed": []}
  },
  "disabled": [
    {"service": "embedding", "url": "http://127.0.0.1:12353", "reason": "connect refused"}
  ]
}
```

> `server.listen`、`server.admin_listen`、`server.control_socket` 的變更無法熱重載，需走 §7.2 重啟流程（`reload` 會印出警告）。

### 7.2 優雅關閉（升級 binary 或更改監聽埠時）

新 process 啟動時同樣先探測所有後端（§8.2），連不上的後端一開始就是 `disabled`。

收到 `SIGTERM` / `SIGINT`：

1. 進入 draining：`/healthz`、`/readyz` 回 `503`，讓上游 LB 停止送流量
2. 停止接受新連線（axum `with_graceful_shutdown`）
3. 等待所有 in-flight request 完成，最多 `shutdown_timeout_secs`；逾時則強制結束並記錄未完成數量
4. 結束 process

搭配 systemd 的 `Restart=always` 或外部 process manager 重新啟動。若要零空窗升級 binary：監聽 socket 設定 `SO_REUSEPORT`，先啟動新 process，再對舊 process 送 `SIGTERM`。

---

## 8. 後端選擇、健康檢查與錯誤處理

### 8.1 後端選擇
- **加權輪流**（smooth weighted round-robin），每個 service 各自一個原子計數器，不看負載。
  - 依各後端的 `weight` 比例分派，例如 `1` / `3` 即每 4 個 request 中 1 個與 3 個；分佈是平滑交錯的，不會先連續打同一台
  - 拆分（embedding / rerank）時，各份筆數也依 `weight` 比例切分；權重相同即平分
- 拆分時的 H（健康後端數）只計算**啟用中且健康**的後端。
- `disabled` 的後端一律不選。
- 啟用中的後端若全部 `unhealthy`，仍嘗試全部（避免因誤判而完全停擺）；設有 fallback 時改送 fallback（§8.6）。
- 某 service 的後端全部 `disabled` → 該 service 的 request 直接回 `503`（`no available backend`），`/readyz` 回 `503`。

### 8.2 後端狀態與健康檢查

每個後端有三種狀態：

| 狀態 | 意義 | 會被選中？ |
|------|------|-----------|
| `enabled` | 正常 | 是 |
| `unhealthy` | 執行期間暫時失敗，冷卻中 | 否（全部不健康時例外，見 §8.1） |
| `disabled` | 啟動 / reload 時連不上 | 否 |

**啟動 / reload 時的探測**

不假設後端的實作（不依賴 `/health` 之類的專屬端點），而是對每個後端**實際打一次該服務的 API**，並行進行：

| Service | 探測請求 |
|---------|----------|
| embedding | `POST /v1/embeddings` `{"input": "ping"}` |
| reranker | `POST /v1/rerank` `{"query": "ping", "documents": ["ping"]}` |
| decision | `POST /v1/systemone` `{"state": "ping", "questions": {"probe": {"type": "noul", "instructions": "Is this a test?"}}}` |

- 若該 service 設定了 `probe_model`，請求中加上 `"model": <probe_model>`
- 在 `probe.timeout_ms` 內回 `2xx`，且回應是可解析的 JSON（embedding 含 `data`、reranker 含 `results`、decision 含 `answers`）→ 通過
- 其餘情況（連不上、逾時、非 `2xx`、回應格式不符）→ 標記為 `disabled`，記錄 `WARN` 日誌與原因（例如 `connect refused`、`timeout`、`HTTP 503`、`invalid response`）
- 這樣可以抓到「port 開著但模型還沒載入好」或「這個 port 跑的不是這種服務」的情況
- 有後端被停用**不會**讓啟動或 reload 失敗；即使某 service 全部停用，gateway 仍會啟動（記錄 `ERROR` 日誌，該 service 回 `503`）

**停用後端不會自動恢復**
- `disabled` 的後端在本次設定生效期間一律不使用，gateway 不會在背景重新探測
- 後端修好後，執行 `ai-gateway reload` 才會重新探測，通過即加回分流
- 可先用 `ai-gateway check` 確認後端已能通過探測，再執行 `reload`

**執行期間的被動健康檢查**
- 連線失敗、逾時或回 `5xx` 算一次失敗；**連續**失敗 `unhealthy_threshold` 次 → 該後端標記為 `unhealthy` `unhealthy_cooldown_secs` 秒
- 成功一次即將連續失敗次數歸零
- 冷卻結束後自動恢復可被選中；若下一次仍失敗，立即再次標記為 `unhealthy`（不必再累積到門檻）

### 8.3 重試
- Embedding 與 Rerank 為冪等運算：子請求失敗（連線錯誤、逾時、`5xx`）時，改送**其他**健康後端，最多 `max_retries` 次
- Decision：只在**連線建立失敗**（request 尚未送出）時重試，避免非冪等操作重複執行
- 後端回 `4xx` 不重試，直接把該錯誤回傳給 client

### 8.4 拆分請求的失敗語意
- 任一子請求在重試後仍失敗 → **整個 request 失敗**，不回傳部分結果
- 若子請求回 `4xx`，原樣回傳第一個 `4xx`（例如 input 過長）
- 其餘情況回 `502`（後端錯誤）或 `504`（逾時）

### 8.5 錯誤格式
- Embedding 依 OpenAI 格式：

  ```json
  {"error": {"message": "all backends failed", "type": "upstream_error", "code": "bad_gateway"}}
  ```

- Rerank 依 Cohere 格式：`{"message": "all backends failed"}`
- Decision：後端錯誤原樣傳遞；gateway 自身錯誤回 `{"error": "..."}`

### 8.6 Fallback

`services.<service>.fallback` 指定一個「最後手段」後端，只在一般後端都失效時使用。`fallback-fastapi/server.py` 同時提供三種：embedding 用 CPU 跑 voyage-4-nano；reranker 轉送 Cohere `rerank-v4.0-fast`；decision 轉送 Jev `jev-1.14.0`。

- 不參與加權輪流與拆分。
- 一般後端全部 `disabled` 或全部 `unhealthy` 時，整個 request 直接送 fallback（不拆分），不再嘗試不健康的一般後端（取代 §8.1「全部不健康仍嘗試全部」）。
- Embedding：子請求在一般後端依 §8.3 重試後仍失敗時，**只有該份**改送 fallback 一次（不計入 `max_retries`）。同一模型的向量可以混用。
- Reranker：fallback 是不同模型，分數尺度不同，不能和一般後端的分數混排。因此拆分後任一份重試失敗時，**整個原始 request** 改送 fallback，回應原樣傳回（不合併）。
- Decision：依 §8.3 只在連線建立失敗時重試；一般後端都連不上時改送 fallback。後端回 5xx 仍原樣傳回，不改送（不保證冪等）。
- 4xx 不會觸發 fallback。
- 啟動 / reload 時也會探測 fallback；探測失敗則停用，行為與沒有 fallback 相同。reranker / decision 的探測會實際呼叫 Cohere / Jev 各一次。執行期間 fallback 標記為 `unhealthy` 時仍會使用。
- `request_timeout_ms` 可單獨覆寫（CPU 推論較慢）。
- fallback 可用時，`/readyz` 視該 service 為可用；`status` 與 `/admin/backends` 以 `fallback` 標示。

---

## 9. 其他

- **驗證**：gateway 不做 API key 驗證，任何能連到 `server.listen` 的 client 都可使用；client 帶的 `Authorization` header 原樣轉送後端。
- **Header**：每個回應加上 `x-request-id`（沿用 client 帶入的值或自動產生），並轉送給後端以便追蹤。
- **Body 大小上限**：預設 32 MB，可設定。
- **可觀測性**：`/metrics`（Prometheus 格式，管理埠）— 各 service/backend 的請求數、錯誤數、延遲分佈、in-flight 數。
- **效能目標**：gateway 自身額外延遲 p99 < 1 ms（不含後端時間）；JSON 合併時只解析需要改寫的欄位（`index`、`usage`），`embedding` 向量以 `serde_json::value::RawValue` 原樣搬移，不做浮點數重新解析與序列化。

### 9.1 Access log

每個 request（`/healthz`、`/readyz` 除外）在回應時寫一行 `INFO` 日誌，`target` 為 `access`、`message` 為 `request`，欄位：

| 欄位 | 說明 |
|------|------|
| `request_id` | `x-request-id`（client 帶的或 gateway 產生的） |
| `client` | client IP |
| `method`、`path`、`status` | 回給 client 的狀態碼 |
| `duration_ms` | gateway 收到 request 到送出回應 header 的時間；decision 以串流回傳，不含 body 傳輸 |
| `items` | embedding 的 input 筆數、rerank 的 documents 筆數；decision 沒有此欄位 |
| `backends` | 依序列出每次後端呼叫與結果，例如 `http://a:1 500, http://b:1 200`；結果為狀態碼或錯誤原因（`timeout`、`connect failed: ...`、`invalid response: ...`）。沒呼叫後端（例如 400、503）時為空字串 |
| `fallback` | 是否呼叫過 fallback 後端 |

範例：

```json
{"timestamp":"2026-01-01T00:00:00Z","level":"INFO","fields":{"message":"request","request_id":"f9fb90d9-...","client":"192.0.2.10","method":"POST","path":"/v1/embeddings","status":200,"duration_ms":1.02,"items":1,"backends":"http://127.0.0.1:12352 500, http://127.0.0.1:12351 500, http://127.0.0.1:12359 200","fallback":true},"target":"access"}
```

不想要 access log 時可設 `RUST_LOG=info,access=off`。

