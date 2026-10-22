# ai-gateway

以 Rust 實作的 API gateway，前置 embedding（OpenAI 格式）、reranker（Cohere 格式）與 decision（Jev / SystemOne 格式）三類後端。完整規格見 [SPEC.md](SPEC.md)。

- Embedding / reranker：多筆 input / documents 平分到各後端並行處理，再合併回應
- Decision：整個 request 輪流分派到單一後端
- 啟動與 reload 時實際打一次各後端 API，連不上的後端停用到下次 reload
- `ai-gateway reload` 熱重載後端清單，進行中的 request 用舊清單完成

## 建置

```bash
cargo build --release
```

## 使用

```bash
# 啟動（預設讀 ./config.yaml，可用 --config 或 AI_GATEWAY_CONFIG 指定）
./target/release/ai-gateway serve

# 修改 config.yaml 後熱重載，印出後端增減與被停用的後端
./target/release/ai-gateway reload

# reload 前先檢查設定檔並探測後端，不影響執行中的 gateway
./target/release/ai-gateway check

# 查看目前各後端狀態
./target/release/ai-gateway status
```

`reload` / `status` 的 exit code：`0` 成功；`1` 設定錯誤或 reload 失敗（舊設定保留）；`2` 連不到執行中的 gateway。

升級 binary 或修改監聽位址時，送 `SIGTERM`：gateway 停止接受新連線，等進行中的 request 完成（最多 `shutdown_timeout_secs`）後結束。

## Embedding CPU fallback

兩台 embedding 後端都失效時，gateway 會改送 `services.embedding.fallback`（見 SPEC §8.6）。Fallback 是獨立的 Python process，用 Sentence Transformers 在 CPU 上跑 `voyageai/voyage-4-nano`，回應格式、prompt（`input_type`）、`dimensions` 與 `usage` 都與 vLLM 後端一致：

```bash
cd fallback-fastapi
uv run server.py              # 預設 127.0.0.1:12359；FALLBACK_DTYPE=fp32 可改用 float32
uv run pytest -q              # 對照 vLLM 後端驗證（後端連不上時跳過）
```

Gateway 啟動或 reload 時，fallback 必須已在執行，否則探測失敗會被停用，要等下次 `reload` 才會重新探測。

## 端點

| Method | Path | 說明 |
|--------|------|------|
| POST | `/v1/embeddings` | OpenAI 格式 embedding |
| POST | `/v1/rerank`、`/v2/rerank` | Cohere 格式 rerank |
| POST | `/v1/systemone` | Decision（Jev / SystemOne 格式） |
| GET | `/healthz`、`/readyz` | 健康檢查 |
| GET | `/admin/backends`、`/admin/config`、`/metrics` | 管理埠（唯讀） |

## 測試

```bash
cargo test
```
