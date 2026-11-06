# ai-gateway

[ihower](https://ihower.tw) 的地端 AI 推論工具箱：整合 embedding、reranking 與 decision 服務，包含 Rust API gateway、NVIDIA CUDA / AMD ROCm 推論後端，以及以 Python FastAPI 實作的 fallback 服務。

當地端 GPU 後端不可用時，fallback 提供以下替代方案：

- **Embedding**：模型體積小，可暫時用 CPU 推論維持服務，但吞吐量較低。
- **Reranker**：改用 Cohere API。分數尺度與地端模型不同，不宜沿用相同門檻，但仍可用於相關性排序。
- **Decision**：改用 Jev API，維持相同的 SystemOne 請求與回應格式。

## Gateway 功能

以 Rust 實作的 API gateway，前置 embedding（OpenAI 格式）、reranker（Cohere 格式）與 decision（Jev / SystemOne 格式）三類後端。完整規格見 [SPEC.md](SPEC.md)。

- Embedding / reranker：多筆 input / documents 平分到各後端並行處理，再合併回應
- Rerank 長文件切段：超過設定字元數的 document 切成重疊的多段分別算分，取最高分（`/v2/rerank` 與 `/decision/v2/rerank` 都支援）
- Decision：整個 request 輪流分派到單一後端
- 啟動與 reload 時實際打一次各後端 API，連不上的後端停用到下次 reload
- `ai-gateway reload` 熱重載後端清單，進行中的 request 用舊清單完成

## 後端專案

GPU 後端放在 `backends/`，Python FastAPI fallback 放在根目錄的 `fallback-fastapi/`，由此 Git repo 統一管理：

| 子目錄 | 用途 |
|--------|------|
| [fallback-fastapi](#fallback) | CPU embedding 與 Cohere / Jev fallback |
| [voyage-4-nano-cuda](backends/voyage-4-nano-cuda/README.md) | NVIDIA CUDA embedding |
| [voyage-4-nano-rocm](backends/voyage-4-nano-rocm/README.md) | AMD ROCm embedding |
| [llama-nemotron-rerank-1b-v2-cuda](backends/llama-nemotron-rerank-1b-v2-cuda/README.md) | NVIDIA CUDA reranker |
| [llama-nemotron-rerank-1b-v2-rocm](backends/llama-nemotron-rerank-1b-v2-rocm/README.md) | AMD ROCm reranker |

各後端保留獨立的 `pyproject.toml`、`uv.lock` 與 `.venv`，Python、vLLM 和 GPU 套件版本不可混用。請先進入對應子目錄，再依其 README 安裝與啟動；gateway 的建置方式不變。模型快取與原 repo 的虛擬環境不隨整併搬移。


## 建置

```bash
./build.sh    # cargo build --release，並複製到根目錄的 ./ai-gateway
```

根目錄隨 Git 提供的 `ai-gateway` binary 適用於 **Linux x86-64（Intel / AMD 64 位元），需要 glibc ≥ 2.34**，採動態連結。符合條件的部署端 `git pull` 後即可直接執行，不需要安裝 Rust；gateway 本身不需要 CUDA 或 ROCm，GPU 執行環境由各推論後端負責。

此 binary 無法直接在 macOS、Windows、Linux ARM64 或使用 musl 的 Alpine Linux 執行。其他平台需另行確認原始碼與依賴的相容性並自行建置。

修改 Rust 程式碼後，commit 前請先執行 `./build.sh`，讓 binary 與程式碼一致。

## 使用

```bash
# 首次使用：複製範例並依部署環境修改；config.yaml 已被 Git 忽略
cp config.yaml.example config.yaml

# 啟動（預設讀 ./config.yaml，可用 --config 或 AI_GATEWAY_CONFIG 指定）
./ai-gateway serve

# 修改 config.yaml 後熱重載，印出後端增減與被停用的後端
./ai-gateway reload

# reload 前先檢查設定檔並探測後端，不影響執行中的 gateway
./ai-gateway check

# 查看目前各後端狀態
./ai-gateway status
```

`reload` / `status` 的 exit code：`0` 成功；`1` 設定錯誤或 reload 失敗（舊設定保留）；`2` 連不到執行中的 gateway。

升級 binary 或修改監聽位址時，送 `SIGTERM`：gateway 停止接受新連線，等進行中的 request 完成（最多 `shutdown_timeout_secs`）後結束。

## Fallback

各 service 的後端都失效時，gateway 會改送該 service 的 `fallback`（見 SPEC §8.6）。Fallback 是獨立的 Python process（`fallback-fastapi/server.py`），三種 service 共用：

| Service | Fallback |
|---------|----------|
| embedding | CPU 上以 Sentence Transformers 跑 `voyageai/voyage-4-nano`；回應格式、prompt（`input_type`）、`dimensions` 與 `usage` 都與 vLLM 後端一致 |
| reranker | 轉送 Cohere `rerank-v4.0-fast`；回應改成 vLLM 的形狀（帶 `document.text`），沒有 `usage`，分數尺度與 nemotron 不同 |
| decision | 轉送 Jev `jev-1.14.0`（SystemOne 格式），回應原樣傳回 |

API key 放在 `fallback-fastapi/.env`（`COHERE_API_KEY`、`JEV_API_KEY`），以 override 方式載入；client 帶的 `Authorization` 不會轉送給雲端 API。

```bash
cd fallback-fastapi
cp -n .env.example .env       # 首次設定：填入需要的 API key；不覆蓋既有 .env
uv run server.py              # 預設 127.0.0.1:12359；FALLBACK_DTYPE=fp32 可改用 float32
uv run pytest -q              # 對照 vLLM 後端驗證，並實際呼叫 Cohere / Jev 少量請求
```

Gateway 啟動或 reload 時，fallback 必須已在執行，否則探測失敗會被停用，要等下次 `reload` 才會重新探測。

## 日誌

Gateway 以 JSON 格式輸出到 stdout，沒有寫入檔案；需要保存時請自行導向檔案或交給 systemd / journald。每個 request 會寫一行 access log（`"target":"access"`），包含狀態碼、耗時、呼叫過的後端與結果，以及是否用到 fallback，欄位見 SPEC §9.1。

```bash
./ai-gateway serve | jq -c 'select(.target == "access") | .fields'   # 只看 access log
RUST_LOG=info,access=off ./ai-gateway serve              # 關閉 access log
```

## 端點

| Method | Path | 說明 |
|--------|------|------|
| POST | `/v1/embeddings` | OpenAI 格式 embedding |
| POST | `/v2/rerank` | Cohere v2 格式 rerank |
| POST | `/v1/systemone` | Decision（Jev / SystemOne 格式） |
| GET | `/healthz`、`/readyz` | 健康檢查 |
| GET | `/admin/backends`、`/admin/config`、`/metrics` | 管理埠（唯讀） |

## 部署

部署到另一台 Linux server（systemd、設定檔放置、更新流程）見 [docs/deploy.md](docs/deploy.md)。

## 測試

```bash
cargo test
```
