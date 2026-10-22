# 部署到 Linux server

本文件說明如何在另一台 Linux server 上，從 git repo 取得程式碼後部署 ai-gateway 與 fallback server。

做法：使用 repo 根目錄已編譯好的 `ai-gateway` binary（不需要在 server 上安裝 Rust），兩個 process 都交給 systemd 管理，設定檔放在 repo 外面，避免 `git pull` 時發生衝突。

| Process | 說明 | 預設位址 |
|---------|------|----------|
| `ai-gateway` | Rust gateway | `0.0.0.0:54321`（API）、`54322`（管理埠） |
| `ai-gateway-fallback` | `fallback-fastapi/server.py`：CPU embedding、轉送 Cohere / Jev | `127.0.0.1:12359` |

以下的 `<user>` 請換成執行服務的使用者名稱。

## 1. 安裝工具（只需一次）

```bash
# uv（執行 fallback server）
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install curl jq
```

## 2. 取得程式碼

```bash
git clone <repo-url> ~/ai-gateway
cd ~/ai-gateway
./ai-gateway --help                   # 確認 binary 能執行
cd fallback-fastapi && uv sync                # 依 uv.lock 安裝；torch 使用 CPU 版的 index
```

根目錄的 `ai-gateway` 是開發機用 `./build.sh` 編譯後 commit 進來的（x86-64 Linux，需要 glibc ≥ 2.34，例如 Ubuntu 22.04+、Debian 12+、RHEL 9+）。可用 `ldd --version` 查看 server 的 glibc 版本。若無法執行（架構或 glibc 不符），改在 server 上安裝 Rust 自行編譯：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install build-essential
./build.sh                            # 編譯並覆蓋根目錄的 ai-gateway（不要 commit 這份）
```

## 3. git 不會帶過去的東西

### API key：`fallback-fastapi/.env`

`.env` 在 `.gitignore` 內，需要自行從原本的機器複製過去，並限制權限：

```bash
scp fallback-fastapi/.env <server>:~/ai-gateway/fallback-fastapi/.env
ssh <server> chmod 600 ~/ai-gateway/fallback-fastapi/.env
```

內容為 `COHERE_API_KEY` 與 `JEV_API_KEY`。fallback server 以 override 方式載入，`.env` 的值優先於既有的環境變數。

### 模型：voyage-4-nano（約 670 MB）

fallback server 第一次啟動時會自動從 Hugging Face 下載。建議先下載好，避免第一次啟動等太久，以及 gateway 啟動時 fallback 還沒就緒：

```bash
cd ~/ai-gateway/fallback-fastapi
uv run python -c "from sentence_transformers import SentenceTransformer as S; S('voyageai/voyage-4-nano', trust_remote_code=True)"
```

## 4. 設定檔放在 repo 外面

`config.yaml` 有進 git，若直接在 server 上修改，之後 `git pull` 會衝突。請複製一份到 repo 外面再修改：

```bash
mkdir -p ~/ai-gateway-config
cp ~/ai-gateway/config.yaml ~/ai-gateway-config/config.yaml
```

需要確認的項目：

- **後端位址**：預設都是 `127.0.0.1:*`，請確認新 server 連得到。
- **fallback 位址**：`127.0.0.1:12359`，前提是 fallback server 與 gateway 在同一台機器上（照本文件部署即是如此）。
- **`control_socket`**：`run/control.sock` 是相對於設定檔所在目錄，會放在 `~/ai-gateway-config/run/`，目錄會自動建立。
- **`admin_listen`**：預設為 `0.0.0.0:54322`，會綁定所有網路介面。管理埠沒有驗證，建議改成 `127.0.0.1:54322`，或用防火牆擋住。

執行 CLI（`status`、`reload`、`check`）時也要能找到這份設定，建議寫進 `~/.bashrc`：

```bash
export AI_GATEWAY_CONFIG=~/ai-gateway-config/config.yaml
```

## 5. systemd 服務

gateway 啟動時會探測各後端，包括 fallback。如果 fallback 還沒載入完模型，探測會失敗，該 fallback 會停用，直到下次 `reload`。`After=` 只會等 process 啟動，不會等它準備好，因此 gateway 的 `ExecStartPre` 會先輪詢 fallback 的 `/health`。

### `/etc/systemd/system/ai-gateway-fallback.service`

```ini
[Unit]
Description=ai-gateway fallback (CPU embedding / Cohere / Jev)
After=network-online.target
Wants=network-online.target

[Service]
User=<user>
WorkingDirectory=/home/<user>/ai-gateway/fallback-fastapi
ExecStart=/home/<user>/.local/bin/uv run server.py --host 127.0.0.1 --port 12359
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### `/etc/systemd/system/ai-gateway.service`

```ini
[Unit]
Description=ai-gateway
After=network-online.target ai-gateway-fallback.service
Wants=network-online.target ai-gateway-fallback.service

[Service]
User=<user>
Environment=AI_GATEWAY_CONFIG=/home/<user>/ai-gateway-config/config.yaml
# 等 fallback 載入完模型，最多 120 秒；逾時仍會啟動 gateway（fallback 會被停用）
ExecStartPre=/bin/sh -c 'for i in $(seq 120); do curl -sf http://127.0.0.1:12359/health >/dev/null && exit 0; sleep 1; done; exit 0'
ExecStart=/home/<user>/ai-gateway/ai-gateway serve
ExecReload=/home/<user>/ai-gateway/ai-gateway reload
Restart=always
RestartSec=5
TimeoutStopSec=70

[Install]
WantedBy=multi-user.target
```

- `TimeoutStopSec=70` 略大於設定檔的 `shutdown_timeout_secs: 60`。restart 時 systemd 送 `SIGTERM`，gateway 會等進行中的 request 完成後才結束。
- 如果等 fallback 逾時，gateway 仍會啟動，但 fallback 會被停用；等 fallback 起來後執行 `sudo systemctl reload ai-gateway`。

### 啟用

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-gateway-fallback ai-gateway
```

## 6. 確認

```bash
~/ai-gateway/ai-gateway status     # 三個 service 的後端與 fallback 狀態
curl -s localhost:54321/readyz
```

三個 service 的 fallback（`127.0.0.1:12359`）都應該是 `enabled`。若 reranker / decision 的 fallback 是 `disabled`，通常是 `.env` 沒有放好，或 server 連不到外網。

## 7. Log

兩個服務的輸出都會進 journald，會自動保存與輪替：

```bash
journalctl -u ai-gateway -f                                                   # 即時查看
journalctl -u ai-gateway -o cat | jq -c 'select(.target=="access") | .fields' # 只看 access log
journalctl -u ai-gateway -o cat | jq -c 'select(.fields.fallback == true)'    # 用到 fallback 的 request
journalctl -u ai-gateway-fallback -f
```

access log 的欄位說明見 [SPEC.md](../SPEC.md) §9.1。

## 8. 之後更新

```bash
cd ~/ai-gateway && git pull           # 會一併更新根目錄的 ai-gateway binary
(cd fallback-fastapi && uv sync)
sudo systemctl restart ai-gateway-fallback ai-gateway   # binary 或 Python 程式有變動時要 restart
```

只修改設定檔時不需要 restart，用 reload 即可，不會中斷服務：

```bash
~/ai-gateway/ai-gateway check     # 先檢查設定檔與後端探測結果，不影響執行中的 gateway
sudo systemctl reload ai-gateway
```

注意：`config.yaml` 在 repo 裡的版本更新時，不會自動套用到 `~/ai-gateway-config/config.yaml`，需要自行比對合併（例如 `diff ~/ai-gateway/config.yaml ~/ai-gateway-config/config.yaml`）。

## 9. 新 server 需要確認的事項

- **glibc / 架構**：根目錄的 binary 是 x86-64、需要 glibc ≥ 2.34；不符時改在 server 上編譯（見第 2 節）。
- **網路**：server 需要能連到 `api.cohere.com` 與 `api.typesafe.ai`，否則 reranker 與 decision 的 fallback 探測會失敗。
- **資源**：embedding fallback 在本機 CPU 上推論，約佔 1.5 GB 記憶體；實際使用時會與 gateway 搶 CPU。
