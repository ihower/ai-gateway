#!/usr/bin/env bash
# 啟動 nvidia/llama-nemotron-rerank-1b-v2 的 vLLM reranker HTTP API（/rerank、/score）
# 可用環境變數覆寫設定，額外參數會直接傳給 vllm serve，例如：
#   PORT=8000 ./start.sh --max-model-len 4096
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-nvidia/llama-nemotron-rerank-1b-v2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-12361}"
DTYPE="${DTYPE:-auto}"
# 直接指定 KV cache 大小（每 token 約 32KB）。rerank 序列一步就算完，同時存活的 token 數
# 受 max_num_batched_tokens（此 GPU 預設 8192，約 256MB）限制，512M 已留兩倍餘裕
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-512M}"
# 設定 KV_CACHE_MEMORY_BYTES 後此比例不影響分配，但 vLLM 啟動時仍會檢查剩餘記憶體是否大於「總量 × 此比例」
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
# 模型官方支援 8192 tokens；config 的 131072 只是沿用 Llama 3.2 的 RoPE 設定
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/nemotron-rerank.jinja}"

# score template 是必要的，缺少時 query/passage 格式不會套用，分數會錯
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "找不到 score template：$CHAT_TEMPLATE" >&2
  exit 1
fi

# 讓 vLLM 能 import 專案目錄下的 rerank_middleware
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec uv run --project "$SCRIPT_DIR" vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --middleware rerank_middleware.DefaultTruncationMiddleware \
  "$@"
