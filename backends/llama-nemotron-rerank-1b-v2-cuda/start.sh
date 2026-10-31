#!/usr/bin/env bash
# 啟動 nvidia/llama-nemotron-rerank-1b-v2 的 vLLM reranker HTTP API（/rerank、/score）
# 可用環境變數覆寫設定，額外參數會直接傳給 vllm serve，例如：
#   PORT=8000 ./start.sh --max-model-len 4096
# 也可用 --gpu 指定 CUDA 裝置編號、--port 指定監聽 port，例如：
#   ./start.sh --gpu 1 --port 8001
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-nvidia/llama-nemotron-rerank-1b-v2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-12361}"
DTYPE="${DTYPE:-auto}"
# 直接指定 KV cache 大小（每 token 32KiB）。128M（大寫 M = MiB）剛好容納 4096 tokens，
# 即一組 max_model_len 的序列；調高 MAX_MODEL_LEN 時要一併調高（每 4096 tokens 128M）
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-128M}"
# 設定 KV_CACHE_MEMORY_BYTES 後此比例不影響分配，只用於 vLLM 啟動時檢查「剩餘記憶體 ≥ 總量 × 此比例」
# 實際用量約 權重 2.3GB + KV 128MB + 運算空間，16GB GPU 取 0.2（約 3.2GB）
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.2}"
# 每組 query + 文件取 4096 tokens（模型官方上限 8192；config 的 131072 只是沿用 Llama 3.2 的 RoPE 設定）
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/nemotron-rerank.jinja}"

# 解析 --gpu / --port（支援 `--opt 值` 與 `--opt=值`），其餘參數原樣傳給 vllm serve
VLLM_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { echo "--gpu 需要指定 CUDA 裝置編號" >&2; exit 1; }
      export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --gpu=*)
      export CUDA_VISIBLE_DEVICES="${1#*=}"; shift ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port 需要指定 port" >&2; exit 1; }
      PORT="$2"; shift 2 ;;
    --port=*)
      PORT="${1#*=}"; shift ;;
    *)
      VLLM_ARGS+=("$1"); shift ;;
  esac
done

# score template 是必要的，缺少時 query/passage 格式不會套用，分數會錯
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "找不到 score template：$CHAT_TEMPLATE" >&2
  exit 1
fi

# 程序名稱前綴，nvidia-smi / ps 會顯示為 nemotron-rerank::EngineCore（取代預設的 VLLM）
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-nemotron-rerank}"

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
  "${VLLM_ARGS[@]}"
