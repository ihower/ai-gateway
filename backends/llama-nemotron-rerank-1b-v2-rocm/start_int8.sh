#!/usr/bin/env bash
# 以 INT8 啟動：讀取 quantize_int8.py 產生的 checkpoint，線性層權重為 INT8，輸入由 vLLM 動態量化。
# 第一次使用前先執行：uv run python quantize_int8.py
# 參數與環境變數同 start.sh，例如：
#   ./start_int8.sh --port 8001
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-$SCRIPT_DIR/models/llama-nemotron-rerank-1b-v2-int8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/llama-nemotron-rerank-1b-v2}"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "找不到 INT8 checkpoint：$MODEL" >&2
  echo "請先執行：uv run --project \"$SCRIPT_DIR\" python \"$SCRIPT_DIR/quantize_int8.py\"" >&2
  exit 1
fi

# API 的 model 名稱維持和 BF16 版相同，呼叫端不用改
# --enforce-eager：實測 compile 模式的長文件 rerank 約慢一倍（7.04 秒 vs 3.64 秒）
exec "$SCRIPT_DIR/start.sh" --enforce-eager --served-model-name "$SERVED_MODEL_NAME" "$@"
