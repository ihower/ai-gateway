#!/usr/bin/env bash
# 在 distrobox 容器（llama-vulkan-radv）內以 llama-server 啟動 vllm-d3-nano（Q8_0 GGUF，Vulkan）
# 用 --device 指定 Vulkan 裝置、--port 指定監聽 port，例如：
#   ./start.sh --device Vulkan0 --port 12373
# 也可用環境變數覆寫（CONTAINER、DEVICE、PORT、HOST、MODEL、CTX），額外參數會直接傳給 llama-server
set -euo pipefail

CONTAINER="${CONTAINER:-llama-vulkan-radv}"
DEVICE="${DEVICE:-Vulkan0}"
PORT="${PORT:-12373}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-/mnt/data/models/vllm/vllm-d3-nano-Q8_0.gguf}"
CTX="${CTX:-32768}"

# 解析 --device / --port（支援 `--opt 值` 與 `--opt=值`），其餘參數原樣傳給 llama-server
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || { echo "--device 需要指定 Vulkan 裝置（例如 Vulkan0）" >&2; exit 1; }
      DEVICE="$2"; shift 2 ;;
    --device=*)
      DEVICE="${1#*=}"; shift ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port 需要指定 port" >&2; exit 1; }
      PORT="$2"; shift 2 ;;
    --port=*)
      PORT="${1#*=}"; shift ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$MODEL" ]]; then
  echo "找不到模型檔：$MODEL" >&2
  exit 1
fi

exec distrobox enter "$CONTAINER" -- llama-server \
  -m "$MODEL" \
  --device "$DEVICE" \
  -c "$CTX" -b 2048 -ub 512 \
  -np 1 \
  -fa on \
  -ngl 99 \
  --host "$HOST" --port "$PORT" \
  "${EXTRA_ARGS[@]}"
