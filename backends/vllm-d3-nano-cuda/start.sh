#!/usr/bin/env bash
# 以 llama serve 啟動 vllm-d3-nano（Q8_0 GGUF）
# 用 --gpu 指定 CUDA 裝置編號、--port 指定監聽 port，例如：
#   ./start.sh --gpu 0 --port 12371
#   ./start.sh --gpu 1 --port 12372
# 也可用環境變數覆寫（GPU、PORT、HOST、MODEL、CTX），額外參數會直接傳給 llama serve
set -euo pipefail

GPU="${GPU:-0}"
PORT="${PORT:-12371}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-/mnt/data/models/vllm/vllm-d3-nano-Q8_0.gguf}"
CTX="${CTX:-32768}"

# 解析 --gpu / --port（支援 `--opt 值` 與 `--opt=值`），其餘參數原樣傳給 llama serve
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { echo "--gpu 需要指定 CUDA 裝置編號" >&2; exit 1; }
      GPU="$2"; shift 2 ;;
    --gpu=*)
      GPU="${1#*=}"; shift ;;
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

export CUDA_VISIBLE_DEVICES="$GPU"

exec llama serve \
  -m "$MODEL" \
  -c "$CTX" -b "$CTX" -ub "$CTX" \
  -np 1 \
  -fa on \
  -ngl 99 \
  --host "$HOST" --port "$PORT" \
  "${EXTRA_ARGS[@]}"
