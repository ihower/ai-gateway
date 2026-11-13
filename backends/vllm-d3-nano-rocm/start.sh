#!/usr/bin/env bash
# 以 Docker（llama.cpp server-rocm 映像）啟動 vllm-d3-nano（Q8_0 GGUF，ROCm）
# 用 --port 指定對外 port，例如：
#   ./start.sh --port 12373
# 也可用環境變數覆寫（IMAGE、NAME、PORT、MODEL_DIR、MODEL_FILE、CTX、RENDER_DEVICE、CARD_DEVICE），
# 額外參數會直接傳給 llama-server
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-rocm}"
NAME="${NAME:-llama-rocm}"
PORT="${PORT:-12373}"
MODEL_DIR="${MODEL_DIR:-/mnt/data/models/vllm}"
MODEL_FILE="${MODEL_FILE:-vllm-d3-nano-Q8_0.gguf}"
CTX="${CTX:-32768}"
RENDER_DEVICE="${RENDER_DEVICE:-/dev/dri/renderD129}"
CARD_DEVICE="${CARD_DEVICE:-/dev/dri/card2}"

# 解析 --port（支援 `--port 值` 與 `--port=值`），其餘參數原樣傳給 llama-server
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { echo "--port 需要指定 port" >&2; exit 1; }
      PORT="$2"; shift 2 ;;
    --port=*)
      PORT="${1#*=}"; shift ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$MODEL_DIR/$MODEL_FILE" ]]; then
  echo "找不到模型檔：$MODEL_DIR/$MODEL_FILE" >&2
  exit 1
fi

exec docker run --rm -it --name "$NAME" \
  --user "$(id -u):$(id -g)" \
  --device /dev/kfd \
  --device "$RENDER_DEVICE" --device "$CARD_DEVICE" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  -v "$MODEL_DIR":/models:ro \
  -p "$PORT":8080 \
  "$IMAGE" \
    -m "/models/$MODEL_FILE" \
    -c "$CTX" -b "$CTX" -ub "$CTX" \
    -np 1 -fa on -ngl 99 \
    --host 0.0.0.0 --port 8080 \
    "${EXTRA_ARGS[@]}"
