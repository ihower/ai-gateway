#!/usr/bin/env bash
# 以 Docker（vLLM ROCm 映像）在 AMD RX 9070 XT（gfx1201）上啟動 pplx-decider-flash-v1.3，
# 提供 SystemOne 格式的 POST /v1/systemone（middleware 與 ../pplx-decider-flash-cuda 共用）
# 用 --port 指定對外 port，例如：
#   ./start.sh --port 12373
# 也可用環境變數覆寫（IMAGE、NAME、PORT、MODEL、QUANTIZATION、QUANT_CONFIG、MAX_MODEL_LEN、
# GPU_MEMORY_UTILIZATION、KV_CACHE_MEMORY、MAX_CUDAGRAPH_TOKENS、CUDAGRAPH_MODE、RENDER_DEVICE、CARD_DEVICE、CODE_DIR、CACHE_DIR），額外參數會直接傳給 vllm serve
#
# 預設用 BF16 目錄 + vLLM 線上 FP8 量化（--quantization fp8_per_tensor）；改跑 BF16：
#   QUANTIZATION=none ./start.sh
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.34.0}"
NAME="${NAME:-vllm-rocm-pplx}"
PORT="${PORT:-12373}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/data/models/pplx}"
MODEL="${MODEL:-$MODEL_ROOT/pplx-decider-flash-v1.3-vllm-bf16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-pplx-decider-flash-v1.3}"
QUANTIZATION="${QUANTIZATION:-fp8_per_tensor}"
# 線上量化不處理的層（fnmatch）：readout（score，255 列）在 ROCm hipBLASLt FP8 GEMM 會因
# 「維度須為 16 的倍數」失敗；vision tower 不會用到。兩者保留 BF16
DEFAULT_QUANT_CONFIG='{"ignore": ["score*", "*visual*"]}'
QUANT_CONFIG="${QUANT_CONFIG:-$DEFAULT_QUANT_CONFIG}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# 9070 XT 只有 16 GB，且與 llama-rocm-plamo 共用；vLLM 以「總 VRAM 的比例」計算
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
# 直接指定 KV cache 大小（bytes），繞過 gpu_memory_utilization 的預算估算（9070 XT 與其他服務共用時算不出空間）
# 單一 16384-token sequence 約需 0.5 GiB（8 層 full attention KV）加上 GDN state；預設 1 GiB。設為空字串則交給 vLLM 估算
KV_CACHE_MEMORY="${KV_CACHE_MEMORY-1073741824}"
# prefill CUDA graph 的最大 token 數：預設 512 會吃掉約 3.5 GiB VRAM，16 GB 卡放不下 KV cache
MAX_CUDAGRAPH_TOKENS="${MAX_CUDAGRAPH_TOKENS:-256}"
# CUDA graph 模式：PIECEWISE（vLLM 對 pooling 的預設）或 NONE（只保留 torch.compile）
# 預設 NONE：在 gfx1201 + vLLM 0.34 上，PIECEWISE 會讓輸入隨機輸出 NaN（30 次約 16 次）；NONE 與 --enforce-eager 都不會
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-NONE}"
RENDER_DEVICE="${RENDER_DEVICE:-/dev/dri/renderD129}"
CARD_DEVICE="${CARD_DEVICE:-/dev/dri/card2}"
# middleware、decider_codec、pplx_decider 直接沿用 CUDA 版的程式碼
CODE_DIR="${CODE_DIR:-$(cd ../pplx-decider-flash-cuda && pwd)}"
# torch.compile / Triton / vLLM 的快取，放在主機上讓第二次啟動變快
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/pplx-decider-flash-rocm}"

# 解析 --port（支援 `--port 值` 與 `--port=值`），其餘參數原樣傳給 vllm serve
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

if [[ ! -f "$MODEL/decision_config.json" ]]; then
  echo "找不到模型目錄：$MODEL（BF16 目錄用 ../pplx-decider-flash-cuda/prepare_model.py 建立）" >&2
  exit 1
fi
if [[ ! -f "$CODE_DIR/systemone_middleware.py" ]]; then
  echo "找不到 middleware：$CODE_DIR/systemone_middleware.py" >&2
  exit 1
fi
mkdir -p "$CACHE_DIR"

KV_ARGS=()
if [[ -n "$KV_CACHE_MEMORY" ]]; then
  KV_ARGS=(--kv-cache-memory "$KV_CACHE_MEMORY")
fi

QUANT_ARGS=()
if [[ "$QUANTIZATION" != "none" ]]; then
  QUANT_ARGS=(--quantization "$QUANTIZATION" --quantization-config "$QUANT_CONFIG")
fi

# 模型目錄以相同路徑掛進容器：BF16 目錄內是指向 $MODEL_ROOT 的 symlink
# --runner pooling：vLLM 內建 D3ForSequenceClassification（Qwen3.5 backbone + readout，noncausal full attention）
# --no-enable-prefix-caching：noncausal attention 下前綴的 hidden state 依賴後文，不能共用
# --limit-mm-per-prompt：只支援文字，不配置 vision encoder 的 profiling 記憶體
exec docker run --rm -it --name "$NAME" \
  --user "$(id -u):$(id -g)" \
  --device /dev/kfd \
  --device "$RENDER_DEVICE" --device "$CARD_DEVICE" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  --shm-size 8g \
  -v "$MODEL_ROOT":"$MODEL_ROOT":ro \
  -v "$CODE_DIR":/app/pplx:ro \
  -v "$CACHE_DIR":/cache \
  -e HOME=/cache \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/app/pplx \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PPLX_MODEL_DIR="$MODEL" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e VLLM_PROCESS_NAME_PREFIX=pplx-decider-flash \
  -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -e MAX_JOBS=4 \
  -p "$PORT":8000 \
  --entrypoint vllm \
  "$IMAGE" \
    serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --runner pooling \
    "${QUANT_ARGS[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${KV_ARGS[@]}" \
    --no-enable-prefix-caching \
    --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\", \"max_prefill_cudagraph_capture_tokens\": $MAX_CUDAGRAPH_TOKENS}" \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --middleware systemone_middleware.SystemOneMiddleware \
    --host 0.0.0.0 --port 8000 \
    --disable-uvicorn-access-log \
    "${EXTRA_ARGS[@]}"
