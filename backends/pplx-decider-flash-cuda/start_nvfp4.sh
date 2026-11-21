#!/usr/bin/env bash
# 以 vLLM 啟動 pplx-decider-flash-v1.3（NVFP4 W4A4，attention 保留 BF16），提供 SystemOne 格式的 POST /v1/systemone
# 用 --gpu 指定 CUDA 裝置編號、--port 指定監聽 port，例如：
#   ./start_nvfp4.sh --gpu 1 --port 12372
# 也可用環境變數覆寫（GPU、PORT、HOST、MODEL、MAX_MODEL_LEN、GPU_MEMORY_UTILIZATION），額外參數會直接傳給 vllm serve
# 模型目錄由 quantize_nvfp4.py 產生，見 README.md
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-1}"
PORT="${PORT:-12372}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-/mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-pplx-decider-flash-v1.3}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"

# 解析 --gpu / --port（支援 `--opt 值` 與 `--opt=值`），其餘參數原樣傳給 vllm serve
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

if [[ ! -f "$MODEL/decision_config.json" ]]; then
  echo "找不到模型目錄：$MODEL（先用 quantize_nvfp4.py 產生）" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
# middleware 讀模型目錄的 tokenizer 與 decision_config.json
export PPLX_MODEL_DIR="$MODEL"
# vllm 的 console script 不會把 cwd 放進 sys.path，middleware 模組要靠 PYTHONPATH 才找得到
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# ps 顯示的 process 名稱前綴，例如 pplx-decider-flash::EngineCore
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-pplx-decider-flash}"
# 限制 torch.compile / JIT 的平行編譯數，避免第一次啟動時吃光主機 RAM
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"

# --runner pooling：vLLM 內建 D3ForSequenceClassification（Qwen3.5 backbone + readout，noncausal full attention）
# --linear-backend cutlass：用 vLLM 預先編好的 CUTLASS NVFP4 GEMM；auto 會選 FlashInfer，
#   第一次使用要 JIT 編譯 fp4_gemm_cutlass_sm120，耗時久且很吃 RAM
# --no-enable-prefix-caching：noncausal attention 下前綴的 hidden state 依賴後文，不能共用
# --limit-mm-per-prompt：只支援文字，不配置 vision encoder 的 profiling 記憶體
exec uv run --no-sync vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --runner pooling \
  --linear-backend cutlass \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --middleware systemone_middleware.SystemOneMiddleware \
  --host "$HOST" --port "$PORT" \
  --disable-uvicorn-access-log \
  "${EXTRA_ARGS[@]}"
