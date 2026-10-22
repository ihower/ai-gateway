#!/usr/bin/env bash
# 啟動 vLLM embedding server，並掛上 prompt middleware（預設 0.0.0.0:12351）
set -euo pipefail
cd "$(dirname "$0")"

# vllm 的 console script 不會把 cwd 放進 sys.path，middleware 模組要靠 PYTHONPATH 才找得到
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ps 顯示的 process 名稱前綴，例如 voyage-4-nano::EngineCore（vLLM 預設為 VLLM）
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-voyage-4-nano}"

# --gpu N：指定 CUDA 裝置編號（設定 CUDA_VISIBLE_DEVICES），其餘參數原樣傳給 vllm serve
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) export CUDA_VISIBLE_DEVICES="${2:?--gpu 需要指定 CUDA 裝置編號}"; shift 2 ;;
    --gpu=*) export CUDA_VISIBLE_DEVICES="${1#--gpu=}"; shift ;;
    *) args+=("$1"); shift ;;
  esac
done

# --enforce-eager：vLLM 0.16 的 VoyageQwen3BidirectionalEmbedModel.forward(*args, **kwargs)
# 讓 torch.compile 抓不到動態 shape，capture CUDA graph 時 IndexError；model card 範例也用 eager

exec uv run vllm serve voyageai/voyage-4-nano \
  --served-model-name voyage-4-nano \
  --runner pooling --convert embed \
  --hf-overrides '{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}' \
  --pooler-config '{"pooling_type":"MEAN"}' \
  --trust-remote-code --dtype bfloat16 --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.15}" \
  --api-server-count "${API_SERVER_COUNT:-1}" \
  --middleware prompt_middleware.VoyagePromptMiddleware \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-12351}" \
  --disable-uvicorn-access-log \
  "${args[@]}"
