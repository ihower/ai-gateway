#!/usr/bin/env bash
# 啟動 vLLM embedding server，並掛上 prompt middleware（預設 0.0.0.0:12351）
set -euo pipefail
cd "$(dirname "$0")"

# vllm 的 console script 不會把 cwd 放進 sys.path，middleware 模組要靠 PYTHONPATH 才找得到
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# --enforce-eager：vLLM 0.16 的 VoyageQwen3BidirectionalEmbedModel.forward(*args, **kwargs)
# 讓 torch.compile 抓不到動態 shape，capture CUDA graph 時 IndexError；model card 範例也用 eager

exec uv run vllm serve voyageai/voyage-4-nano \
  --served-model-name voyage-4-nano \
  --runner pooling --convert embed \
  --hf-overrides '{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}' \
  --pooler-config '{"pooling_type":"MEAN"}' \
  --trust-remote-code --dtype bfloat16 --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.3}" \
  --api-server-count "${API_SERVER_COUNT:-1}" \
  --middleware prompt_middleware.VoyagePromptMiddleware \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-12351}" \
  --disable-uvicorn-access-log \
  "$@"
