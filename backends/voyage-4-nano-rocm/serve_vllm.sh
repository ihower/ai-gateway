#!/usr/bin/env bash
# 在 AMD ROCm GPU（Radeon 8060S / gfx1151）上啟動 vLLM embedding server，並掛上 prompt middleware
# 預設 0.0.0.0:12353（NVIDIA 版 voyage-4-nano-cuda 用 12351，兩者可同時跑）
set -euo pipefail
cd "$(dirname "$0")"

# AMD 文件要求：amd_smi 的 Python binding 在 rocm-sdk-core 裡，vLLM 偵測 ROCm 平台需要它
# 另外加上本目錄，讓 vllm 的 console script 找得到 middleware（console script 不會把 cwd 放進 sys.path）
AMD_SMI_PATH="$PWD/.venv/lib/python3.13/site-packages/_rocm_sdk_core/share/amd_smi"
export PYTHONPATH="$PWD:$AMD_SMI_PATH${PYTHONPATH:+:$PYTHONPATH}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# 只載入本專案的 force_rocm platform plugin（見 vllm_force_rocm.py）：
# - 這台機器同時有 NVIDIA GPU，vLLM 內建偵測會同時啟用 cuda 與 rocm 而報錯
# - 順便擋掉 amd-quark 的 general plugin，它和這版 vLLM 不相容（ImportError: RoutedExperts）
export VLLM_PLUGINS="${VLLM_PLUGINS:-force_rocm}"

# 外面 shell 若為了 NVIDIA 設了 CUDA_VISIBLE_DEVICES，HIP 也會吃這個變數而找不到 AMD GPU
unset CUDA_VISIBLE_DEVICES

# ps 顯示的 process 名稱前綴，例如 voyage-4-nano-rocm::EngineCore
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-voyage-4-nano-rocm}"

# --enforce-eager：vLLM 0.19.1（ROCm 版）的 VoyageQwen3BidirectionalEmbedModel 仍有
# torch.compile 的 IndexError（forward(*args, **kwargs) 讓動態 shape 標記失效），和 NVIDIA 版相同
#
# --gpu-memory-utilization：gfx1151 是統一記憶體（GPU 可用約 112 GiB），0.15 約 17 GiB，
# 這個模型只需約 0.7 GiB 權重，其餘給 KV cache；系統 RAM 吃緊時可再調低

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
  --host "${HOST:-0.0.0.0}" --port "${PORT:-12353}" \
  --disable-uvicorn-access-log \
  "$@"
