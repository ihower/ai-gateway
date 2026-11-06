#!/usr/bin/env bash
# 在 AMD ROCm GPU（Radeon 8060S / gfx1151、Radeon RX 9070 XT / gfx1201）上啟動 vLLM embedding server，
# 並掛上 prompt middleware。預設 0.0.0.0:12353（NVIDIA 版 voyage-4-nano-cuda 用 12351，兩者可同時跑）
# 額外參數會直接傳給 vllm serve。用 --gpu 指定 HIP 裝置編號（等同 HIP_VISIBLE_DEVICES），編號可用 --list-gpus 查詢：
#   ./serve_vllm.sh --list-gpus
#   ./serve_vllm.sh --gpu 1
set -euo pipefail
cd "$(dirname "$0")"

# 解析 --gpu（支援 `--gpu 值` 與 `--gpu=值`）與 --list-gpus，其餘參數原樣傳給 vllm serve
VLLM_ARGS=()
LIST_GPUS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || { echo "--gpu 需要指定 HIP 裝置編號" >&2; exit 1; }
      export HIP_VISIBLE_DEVICES="$2"; shift 2 ;;
    --gpu=*)
      export HIP_VISIBLE_DEVICES="${1#*=}"; shift ;;
    --list-gpus)
      LIST_GPUS=1; shift ;;
    *)
      VLLM_ARGS+=("$1"); shift ;;
  esac
done

# AMD 文件要求：amd_smi 的 Python binding 在 rocm-sdk-core 裡，vLLM 偵測 ROCm 平台需要它
# 另外加上本目錄，讓 vllm 的 console script 找得到 middleware（console script 不會把 cwd 放進 sys.path）
AMD_SMI_PATH="$PWD/.venv/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi"
export PYTHONPATH="$PWD:$AMD_SMI_PATH${PYTHONPATH:+:$PYTHONPATH}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# 只載入本專案的 force_rocm platform plugin（見 vllm_force_rocm.py）：
# - 這台機器同時有 NVIDIA GPU，vLLM 內建偵測會同時啟用 cuda 與 rocm 而報錯
# - 順便擋掉 amd-quark 的 general plugin，避免它和 vLLM 版本不相容
export VLLM_PLUGINS="${VLLM_PLUGINS:-force_rocm}"

# 外面 shell 若為了 NVIDIA 設了 CUDA_VISIBLE_DEVICES，HIP 也會吃這個變數而找不到 AMD GPU
unset CUDA_VISIBLE_DEVICES

# 列出 HIP 看得到的 AMD GPU（不受 HIP_VISIBLE_DEVICES / --gpu 限制）
if [[ "$LIST_GPUS" == 1 ]]; then
  exec env -u HIP_VISIBLE_DEVICES uv run python -c '
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    arch = p.gcnArchName.split(":")[0]
    print(f"{i}: {p.name} ({arch}, {p.total_memory / 2**30:.1f} GiB)")
'
fi

# --gpu-memory-utilization：未指定時依所選 GPU 的總記憶體自動計算
# GPU_MEMORY_UTILIZATION = GPU_MEMORY_TARGET_GIB / 總記憶體（上限 0.9）。
# 模型權重只有約 0.7 GiB，其餘給 KV cache。預設 8 GiB：
# 8060S（統一記憶體約 112 GiB，會實際佔用系統 RAM）約 0.071，9070 XT（16 GiB）約 0.5
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
GPU_MEMORY_TARGET_GIB="${GPU_MEMORY_TARGET_GIB:-8}"

# 偵測要用的 GPU（可見裝置中的第 0 張，vLLM 也會用這張），顯示型號並決定 GPU_MEMORY_UTILIZATION
GPU_INFO="$(uv run python -c '
import torch
if torch.cuda.device_count() == 0:
    raise SystemExit("找不到 AMD GPU")
p = torch.cuda.get_device_properties(0)
print(p.name, p.gcnArchName.split(":")[0], p.total_memory, sep="\t")
')" || { echo "偵測 GPU 失敗（HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-未設定}）" >&2; exit 1; }
IFS=$'\t' read -r GPU_NAME GPU_ARCH GPU_TOTAL_BYTES <<<"$GPU_INFO"
if [[ -z "$GPU_MEMORY_UTILIZATION" ]]; then
  GPU_MEMORY_UTILIZATION="$(awk -v t="$GPU_MEMORY_TARGET_GIB" -v m="$GPU_TOTAL_BYTES" \
    'BEGIN { u = t * 1073741824 / m; if (u > 0.9) u = 0.9; printf "%.3f", u }')"
fi
echo "使用 GPU：$GPU_NAME（$GPU_ARCH，$(awk -v m="$GPU_TOTAL_BYTES" 'BEGIN { printf "%.1f", m / 1073741824 }') GiB，HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-未設定}），GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2

# ps 顯示的 process 名稱前綴，例如 voyage-4-nano-rocm::EngineCore
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-voyage-4-nano-rocm}"

# --enforce-eager：vLLM 0.19.1 時 torch.compile 會 IndexError。0.29 offline 測試已可 compile，
# 但 HTTP server 尚未在 compile 模式驗證，先保留；要試可傳 -cc 等參數或自行移除

exec uv run vllm serve voyageai/voyage-4-nano \
  --served-model-name voyage-4-nano \
  --runner pooling --convert embed \
  --hf-overrides '{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}' \
  --pooler-config '{"pooling_type":"MEAN"}' \
  --trust-remote-code --dtype bfloat16 --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --api-server-count "${API_SERVER_COUNT:-1}" \
  --middleware prompt_middleware.VoyagePromptMiddleware \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-12353}" \
  --disable-uvicorn-access-log \
  "${VLLM_ARGS[@]}"
