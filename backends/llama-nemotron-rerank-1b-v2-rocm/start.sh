#!/usr/bin/env bash
# 在 AMD ROCm GPU（Radeon 8060S / gfx1151、Radeon RX 9070 XT / gfx1201）上啟動 nvidia/llama-nemotron-rerank-1b-v2 的 vLLM
# reranker HTTP API（/rerank、/score）。預設 0.0.0.0:12363，可和 NVIDIA 版（12361、12362）同時跑。
# 可用環境變數覆寫設定，額外參數會直接傳給 vllm serve，例如：
#   PORT=8000 ./start.sh --max-num-batched-tokens 16384
#   ./start.sh --port 8001
# 用 --gpu 指定 HIP 裝置編號（等同 HIP_VISIBLE_DEVICES），編號可用 --list-gpus 查詢：
#   ./start.sh --list-gpus
#   ./start.sh --gpu 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-nvidia/llama-nemotron-rerank-1b-v2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-12363}"
DTYPE="${DTYPE:-auto}"
# 直接指定 KV cache 大小（每 token 32KiB）。128M（大寫 M = MiB）剛好容納 4096 tokens，
# 調高 MAX_MODEL_LEN 時要一併調高（每 4096 tokens 128M）
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-128M}"
# 設定 KV_CACHE_MEMORY_BYTES 後此比例不影響分配，只用於 vLLM 啟動時檢查剩餘記憶體。
# 未指定時依所選 GPU 的總記憶體自動計算：GPU_MEMORY_UTILIZATION = GPU_MEMORY_TARGET_GIB / 總記憶體，
# 6 GiB 足夠權重 2.3GB + KV + 運算空間。8060S（統一記憶體約 112 GiB）約 0.054，9070 XT（16 GiB）約 0.377
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
GPU_MEMORY_TARGET_GIB="${GPU_MEMORY_TARGET_GIB:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/nemotron-rerank.jinja}"
# 分數校準 sigmoid(logit / LOGIT_SIGMA)，由 rerank_middleware.ScoreCalibrationMiddleware 處理
export LOGIT_SIGMA="${LOGIT_SIGMA:-4.0}"

# 解析 --gpu / --port（支援 `--opt 值` 與 `--opt=值`）與 --list-gpus，其餘參數原樣傳給 vllm serve
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
    --port)
      [[ $# -ge 2 ]] || { echo "--port 需要指定 port" >&2; exit 1; }
      PORT="$2"; shift 2 ;;
    --port=*)
      PORT="${1#*=}"; shift ;;
    *)
      VLLM_ARGS+=("$1"); shift ;;
  esac
done

# score template 是必要的，缺少時 query/passage 格式不會套用，分數會錯
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "找不到 score template：$CHAT_TEMPLATE" >&2
  exit 1
fi

# AMD 文件要求：amd_smi 的 Python binding 在 rocm-sdk-core 裡，vLLM 偵測 ROCm 平台需要它
# 另外加上本目錄，讓 vLLM 能 import rerank_middleware
AMD_SMI_PATH="$SCRIPT_DIR/.venv/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi"
export PYTHONPATH="$SCRIPT_DIR:$AMD_SMI_PATH${PYTHONPATH:+:$PYTHONPATH}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# 只載入 force_rocm platform plugin（見 vllm_force_rocm.py）：
# - 這台機器同時有 NVIDIA GPU，vLLM 內建偵測會同時啟用 cuda 與 rocm 而報錯
# - 順便擋掉 amd-quark 的 general plugin，它和這版 vLLM 不相容
export VLLM_PLUGINS="${VLLM_PLUGINS:-force_rocm}"

# 外面 shell 若為了 NVIDIA 設了 CUDA_VISIBLE_DEVICES，HIP 也會吃這個變數而找不到 AMD GPU
unset CUDA_VISIBLE_DEVICES

# 列出 HIP 看得到的 AMD GPU（不受 HIP_VISIBLE_DEVICES / --gpu 限制）
if [[ "$LIST_GPUS" == 1 ]]; then
  exec env -u HIP_VISIBLE_DEVICES uv run --project "$SCRIPT_DIR" python -c '
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    arch = p.gcnArchName.split(":")[0]
    print(f"{i}: {p.name} ({arch}, {p.total_memory / 2**30:.1f} GiB)")
'
fi

# 偵測要用的 GPU（可見裝置中的第 0 張，vLLM 也會用這張），顯示型號並決定 GPU_MEMORY_UTILIZATION
GPU_INFO="$(uv run --project "$SCRIPT_DIR" python -c '
import torch
if torch.cuda.device_count() == 0:
    raise SystemExit("找不到 AMD GPU")
p = torch.cuda.get_device_properties(0)
print(p.name, p.gcnArchName.split(":")[0], p.total_memory, sep="\t")
')" || { echo "偵測 GPU 失敗（HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-未設定}）" >&2; exit 1; }
IFS=$'\t' read -r GPU_NAME GPU_ARCH GPU_TOTAL_BYTES <<<"$GPU_INFO"
if [[ -z "$GPU_MEMORY_UTILIZATION" ]]; then
  # 上限 0.9，避免小顯存 GPU 算出超過 1 的比例
  GPU_MEMORY_UTILIZATION="$(awk -v t="$GPU_MEMORY_TARGET_GIB" -v m="$GPU_TOTAL_BYTES" \
    'BEGIN { u = t * 1073741824 / m; if (u > 0.9) u = 0.9; printf "%.3f", u }')"
fi
echo "使用 GPU：$GPU_NAME（$GPU_ARCH，$(awk -v m="$GPU_TOTAL_BYTES" 'BEGIN { printf "%.1f", m / 1073741824 }') GiB，HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-未設定}），GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" >&2

# 程序名稱前綴，ps 會顯示為 nemotron-rerank-rocm::EngineCore
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-nemotron-rerank-rocm}"

exec uv run --project "$SCRIPT_DIR" vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --middleware rerank_middleware.DefaultTruncationMiddleware \
  --middleware rerank_middleware.ScoreCalibrationMiddleware \
  ${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"}
