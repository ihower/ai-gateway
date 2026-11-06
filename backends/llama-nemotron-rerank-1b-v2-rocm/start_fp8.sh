#!/usr/bin/env bash
# 以 FP8 啟動：vLLM 載入時把 BF16 權重動態量化成 FP8，只支援 RX 9070 XT（gfx1201），不支援 8060S（gfx1151）。
# 算法和 NVIDIA 版（12361，--quantization fp8_per_tensor）對齊：
# - vllm_rocm_fp8_pad.py：activation 逐 token 量化、score 層補齊後同樣走 FP8
# - SCORE_CALIBRATION=native：用 vLLM 原生 logit_sigma，分數為 BF16 精度
# 權重記憶體 2.32 GiB → 1.41 GiB。和 12361 的分數平均差約 0.003，在 NVIDIA 自己 compile 對 eager 的差距範圍內
# 參數與環境變數同 start.sh，例如：
#   ./start_fp8.sh --gpu 0 --port 8001
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SCORE_CALIBRATION="${SCORE_CALIBRATION:-native}"
exec "$SCRIPT_DIR/start.sh" --quantization fp8_per_tensor "$@"
