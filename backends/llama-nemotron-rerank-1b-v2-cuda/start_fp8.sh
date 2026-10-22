#!/usr/bin/env bash
# 以 FP8 啟動：vLLM 載入時把 BF16 權重動態量化成 FP8（需 Ada/Hopper/Blackwell GPU）
# 權重記憶體 2.3GB → 1.4GB、長文件 rerank 約快 1.5 倍，分數與 BF16 差距約 0.04 以內
# 參數與環境變數同 start.sh，例如：
#   ./start_fp8.sh --gpu 1 --port 8001
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/start.sh" --quantization fp8_per_tensor "$@"
