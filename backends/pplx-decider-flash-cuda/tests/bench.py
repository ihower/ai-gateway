"""離線量測吞吐量與單筆延遲（不開 server）。

- throughput：一次送出大量 sequences（合成 eval 的題目重複到 --sequences 筆），量 sequences/s 與 tokens/s
- latency：逐一送出單一 request（約 3 題 = 3 個 sequences），量中位數延遲

用法：
  CUDA_VISIBLE_DEVICES=1 uv run python tests/bench.py <model_dir> [--quantization fp8]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import synthetic  # noqa: E402
from decider_codec import DecisionCodec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    codec = DecisionCodec(args.model_dir, max_length=args.max_model_len)
    requests = synthetic.requests("eval") + synthetic.requests("calib")
    plans = [codec.plan(r) for r in requests]
    seqs = [c.input_ids.tolist() for p in plans for c in p]
    batch = [seqs[i % len(seqs)] for i in range(args.sequences)]

    llm = LLM(
        model=args.model_dir,
        runner="pooling",
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = PoolingParams(task="classify", use_activation=False)

    def run(items):
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in items]
        return llm.encode(prompts, pooling_params=params, pooling_task="classify", use_tqdm=False)

    run(batch[:64])  # warmup
    started = time.perf_counter()
    run(batch)
    elapsed = time.perf_counter() - started
    tokens = sum(map(len, batch))
    print(f"throughput: {len(batch)} seqs / {elapsed:.2f}s = {len(batch) / elapsed:.0f} seqs/s, "
          f"{tokens / elapsed:.0f} tokens/s (avg {tokens / len(batch):.0f} tokens/seq)")

    latencies = []
    for plan in plans:
        started = time.perf_counter()
        run([c.input_ids.tolist() for c in plan])
        latencies.append(time.perf_counter() - started)
    print(f"latency (1 request ≈ 3 題): median {statistics.median(latencies) * 1000:.1f} ms, "
          f"max {max(latencies) * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
