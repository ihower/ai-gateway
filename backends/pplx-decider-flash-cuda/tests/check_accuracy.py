"""以 vLLM（離線，不開 server）跑 fixtures，與官方 reference runtime 的機率比對。

fixtures：tests/fixtures/reference.json，每筆含 request、官方 response 與每題的 token ids
（來自 fxoz/pplx-decider-flash-v1.3-GGUF 的 validation/reference.json，官方 runtime FP32 CPU 產生）。

用法：
  CUDA_VISIBLE_DEVICES=1 uv run python tests/check_accuracy.py /mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from decider_codec import DecisionCodec  # noqa: E402


def option_probs(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def top(answer: dict):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    return answer.get("choice", max(answer["probabilities"], key=answer["probabilities"].get))


def compare(label: str, got: dict[str, dict], want: dict[str, dict], verbose: bool) -> None:
    """依 key 比對兩組 answers：top answer 一致率與選項機率誤差。"""
    agree = 0
    errors: list[float] = []
    for key, ref in want.items():
        mine = got[key]
        agree += top(mine) == top(ref)
        diffs = [abs(a - b) for a, b in zip(option_probs(mine), option_probs(ref))]
        errors.extend(diffs)
        if verbose and (top(mine) != top(ref) or max(diffs) > 0.05):
            print(f"  {key}: mine={[round(v, 4) for v in option_probs(mine)]} ref={[round(v, 4) for v in option_probs(ref)]}")
    print(f"[{label}] top-answer agreement: {agree}/{len(want)}, "
          f"option prob |Δ| mean={sum(errors) / len(errors):.6f} max={max(errors):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures" / "reference.json"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--quantization", help="vLLM 載入時量化，例如 fp8（BF16 checkpoint 即時轉 FP8）")
    parser.add_argument("--dump", help="把所有題目的 answers 寫成 JSON（例如 BF16 的結果，給 --baseline 用）")
    parser.add_argument("--baseline", help="另一個模型 --dump 的結果；fixtures 與合成 eval 都和它比對")
    parser.add_argument("-v", "--verbose", action="store_true", help="列出不一致或誤差 > 0.05 的題目")
    args = parser.parse_args()

    from vllm import LLM, PoolingParams
    from vllm.inputs import TokensPrompt

    import synthetic

    fixtures = json.loads(Path(args.fixtures).read_text())
    codec = DecisionCodec(args.model_dir, max_length=args.max_model_len)
    # (名稱前綴, request)：官方 fixtures + 合成 eval split（不參與量化校準）
    cases = [(f"ref{i}", item["request"]) for i, item in enumerate(fixtures)]
    cases += [(f"syn{i}", request) for i, request in enumerate(synthetic.requests("eval"))]

    # tokenization 必須與官方逐 token 一致
    token_mismatch = 0
    plans = [codec.plan(request) for _, request in cases]
    for item, chunks in zip(fixtures, plans):
        for chunk in chunks:
            (key,) = chunk.keys
            if chunk.input_ids.tolist() != item["tokens"][key]:
                token_mismatch += 1
                print(f"TOKEN MISMATCH {key}")
    print(f"tokenization: {sum(len(p) for p in plans[:len(fixtures)])} 個 fixture sequence，不一致 {token_mismatch}")

    llm = LLM(
        model=args.model_dir,
        runner="pooling",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        quantization=args.quantization,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    prompts = [TokensPrompt(prompt_token_ids=c.input_ids.tolist()) for chunks in plans for c in chunks]
    params = PoolingParams(task="classify", use_activation=False)
    llm.encode(prompts[:2], pooling_params=params, pooling_task="classify", use_tqdm=False)  # warmup
    started = time.perf_counter()
    outputs = llm.encode(prompts, pooling_params=params, pooling_task="classify", use_tqdm=False)
    elapsed = time.perf_counter() - started
    logits = [o.outputs.data.tolist() for o in outputs]

    answers: dict[str, dict] = {}
    official: dict[str, dict] = {}
    usage_mismatch = 0
    cursor = 0
    for index, ((name, request), chunks) in enumerate(zip(cases, plans)):
        rows = logits[cursor : cursor + len(chunks)]
        cursor += len(chunks)
        got = codec.respond(request, chunks, rows, model=None)
        for key, value in got["answers"].items():
            answers[f"{name}.{key}"] = value
        if index < len(fixtures):
            want = fixtures[index]["response"]
            usage_mismatch += got["usage"]["input_tokens"] != want["usage"]["input_tokens"]
            for key, value in want["answers"].items():
                official[f"{name}.{key}"] = value

    compare("vs official", {k: answers[k] for k in official}, official, args.verbose)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        compare("vs baseline, all", answers, baseline, args.verbose)
        synthetic_keys = [k for k in baseline if k.startswith("syn")]
        compare("vs baseline, synthetic", answers, {k: baseline[k] for k in synthetic_keys}, False)
    if args.dump:
        Path(args.dump).write_text(json.dumps(answers, ensure_ascii=False, indent=1))
    print(f"usage.input_tokens 不一致: {usage_mismatch}")
    print(f"{len(prompts)} sequences in {elapsed * 1000:.0f} ms")
    return 0 if token_mismatch == 0 and usage_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
