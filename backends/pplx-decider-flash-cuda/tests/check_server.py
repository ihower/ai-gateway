"""對任一個 SystemOne 端點（本專案的 vLLM server、llama.cpp 等）送 fixtures + 合成 eval，比對準確度與延遲。

用法：
  uv run python tests/check_server.py http://127.0.0.1:12372 --baseline bf16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import synthetic  # noqa: E402
from check_accuracy import compare  # noqa: E402


#: 官方 protocol 對缺少 instructions 的題目用這句（pplx_decider/protocol.py DEFAULT_INSTRUCTIONS）。
#: llama.cpp 要求每題都有 instructions，所以送出前先補上，prompt 與官方相同（fxoz 的 client 也這樣做）。
DEFAULT_INSTRUCTIONS = "Choose the best matching option."


def normalize(body: dict) -> dict:
    questions = {
        key: {**q, "instructions": q.get("instructions") or DEFAULT_INSTRUCTIONS}
        for key, q in body["questions"].items()
    }
    return {**body, "questions": questions}


def post(url: str, body: dict) -> dict:
    body = normalize(body)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--fixtures", default=str(ROOT / "tests" / "fixtures" / "reference.json"))
    parser.add_argument("--baseline", help="check_accuracy.py --dump 的結果（例如 BF16）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    fixtures = json.loads(Path(args.fixtures).read_text())
    cases = [(f"ref{i}", item["request"]) for i, item in enumerate(fixtures)]
    cases += [(f"syn{i}", request) for i, request in enumerate(synthetic.requests("eval"))]

    post(args.url, cases[0][1])  # warmup
    answers: dict[str, dict] = {}
    official: dict[str, dict] = {}
    latencies: list[float] = []
    usage_mismatch = 0
    for index, (name, request) in enumerate(cases):
        started = time.perf_counter()
        response = post(args.url, request)
        latencies.append(time.perf_counter() - started)
        for key, value in response["answers"].items():
            answers[f"{name}.{key}"] = value
        if index < len(fixtures):
            want = fixtures[index]["response"]
            usage_mismatch += response["usage"]["input_tokens"] != want["usage"]["input_tokens"]
            for key, value in want["answers"].items():
                official[f"{name}.{key}"] = value

    compare("vs official", {k: answers[k] for k in official}, official, args.verbose)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        compare("vs baseline, all", answers, baseline, args.verbose)
        compare("vs baseline, synthetic", answers, {k: v for k, v in baseline.items() if k.startswith("syn")}, False)
    print(f"usage.input_tokens 不一致: {usage_mismatch}")
    print(f"{len(cases)} requests 逐一送出：total {sum(latencies):.2f}s, "
          f"median {statistics.median(latencies) * 1000:.0f} ms, max {max(latencies) * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
