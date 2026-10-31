"""對執行中的 reranker server 做整合測試。

用法：
    uv run python test_client.py                          # 預設 http://localhost:12361
    uv run python test_client.py http://127.0.0.1:12361
"""

import sys
import time

import httpx

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:12361").rstrip("/")
MODEL = "nvidia/llama-nemotron-rerank-1b-v2"

QUERY = "台灣半導體產業的競爭優勢是什麼？"
DOCS = [
    "今天台北天氣晴朗，適合去陽明山爬山。",
    "台積電憑藉先進製程與晶圓代工模式，讓台灣在全球半導體供應鏈中占有關鍵地位。",
    "這家拉麵店的湯頭濃郁，叉燒軟嫩，假日常常大排長龍。",
]
# 中文約 1.5 字元/token，30000 字約 2 萬 tokens，確定超過 max_model_len（預設 4096，上限 8192）
LONG_DOC = "台灣的半導體產業在全球供應鏈中扮演關鍵角色，晶圓代工技術領先。" * 1000

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f"：{detail}" if detail else ""))


def rerank(client: httpx.Client, documents: list[str], **extra) -> httpx.Response:
    return client.post(
        f"{BASE_URL}/rerank",
        json={"model": MODEL, "query": QUERY, "documents": documents, **extra},
    )


def main() -> int:
    with httpx.Client(timeout=120) as client:
        try:
            r = client.get(f"{BASE_URL}/health")
        except httpx.ConnectError:
            print(f"無法連線到 {BASE_URL}，請先執行 ./start.sh")
            return 1
        check("health", r.status_code == 200, f"HTTP {r.status_code}")

        r = rerank(client, DOCS)
        ok = r.status_code == 200
        if ok:
            ranked = r.json()["results"]
            scores = ", ".join(f"#{x['index']}={x['relevance_score']:.4f}" for x in ranked)
            check("基本 rerank：相關文件排第一", ranked[0]["index"] == 1, scores)
        else:
            check("基本 rerank", False, f"HTTP {r.status_code} {r.text[:200]}")

        r = rerank(client, DOCS, top_n=2)
        check("top_n=2 只回傳 2 筆", r.status_code == 200 and len(r.json()["results"]) == 2)

        r = rerank(client, [LONG_DOC, DOCS[0]])
        check(
            "超長文件（未帶參數）由 middleware 預設截斷，不報錯",
            r.status_code == 200,
            f"HTTP {r.status_code}" + ("" if r.status_code == 200 else f" {r.text[:200]}"),
        )
        if r.status_code == 200:
            check(
                "截斷後的超長相關文件仍排第一",
                r.json()["results"][0]["index"] == 0,
                ", ".join(f"#{x['index']}={x['relevance_score']:.4f}" for x in r.json()["results"]),
            )

        # 截斷是對每一組 query+文件各自處理：超長文件不影響同一請求中的其他文件
        alone = rerank(client, [DOCS[0], DOCS[1]])
        mixed = rerank(client, [DOCS[0], LONG_DOC, DOCS[1]])
        if alone.status_code == 200 and mixed.status_code == 200:
            a = {x["index"]: x["relevance_score"] for x in alone.json()["results"]}
            m = {x["index"]: x["relevance_score"] for x in mixed.json()["results"]}
            # bf16 下 batch 組成不同時分數會有約 0.005 的雜訊，容忍度取 1e-2（約 0.1 logit）
            same = abs(a[0] - m[0]) < 1e-2 and abs(a[1] - m[2]) < 1e-2
            check(
                "截斷只影響超長那份：3 份都有分數，短文件分數不變",
                len(m) == 3 and same,
                f"單獨送 {a[0]:.4f}/{a[1]:.4f}，混超長文件 {m[0]:.4f}/{m[2]:.4f}",
            )
        else:
            check("截斷只影響超長那份", False, f"HTTP {alone.status_code}/{mixed.status_code}")

        r = rerank(client, [LONG_DOC], truncate_prompt_tokens=None)
        check(
            "明確帶 truncate_prompt_tokens=null 時不截斷（應回 400）",
            r.status_code == 400,
            f"HTTP {r.status_code}",
        )

        r = rerank(client, [LONG_DOC], truncate_prompt_tokens=512)
        check("明確帶 truncate_prompt_tokens=512", r.status_code == 200, f"HTTP {r.status_code}")

        r = client.post(
            f"{BASE_URL}/score",
            json={"model": MODEL, "queries": QUERY, "documents": [LONG_DOC]},
        )
        check("/score 超長文件也預設截斷", r.status_code == 200, f"HTTP {r.status_code}")

        docs_100 = [DOCS[i % 3] + f"（第 {i} 份）" * 50 for i in range(100)]
        t = time.perf_counter()
        r = rerank(client, docs_100)
        elapsed = (time.perf_counter() - t) * 1000
        check(
            "100 份文件",
            r.status_code == 200 and len(r.json()["results"]) == 100,
            f"{elapsed:.0f} ms",
        )

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通過")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
