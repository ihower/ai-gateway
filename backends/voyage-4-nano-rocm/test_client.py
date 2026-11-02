"""HTTP smoke test：只透過 HTTP API 驗證 server，不在本機載入模型。

需先啟動 serve_vllm.sh。
用法：uv run python test_client.py [--base-url http://localhost:12353]
"""

import argparse
import base64

import httpx
import numpy as np

MODEL = "voyage-4-nano"
QUERY = "Which planet is known as the Red Planet?"
DOCS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "火星因表面富含氧化鐵而呈現紅色，常被稱為紅色星球。",
]

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def embed(client, texts, input_type, **extra):
    r = client.post("/v1/embeddings", json={
        "model": MODEL, "input": texts, "input_type": input_type, **extra})
    r.raise_for_status()
    data = r.json()["data"]
    if extra.get("encoding_format") == "base64":
        return np.array([np.frombuffer(base64.b64decode(d["embedding"]), dtype="<f4")
                         for d in data])
    return np.array([d["embedding"] for d in data], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:12353")
    args = p.parse_args()
    client = httpx.Client(base_url=args.base_url, timeout=60)

    r = client.get("/health")
    check("GET /health", r.status_code == 200, f"status={r.status_code}")

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    check("GET /v1/models 含 voyage-4-nano", MODEL in ids, f"models={ids}")

    q = embed(client, [QUERY], "query")
    d = embed(client, DOCS, "document")
    check("預設維度 2048", q.shape == (1, 2048) and d.shape == (3, 2048), f"{q.shape}, {d.shape}")
    norms = np.linalg.norm(np.vstack([q, d]), axis=1)
    check("向量已 L2 normalize", np.allclose(norms, 1, atol=1e-3), f"norms={norms.round(4)}")

    scores = (q @ d.T)[0]
    check("檢索排序：火星文件分數高於金星", scores[1] > scores[0] and scores[2] > scores[0],
          f"scores={scores.round(4)}")

    q_as_doc = embed(client, [QUERY], "document")
    q_raw = embed(client, [QUERY], "none")
    cos_doc, cos_raw = float(q[0] @ q_as_doc[0]), float(q[0] @ q_raw[0])
    check("input_type 有作用（query/document/none 向量不同）",
          cos_doc < 0.9999 and cos_raw < 0.9999,
          f"cos(query,document)={cos_doc:.4f}, cos(query,none)={cos_raw:.4f}")

    for dim in (256, 512, 1024):
        sub = embed(client, DOCS, "document", dimensions=dim)
        check(f"dimensions={dim}", sub.shape == (3, dim), f"shape={sub.shape}")

    # bf16 推論每次請求間有約 1e-4 等級的數值抖動，用 cosine 比對而非逐值相等
    b64 = embed(client, DOCS, "document", encoding_format="base64")
    min_cos = float((b64 * d).sum(1).min()) if b64.shape == d.shape else 0.0
    check("base64 與 float 一致", min_cos > 0.9999, f"min cosine={min_cos:.6f}")

    r = client.post("/v1/embeddings", json={"model": MODEL, "input": "x", "input_type": "foo"})
    check("無效 input_type 回 400", r.status_code == 400, f"status={r.status_code}")

    r = client.post("/v1/embeddings", json={"model": MODEL, "input": "x"})
    usage = r.json().get("usage", {})
    check("回應含 usage token 數", usage.get("prompt_tokens", 0) > 0, f"usage={usage}")

    print("\nALL PASS" if not failures else f"\n{len(failures)} FAILED: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
