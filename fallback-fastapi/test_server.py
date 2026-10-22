"""對照 vLLM 後端驗證 fallback server 的行為；後端連不上時跳過對照部分。
rerank / decision 會實際呼叫 Cohere 與 Jev（少量請求）；未設定 API key 時跳過。

執行：uv run pytest -q
"""

import base64
import json
import urllib.error
import urllib.request

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from server import app

VLLM = "http://127.0.0.1:12351/v1/embeddings"
TEXTS = ["The capital of France is Paris.", "Embedding models map text to vectors.", "台北是台灣的首都。"]
client = TestClient(app)
# 進入 lifespan，建立轉送雲端 API 用的 httpx client
client.__enter__()


def vllm(body):
    req = urllib.request.Request(VLLM, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)
    except OSError:
        pytest.skip("vLLM backend not reachable")


def vectors(resp):
    out = []
    for d in sorted(resp["data"], key=lambda d: d["index"]):
        e = d["embedding"]
        out.append(np.frombuffer(base64.b64decode(e), "<f4") if isinstance(e, str) else np.array(e))
    return out


def cos(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.parametrize(
    "body",
    [
        {"input": TEXTS[0]},
        {"input": TEXTS},
        {"input": TEXTS, "input_type": "query"},
        {"input": TEXTS, "input_type": "document"},
        {"input": TEXTS, "input_type": "none"},
        {"input": TEXTS, "dimensions": 256},
        {"input": TEXTS, "dimensions": 1024, "input_type": "query"},
        {"input": TEXTS, "encoding_format": "base64"},
        {"input": TEXTS, "model": "voyage-4-nano", "user": "u"},
        {"input": [[9707, 1879], [9707]], "input_type": "none"},
        {"input": [9707, 1879], "input_type": "none"},
    ],
)
def test_matches_vllm(body):
    status, direct = vllm(body)
    r = client.post("/v1/embeddings", json=body)
    assert r.status_code == status == 200
    mine = r.json()
    assert mine.keys() == direct.keys()
    assert [d.keys() for d in mine["data"]] == [d.keys() for d in direct["data"]]
    assert mine["model"] == direct["model"]
    assert mine["usage"] == direct["usage"]
    a, b = vectors(mine), vectors(direct)
    assert [len(v) for v in a] == [len(v) for v in b]
    for x, y in zip(a, b):
        assert abs(np.linalg.norm(x) - 1) < 1e-3
        assert cos(x, y) > 0.999


@pytest.mark.parametrize(
    "body",
    [
        {"input": "x", "input_type": "bogus"},
        {"input": "x", "model": "nope"},
        {"input": []},
        {"input": [[1, 2]]},
        {"input": "x", "dimensions": 100},
    ],
)
def test_errors_match_vllm(body):
    status, direct = vllm(body)
    r = client.post("/v1/embeddings", json=body)
    assert r.status_code == status
    assert r.json()["error"]["message"] == direct["error"]["message"]


DOCS = ["Paris is the capital of France.", "Bananas are yellow.", "France's capital city is Paris."]
needs_cohere = pytest.mark.skipif(not server.os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
needs_jev = pytest.mark.skipif(not server.os.environ.get("JEV_API_KEY"), reason="JEV_API_KEY not set")


@needs_cohere
@pytest.mark.parametrize("path", ["/v1/rerank", "/v2/rerank"])
def test_rerank_via_cohere(path):
    body = {"model": "nvidia/llama-nemotron-rerank-1b-v2", "query": "capital of France", "documents": DOCS, "top_n": 2}
    r = client.post(path, json=body, headers={"Authorization": "Bearer client-token"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"] == "rerank-v4.0-fast"
    assert out["meta"]["billed_units"]["search_units"] == 1
    assert sorted(x["index"] for x in out["results"]) == [0, 2]
    for x in out["results"]:
        assert x["document"]["text"] == DOCS[x["index"]]
        assert 0 <= x["relevance_score"] <= 1
    scores = [x["relevance_score"] for x in out["results"]]
    assert scores == sorted(scores, reverse=True)

    # top_n 0 與 vLLM 一致：回傳全部
    r = client.post(path, json={**body, "top_n": 0})
    assert len(r.json()["results"]) == 3


@needs_cohere
def test_rerank_client_error_passes_through():
    r = client.post("/v1/rerank", json={"query": "q", "documents": []})
    assert r.status_code == 400
    assert "message" in r.json()


@needs_jev
def test_decision_via_jev():
    body = {
        "model": "clef-flash",
        "state": "The user asks: is it raining?",
        "questions": {"q": {"type": "noul", "instructions": "Is this a question about weather?"}},
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"] == "jev-1.14.0"
    assert out["answers"]["q"]["type"] == "noul"
    assert out["answers"]["q"]["noul"] > 0.5
    assert out.keys() >= {"model", "answers", "usage"}


@needs_jev
def test_decision_client_error_passes_through():
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "bogus"}}})
    assert r.status_code == 400


def test_upstream_failures_become_gateway_errors(monkeypatch):
    def handler(request: httpx.Request):
        assert request.headers["authorization"] == "Bearer test-key"
        if "cohere" in str(request.url):
            return httpx.Response(503, json={"message": "overloaded"})
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(server, "http", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    r = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert r.status_code == 502 and "HTTP 503" in r.json()["error"]["message"]
    r = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 502

    monkeypatch.delenv("JEV_API_KEY")
    r = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 503 and "JEV_API_KEY" in r.json()["error"]["message"]
