"""對照 vLLM 後端驗證 fallback server 的行為；後端連不上時跳過對照部分。

執行：uv run pytest -q
"""

import base64
import json
import urllib.error
import urllib.request

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app

VLLM = "http://127.0.0.1:12351/v1/embeddings"
TEXTS = ["The capital of France is Paris.", "Embedding models map text to vectors.", "台北是台灣的首都。"]
client = TestClient(app)


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
