"""SystemOneMiddleware 的單元測試：用假的 vLLM app 取代 /classify，不需要 GPU。

需要模型目錄（只讀 tokenizer 與設定）：PPLX_MODEL_DIR，預設 /mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4
  uv run --group dev pytest tests/test_middleware.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import httpx
import orjson
import pytest

os.environ.setdefault("PPLX_MODEL_DIR", "/mnt/data/models/pplx/pplx-decider-flash-v1.3-vllm-nvfp4")
if not Path(os.environ["PPLX_MODEL_DIR"], "decision_config.json").exists():
    pytest.skip("找不到模型目錄", allow_module_level=True)

from systemone_middleware import SystemOneMiddleware  # noqa: E402

TEMPERATURE = json.loads(Path(os.environ["PPLX_MODEL_DIR"], "decision_config.json").read_text())["temperature"]


class FakeVLLM:
    """記錄收到的 /classify request，回傳固定的 logits（第 i 個選項 logit = i）。"""

    def __init__(self, status: int = 200):
        self.calls: list[tuple[str, dict]] = []
        self.status = status

    async def __call__(self, scope, receive, send):
        body = (await receive())["body"]
        payload = orjson.loads(body) if body else None
        self.calls.append((scope["path"], payload))
        if self.status != 200:
            out = {"error": {"code": self.status, "message": "boom", "type": "BadRequestError"}}
        elif scope["path"] == "/classify":
            out = {"model": "pplx-decider-flash-v1.3",
                   "data": [{"index": i, "label": None, "probs": [float(j) for j in range(255)], "num_classes": 255}
                            for i in reversed(range(len(payload["input"])))]}
        else:
            out = {"other": True}
        data = orjson.dumps(out)
        await send({"type": "http.response.start", "status": self.status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": data})


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


REQUEST = {
    "model": "anything",
    "state": "My integration fails with a 401.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Money", "support": "Bugs"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "level": {"type": "score", "criteria": ["low", "mid", "high"]},
    },
}


def softmax(values):
    exps = [math.exp(v / TEMPERATURE) for v in values]
    return [e / sum(exps) for e in exps]


@pytest.mark.anyio
async def test_systemone_round_trip():
    fake = FakeVLLM()
    async with client(SystemOneMiddleware(fake)) as c:
        response = await c.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    # 每題一個 sequence，順序同 questions；只送 token ids 與 use_activation=false
    (path, payload), = fake.calls
    assert path == "/classify"
    assert payload["use_activation"] is False and len(payload["input"]) == 3
    assert body["model"] == "pplx-decider-flash-v1.3"
    assert list(body["answers"]) == ["team", "urgent", "level"]
    # 只取有效選項的 logits，除以溫度後 softmax（假 logits：選項 i = i）
    assert body["answers"]["team"]["probabilities"] == pytest.approx(dict(zip(["billing", "support"], softmax([0, 1]))))
    assert body["answers"]["team"]["choice"] == "support"
    assert body["answers"]["urgent"] == {"type": "noul", "noul": pytest.approx(softmax([0, 1])[1])}
    assert body["answers"]["level"]["score"] == pytest.approx(sum(i * p for i, p in enumerate(softmax([0, 1, 2]))))
    assert body["usage"]["sequences"] == 3
    assert body["usage"]["input_tokens"] == sum(len(ids) for ids in payload["input"])
    assert body["usage"]["output_tokens"] == 0


@pytest.mark.anyio
async def test_temperature_override():
    async with client(SystemOneMiddleware(FakeVLLM())) as c:
        response = await c.post("/v1/systemone", json={**REQUEST, "temperature": 1.0})
    assert response.json()["answers"]["urgent"]["noul"] == pytest.approx(math.exp(1) / (1 + math.exp(1)))


@pytest.mark.anyio
@pytest.mark.parametrize("body, message", [
    (b"{not json", "not valid JSON"),
    (orjson.dumps({"state": "x"}), "question"),
    (orjson.dumps({"state": "x", "questions": {"q": {"type": "bogus"}}}), "type must be"),
    (orjson.dumps({"state": "x", "questions": {"q": {"type": "score", "criteria": ["only"]}}}), "at least two"),
    (orjson.dumps({"state": "x", "questions": {"q": {"type": "locate"}}}), "locate"),
    (orjson.dumps({**REQUEST, "images": ["a.png"]}), "images"),
    (orjson.dumps({**REQUEST, "temperature": 0}), "temperature"),
])
async def test_bad_requests_return_400_without_engine_call(body, message):
    fake = FakeVLLM()
    async with client(SystemOneMiddleware(fake)) as c:
        response = await c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert fake.calls == []


@pytest.mark.anyio
async def test_too_long_returns_400(monkeypatch):
    monkeypatch.setenv("MAX_MODEL_LEN", "64")
    async with client(SystemOneMiddleware(FakeVLLM())) as c:
        response = await c.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 400
    assert "token" in response.json()["error"]["message"]


@pytest.mark.anyio
async def test_engine_error_is_forwarded():
    async with client(SystemOneMiddleware(FakeVLLM(status=400))) as c:
        response = await c.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "boom"


@pytest.mark.anyio
async def test_other_paths_pass_through():
    fake = FakeVLLM()
    async with client(SystemOneMiddleware(fake)) as c:
        response = await c.post("/v1/models", json={})
    assert response.json() == {"other": True}
    assert fake.calls[0][0] == "/v1/models"


@pytest.fixture
def anyio_backend():
    return "asyncio"
