import json

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from rerank_middleware import DefaultTruncationMiddleware


async def echo(request: Request) -> JSONResponse:
    """假的下游 app：回傳收到的 body 與 content-length，供測試檢查改寫結果。"""
    body = await request.body()
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    return JSONResponse(
        {
            "body": parsed,
            "raw": body.decode(),
            "content_length": request.headers.get("content-length"),
        }
    )


PATHS = ["/rerank", "/v1/rerank", "/v2/rerank", "/score", "/v1/score"]


@pytest.fixture
def client() -> TestClient:
    routes = [Route(p, echo, methods=["GET", "POST"]) for p in PATHS + ["/v1/embeddings"]]
    app = Starlette(routes=routes)
    app.add_middleware(DefaultTruncationMiddleware)
    return TestClient(app)


@pytest.mark.parametrize("path", PATHS)
def test_injects_default_truncation_when_missing(client: TestClient, path: str) -> None:
    r = client.post(path, json={"query": "q", "documents": ["d"]})
    data = r.json()
    assert data["body"] == {"query": "q", "documents": ["d"], "truncate_prompt_tokens": -1}
    assert data["content_length"] == str(len(data["raw"].encode()))


def test_keeps_explicit_value(client: TestClient) -> None:
    r = client.post("/rerank", json={"query": "q", "documents": ["d"], "truncate_prompt_tokens": 512})
    assert r.json()["body"]["truncate_prompt_tokens"] == 512


def test_keeps_explicit_null_as_opt_out(client: TestClient) -> None:
    r = client.post("/rerank", json={"query": "q", "documents": ["d"], "truncate_prompt_tokens": None})
    assert r.json()["body"]["truncate_prompt_tokens"] is None


def test_preserves_unicode_text(client: TestClient) -> None:
    r = client.post("/rerank", json={"query": "台灣半導體", "documents": ["晶圓代工"]})
    assert r.json()["body"]["query"] == "台灣半導體"
    assert r.json()["body"]["documents"] == ["晶圓代工"]


def test_other_paths_untouched(client: TestClient) -> None:
    r = client.post("/v1/embeddings", json={"input": "x"})
    assert r.json()["body"] == {"input": "x"}


def test_get_request_untouched(client: TestClient) -> None:
    r = client.get("/rerank")
    assert r.json()["body"] is None


def test_invalid_json_passed_through(client: TestClient) -> None:
    r = client.post("/rerank", content=b"{not json", headers={"content-type": "application/json"})
    assert r.json()["raw"] == "{not json"


def test_non_object_json_passed_through(client: TestClient) -> None:
    r = client.post("/rerank", json=["a", "b"])
    assert r.json()["body"] == ["a", "b"]


def test_streamed_body_in_multiple_chunks(client: TestClient) -> None:
    payload = json.dumps({"query": "q", "documents": ["d" * 5000]}).encode()

    def gen():
        for i in range(0, len(payload), 1000):
            yield payload[i : i + 1000]

    r = client.post("/rerank", content=gen(), headers={"content-type": "application/json"})
    assert r.json()["body"]["truncate_prompt_tokens"] == -1
    assert r.json()["body"]["documents"] == ["d" * 5000]
