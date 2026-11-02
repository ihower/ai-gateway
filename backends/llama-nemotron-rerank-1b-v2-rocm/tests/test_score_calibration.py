"""ScoreCalibrationMiddleware 測試。

假的下游模擬 vLLM 0.19.1：use_activation=false 回傳原始 logit，否則回傳 sigmoid(logit)
（float32 下高 logit 會飽和成 1.0）。若 middleware 沒有改寫請求，分數就算不對。
"""

import json
import math

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from rerank_middleware import ScoreCalibrationMiddleware

LOGITS = [20.0, -1.0, -23.0]


def raw_or_sigmoid(payload: dict) -> list[float]:
    if payload.get("use_activation") is False:
        return LOGITS
    return [1.0 if x > 17 else 1 / (1 + math.exp(-x)) for x in LOGITS]


async def fake_rerank(request: Request) -> JSONResponse:
    payload = json.loads(await request.body())
    scores = raw_or_sigmoid(payload)
    results = [
        {"index": i, "document": {"text": f"文件{i}"}, "relevance_score": s}
        for i, s in enumerate(scores)
    ]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    if payload.get("top_n"):
        results = results[: payload["top_n"]]
    return JSONResponse(
        {"id": "rerank-1", "model": "m", "usage": {"prompt_tokens": 9, "total_tokens": 9},
         "results": results}
    )


async def fake_score(request: Request) -> JSONResponse:
    scores = raw_or_sigmoid(json.loads(await request.body()))
    return JSONResponse(
        {"id": "score-1", "object": "list", "model": "m", "usage": {"prompt_tokens": 9},
         "data": [{"index": i, "object": "score", "score": s} for i, s in enumerate(scores)]}
    )


async def fake_error(request: Request) -> JSONResponse:
    return JSONResponse({"error": {"message": "太長", "code": 400}, "score": 1.0}, status_code=400)


async def fake_not_json(request: Request) -> PlainTextResponse:
    return PlainTextResponse("not json")


async def echo_body(request: Request) -> JSONResponse:
    return JSONResponse({"raw": (await request.body()).decode(), "data": [{"score": 20.0}]})


RERANK_PATHS = ["/rerank", "/v1/rerank", "/v2/rerank"]
SCORE_PATHS = ["/score", "/v1/score"]


def make_client(**kwargs) -> TestClient:
    routes = [Route(p, fake_rerank, methods=["POST"]) for p in RERANK_PATHS]
    routes += [Route(p, fake_score, methods=["POST"]) for p in SCORE_PATHS]
    routes += [Route("/v1/embeddings", echo_body, methods=["POST"])]
    app = Starlette(routes=routes)
    app.add_middleware(ScoreCalibrationMiddleware, **kwargs)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return make_client(logit_sigma=4.0)


def calibrated(logit: float, sigma: float = 4.0) -> float:
    return 1 / (1 + math.exp(-logit / sigma))


@pytest.mark.parametrize("path", RERANK_PATHS)
def test_rerank_default_returns_sigmoid_of_scaled_logit(client: TestClient, path: str) -> None:
    results = client.post(path, json={"query": "q", "documents": ["a"]}).json()["results"]
    assert [r["index"] for r in results] == [0, 1, 2]
    assert [r["relevance_score"] for r in results] == pytest.approx(
        [calibrated(x) for x in LOGITS]
    )


@pytest.mark.parametrize("path", SCORE_PATHS)
def test_score_default_returns_sigmoid_of_scaled_logit(client: TestClient, path: str) -> None:
    data = client.post(path, json={"queries": "q", "documents": ["a"]}).json()["data"]
    assert [d["score"] for d in data] == pytest.approx([calibrated(x) for x in LOGITS])


def test_high_logit_does_not_saturate(client: TestClient) -> None:
    results = client.post("/rerank", json={"query": "q", "documents": ["a"]}).json()["results"]
    assert results[0]["relevance_score"] < 1.0


def test_explicit_true_is_same_as_default(client: TestClient) -> None:
    results = client.post(
        "/rerank", json={"query": "q", "documents": ["a"], "use_activation": True}
    ).json()["results"]
    assert results[0]["relevance_score"] == pytest.approx(calibrated(20.0))


@pytest.mark.parametrize("field,path", [("results", "/rerank"), ("data", "/score")])
def test_explicit_false_returns_scaled_logit(client: TestClient, field: str, path: str) -> None:
    body = {"query": "q", "queries": "q", "documents": ["a"], "use_activation": False}
    items = client.post(path, json=body).json()[field]
    key = "relevance_score" if field == "results" else "score"
    assert [i[key] for i in items] == pytest.approx([x / 4 for x in LOGITS])


def test_top_n_keeps_highest_logits(client: TestClient) -> None:
    results = client.post(
        "/rerank", json={"query": "q", "documents": ["a"], "top_n": 2}
    ).json()["results"]
    assert [r["index"] for r in results] == [0, 1]


def test_other_response_fields_preserved(client: TestClient) -> None:
    body = client.post("/rerank", json={"query": "q", "documents": ["a"]}).json()
    assert body["id"] == "rerank-1"
    assert body["usage"] == {"prompt_tokens": 9, "total_tokens": 9}
    assert body["results"][0]["document"] == {"text": "文件0"}


def test_content_length_matches_rewritten_body(client: TestClient) -> None:
    r = client.post("/rerank", json={"query": "q", "documents": ["a"]})
    assert r.headers["content-length"] == str(len(r.content))


def test_custom_sigma() -> None:
    results = make_client(logit_sigma=2.0).post(
        "/rerank", json={"query": "q", "documents": ["a"]}
    ).json()["results"]
    assert results[1]["relevance_score"] == pytest.approx(calibrated(-1.0, 2.0))


def test_sigma_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGIT_SIGMA", "2")
    results = make_client().post(
        "/rerank", json={"query": "q", "documents": ["a"]}
    ).json()["results"]
    assert results[1]["relevance_score"] == pytest.approx(calibrated(-1.0, 2.0))


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_non_positive_sigma_rejected_at_startup(sigma: float) -> None:
    with pytest.raises(ValueError, match="LOGIT_SIGMA"):
        ScoreCalibrationMiddleware(lambda *_: None, logit_sigma=sigma)


def test_extreme_logit_does_not_overflow() -> None:
    from rerank_middleware import sigmoid

    assert sigmoid(-5000.0) == 0.0
    assert sigmoid(5000.0) == 1.0


def test_error_response_untouched() -> None:
    app = Starlette(routes=[Route("/rerank", fake_error, methods=["POST"])])
    app.add_middleware(ScoreCalibrationMiddleware, logit_sigma=4.0)
    r = TestClient(app).post("/rerank", json={"query": "q", "documents": ["a"]})
    assert r.status_code == 400
    assert r.json() == {"error": {"message": "太長", "code": 400}, "score": 1.0}


def test_non_json_response_untouched() -> None:
    app = Starlette(routes=[Route("/rerank", fake_not_json, methods=["POST"])])
    app.add_middleware(ScoreCalibrationMiddleware, logit_sigma=4.0)
    r = TestClient(app).post("/rerank", json={"query": "q", "documents": ["a"]})
    assert r.text == "not json"


def test_invalid_request_json_passed_through() -> None:
    async def echo(request: Request) -> PlainTextResponse:
        return PlainTextResponse((await request.body()).decode())

    app = Starlette(routes=[Route("/rerank", echo, methods=["POST"])])
    app.add_middleware(ScoreCalibrationMiddleware, logit_sigma=4.0)
    r = TestClient(app).post("/rerank", content=b"{not json")
    assert r.text == "{not json"


def test_other_paths_untouched(client: TestClient) -> None:
    body = client.post("/v1/embeddings", json={"input": "x"}).json()
    assert json.loads(body["raw"]) == {"input": "x"}
    assert body["data"] == [{"score": 20.0}]
