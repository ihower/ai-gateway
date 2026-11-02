import orjson
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from prompt_middleware import PROMPTS, VoyagePromptMiddleware

QUERY = PROMPTS["query"]
DOC = PROMPTS["document"]


@pytest.fixture
def client():
    """模擬 vLLM：把收到的 body 與 content-length 原樣回傳。"""
    app = FastAPI()

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        raw = await request.body()
        assert int(request.headers["content-length"]) == len(raw)
        return orjson.loads(raw)

    @app.post("/other")
    async def other(request: Request):
        return orjson.loads(await request.body())

    app.add_middleware(VoyagePromptMiddleware)
    with TestClient(app) as c:
        yield c


def test_query_prompt_single_string(client):
    r = client.post("/v1/embeddings", json={"input": "hi", "model": "m", "input_type": "query"})
    assert r.json() == {"input": QUERY + "hi", "model": "m"}


def test_document_prompt_list_keeps_other_fields(client):
    r = client.post("/v1/embeddings", json={
        "input": ["a", "b"], "input_type": "document",
        "dimensions": 512, "encoding_format": "base64"})
    assert r.json() == {"input": [DOC + "a", DOC + "b"],
                        "dimensions": 512, "encoding_format": "base64"}


def test_default_input_type_is_document(client):
    assert client.post("/v1/embeddings", json={"input": "x"}).json()["input"] == DOC + "x"


def test_multibyte_content_length(client):
    r = client.post("/v1/embeddings", json={"input": "火星", "input_type": "query"})
    assert r.json()["input"] == QUERY + "火星"


def test_none_passes_token_ids_untouched(client):
    r = client.post("/v1/embeddings", json={"input": [1, 2, 3], "input_type": "none"})
    assert r.json()["input"] == [1, 2, 3]


@pytest.mark.parametrize("body", [
    {"input": [1, 2, 3], "input_type": "query"},
    {"input": "x", "input_type": "foo"},
    {"model": "m"},
])
def test_bad_requests_rejected(client, body):
    r = client.post("/v1/embeddings", json=body)
    assert r.status_code == 400
    assert "message" in r.json()["error"]


def test_invalid_json(client):
    r = client.post("/v1/embeddings", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_other_paths_untouched(client):
    r = client.post("/other", json={"input": "x", "input_type": "query"})
    assert r.json() == {"input": "x", "input_type": "query"}
