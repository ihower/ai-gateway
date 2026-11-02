import socket
import threading
import time

import orjson
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import prompt_middleware
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


class ClientError(ValueError):
    """替身：代表 vLLM 判定為 4xx 的例外（例如超過 max-model-len）。"""


@pytest.fixture
def raising_app(monkeypatch):
    """模擬 vLLM：Exception handler 交給 ServerErrorMiddleware，路由直接拋例外。"""
    def fake_payload(exc):
        if isinstance(exc, ClientError):
            return 400, {"error": {"message": str(exc), "type": "BadRequestError",
                                   "param": "input_tokens", "code": 400}}
        return None

    monkeypatch.setattr(prompt_middleware, "client_error_payload", fake_payload)

    app = FastAPI()
    app.exception_handler(Exception)(
        lambda request, exc: JSONResponse({"from": "server-error-handler"}, status_code=500))

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body = orjson.loads(await request.body())
        if body["input"].endswith("too-long"):
            raise ClientError("too long")
        if body["input"].endswith("boom"):
            raise RuntimeError("boom")
        return {"ok": True}

    @app.post("/v2/embed")
    async def other():
        raise ClientError("too long")

    app.add_middleware(VoyagePromptMiddleware)
    return app


@pytest.mark.parametrize("path, body", [
    ("/v1/embeddings", {"input": "too-long"}),
    ("/v2/embed", {}),
])
def test_client_error_becomes_response(raising_app, path, body):
    with TestClient(raising_app) as c:  # 例外若沒被接住，TestClient 會直接拋出
        r = c.post(path, json=body)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "input_tokens"


def test_server_error_still_propagates(raising_app):
    with TestClient(raising_app) as c, pytest.raises(RuntimeError):
        c.post("/v1/embeddings", json={"input": "boom"})


def test_exception_after_response_started_propagates(monkeypatch):
    monkeypatch.setattr(prompt_middleware, "client_error_payload",
                        lambda exc: pytest.fail("回應已開始，不應再送錯誤回應"))

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ClientError("too late")

    with pytest.raises(ClientError):  # 不用 with TestClient：這個假 app 不處理 lifespan
        TestClient(VoyagePromptMiddleware(app)).get("/anything")


def test_connection_stays_open_after_client_error(raising_app):
    """回歸測試：真的跑 uvicorn，超長 400 之後同一條 TCP 連線還能再用。"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(raising_app, log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)

    def request(text):
        body = orjson.dumps({"input": text})
        return (b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Length: "
                + str(len(body)).encode() + b"\r\n\r\n" + body)

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            conn.sendall(request("too-long"))
            assert conn.recv(65536).startswith(b"HTTP/1.1 400")
            time.sleep(0.1)  # 留時間讓 server 真的關閉（修正前這裡會收到 FIN）
            conn.sendall(request("ok"))
            assert conn.recv(65536).startswith(b"HTTP/1.1 200")
    finally:
        server.should_exit = True
        thread.join(5)
