"""掛進 vLLM API server 的 ASGI middleware（vllm serve --middleware），提供 SystemOne 格式的 POST /v1/systemone。

流程（都在同一個 process 內，不另開 proxy）：
  request → 官方 protocol v4 編碼（每題一個 sequence，見 decider_codec.py）
          → 在 process 內轉成 POST /classify（token ids、use_activation=false）交給 vLLM
          → 每個 sequence 的 255 維 readout logits → 除以校準溫度、softmax → SystemOne response

回應格式與 llama.cpp 的 /v1/systemone 相同：{"model", "answers", "usage"}；錯誤為
{"error": {"code", "message", "type"}}。其他路徑直接放行。

設定（環境變數，由 start_nvfp4.sh 設定）：
  PPLX_MODEL_DIR  模型目錄（讀 tokenizer、decision_config.json、joint_config.json）
  MAX_MODEL_LEN   單一 sequence（state + 一題）的 token 上限，超過回 400，不截斷
"""

from __future__ import annotations

import asyncio
import os

import orjson

from decider_codec import DecisionCodec, LengthError, RequestError

SYSTEMONE_PATH = "/v1/systemone"
CLASSIFY_PATH = "/classify"


async def read_body(receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ConnectionError("client disconnected")
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            return b"".join(chunks)


async def send_json(send, status: int, payload) -> None:
    body = orjson.dumps(payload)
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


def error(status: int, message: str, kind: str = "invalid_request_error") -> dict:
    return {"error": {"code": status, "message": message, "type": kind}}


class SystemOneMiddleware:
    def __init__(self, app):
        self.app = app
        self.codec = DecisionCodec(
            os.environ["PPLX_MODEL_DIR"], max_length=int(os.environ.get("MAX_MODEL_LEN", "32768"))
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != SYSTEMONE_PATH:
            return await self.app(scope, receive, send)
        if scope["method"] != "POST":
            return await send_json(send, 405, error(405, "use POST"))
        try:
            raw = await read_body(receive)
        except ConnectionError:
            return
        try:
            request = orjson.loads(raw)
            # 官方 Encoder 的 tokenization 是 CPU 工作，放到 thread 以免卡住 event loop
            chunks = await asyncio.to_thread(self.codec.plan, request)
            self.codec.temperature_of(request)
        except orjson.JSONDecodeError:
            return await send_json(send, 400, error(400, "request body is not valid JSON"))
        except (RequestError, LengthError) as exc:
            return await send_json(send, 400, error(400, str(exc)))

        status, body = await self._classify(scope, [chunk.input_ids.tolist() for chunk in chunks])
        if status != 200:
            # vLLM 的錯誤（例如超過 max_model_len、引擎錯誤）原樣轉回
            return await send_json(send, status, orjson.loads(body) if body else error(status, "engine error"))
        result = orjson.loads(body)
        rows = sorted(result["data"], key=lambda item: item["index"])
        try:
            response = self.codec.respond(request, chunks, [row["probs"] for row in rows], model=result.get("model"))
        except ValueError as exc:
            return await send_json(send, 500, error(500, str(exc), "server_error"))
        await send_json(send, 200, response)

    async def _classify(self, scope, sequences: list[list[int]]) -> tuple[int, bytes]:
        """在 process 內呼叫 vLLM 的 POST /classify，收集 status 與 body。"""
        payload = orjson.dumps({"input": sequences, "use_activation": False})
        headers = [(k, v) for k, v in scope.get("headers", []) if k not in (b"content-length", b"content-type")]
        headers += [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
        inner = {**scope, "path": CLASSIFY_PATH, "raw_path": CLASSIFY_PATH.encode(), "headers": headers,
                 "query_string": b""}
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                # 回應期間 Starlette 會監聽斷線；不讓它提早結束
                await asyncio.Event().wait()
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}

        status = 500
        parts: list[bytes] = []

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                parts.append(message.get("body", b""))

        await self.app(inner, receive, capture)
        return status, b"".join(parts)
