"""掛進 vLLM API server 的 ASGI middleware（vllm serve --middleware）。

攔截 POST /v1/embeddings，依擴充欄位 input_type 加上 voyage-4-nano 的
query/document prompt 後交給 vLLM；回應不經過這裡處理，向量只序列化一次。
其他路徑直接放行。

另外在 vLLM 0.19.1 上保住 keep-alive：vLLM 把驗證錯誤（例如超過 max-model-len）交給
app.exception_handler(Exception)，也就是 Starlette 的 ServerErrorMiddleware。它送出回應後
一定會重新拋出例外，uvicorn 收到後就關閉連線，client 沿用這條連線的下一個請求會失敗。
這個 middleware 位於 ServerErrorMiddleware 之內，所以在這裡把 4xx 錯誤轉成回應，例外就不會往外傳。
"""

import os

import orjson

DEFAULT_INPUT_TYPE = os.environ.get("DEFAULT_INPUT_TYPE", "document")

PROMPTS = {
    "query": "Represent the query for retrieving supporting documents: ",
    "document": "Represent the document for retrieval: ",
    "none": "",
}

EMBEDDINGS_PATH = "/v1/embeddings"


class PromptError(ValueError):
    pass


def add_prompt(inp, prefix: str):
    """字串或字串陣列加上 prefix；token id 陣列無法加文字 prompt。"""
    if isinstance(inp, str):
        return prefix + inp
    if isinstance(inp, list) and inp and all(isinstance(t, str) for t in inp):
        return [prefix + t for t in inp]
    raise PromptError("input must be a string or a list of strings when input_type "
                      "adds a prompt; use input_type='none' for token ids")


def client_error_payload(exc: Exception) -> tuple[int, dict] | None:
    """用 vLLM 自己的對應規則判斷例外；4xx 回傳 (status, payload)，其他回傳 None。

    延後 import：這支檔案要能在沒有載入 vLLM 的環境下 import（例如單元測試）。
    """
    from vllm.entrypoints.serve.exception_handling.error_response import create_error_response

    err = create_error_response(exc)
    if not 400 <= err.error.code < 500:
        return None
    return err.error.code, err.model_dump()


def rewrite_body(raw: bytes) -> bytes:
    try:
        body = orjson.loads(raw)
    except orjson.JSONDecodeError:
        raise PromptError("Request body is not valid JSON")
    if not isinstance(body, dict) or "input" not in body:
        raise PromptError("Missing required field: input")

    input_type = body.pop("input_type", None) or DEFAULT_INPUT_TYPE
    prefix = PROMPTS.get(input_type)
    if prefix is None:
        raise PromptError(f"input_type must be one of {list(PROMPTS)}, got {input_type!r}")
    if prefix:
        body["input"] = add_prompt(body["input"], prefix)
    return orjson.dumps(body)


class VoyagePromptMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        response_started = False

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            if scope["method"] == "POST" and scope["path"] == EMBEDDINGS_PATH:
                await self.handle_embeddings(scope, receive, tracked_send)
            else:
                await self.app(scope, receive, tracked_send)
        except Exception as exc:
            # 回應已開始送出就無法再改；5xx 維持原行為，讓 uvicorn 記錄 traceback
            error = None if response_started else client_error_payload(exc)
            if error is None:
                raise
            await send_json(send, *error)

    async def handle_embeddings(self, scope, receive, send):
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        try:
            body = rewrite_body(b"".join(chunks))
        except PromptError as e:
            return await send_error(send, 400, str(e))

        headers = [(k, v) for k, v in scope["headers"] if k != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app({**scope, "headers": headers}, replay, send)


async def send_error(send, status: int, message: str):
    await send_json(send, status,
                    {"error": {"message": message, "type": "BadRequestError", "code": status}})


async def send_json(send, status: int, payload: dict):
    body = orjson.dumps(payload)
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
