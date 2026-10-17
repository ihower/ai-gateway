"""掛進 vLLM API server 的 ASGI middleware（vllm serve --middleware）。

攔截 POST /v1/embeddings，依擴充欄位 input_type 加上 voyage-4-nano 的
query/document prompt 後交給 vLLM；回應不經過這裡處理，向量只序列化一次。
其他路徑直接放行。
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
        if (scope["type"] != "http" or scope["method"] != "POST"
                or scope["path"] != EMBEDDINGS_PATH):
            return await self.app(scope, receive, send)

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
    payload = orjson.dumps(
        {"error": {"message": message, "type": "BadRequestError", "code": status}})
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})
