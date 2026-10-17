"""vLLM ASGI middleware：rerank/score 請求未指定 truncate_prompt_tokens 時，預設截斷到 max_model_len。

vLLM 預設遇到超過 max_model_len 的 query+document 會回 400；這裡改成預設截斷（從右側，
保留 query 與文件開頭）。截斷發生在 vLLM 原本就要做的 tokenize 之後，不會多一次 tokenize。

client 若明確帶入 truncate_prompt_tokens（包含 null）則尊重其設定，不覆寫。

用法：vllm serve ... --middleware rerank_middleware.DefaultTruncationMiddleware
"""

import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TRUNCATE_PATHS = frozenset({"/rerank", "/v1/rerank", "/v2/rerank", "/score", "/v1/score"})
DEFAULT_TRUNCATE_PROMPT_TOKENS = -1  # -1 代表截斷到 max_model_len


class DefaultTruncationMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in TRUNCATE_PATHS
        ):
            await self.app(scope, receive, send)
            return

        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = _inject_default(b"".join(chunks))

        headers = [(k, v) for k, v in scope["headers"] if k != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        scope = {**scope, "headers": headers}

        sent = False

        async def replay() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def _inject_default(body: bytes) -> bytes:
    """body 是 JSON object 且沒有 truncate_prompt_tokens 時補上預設值；其他情況原樣回傳，交給 vLLM 處理錯誤。"""
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError:
        return body
    if not isinstance(payload, dict) or "truncate_prompt_tokens" in payload:
        return body
    payload["truncate_prompt_tokens"] = DEFAULT_TRUNCATE_PROMPT_TOKENS
    return orjson.dumps(payload)
