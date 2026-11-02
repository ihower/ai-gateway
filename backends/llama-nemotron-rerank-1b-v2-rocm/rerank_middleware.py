"""vLLM ASGI middleware（vllm serve --middleware）。

DefaultTruncationMiddleware：rerank/score 請求未指定 truncate_prompt_tokens 時，預設截斷到
max_model_len。vLLM 預設遇到超過 max_model_len 的 query+document 會回 400；這裡改成預設截斷
（從右側，保留 query 與文件開頭）。client 明確帶入 truncate_prompt_tokens（包含 null）則不覆寫。

ScoreCalibrationMiddleware：讓分數和 NVIDIA 版（vLLM 0.31 的 --pooler-config logit_sigma）一致。
這版 vLLM（0.19.1）沒有 logit_sigma，所以向 vLLM 要原始 logit，再改寫回應：
- 預設或 use_activation=true：sigmoid(logit / LOGIT_SIGMA)
- use_activation=false：logit / LOGIT_SIGMA
不能直接換算 vLLM 的 sigmoid 輸出，因為高 logit 在 float32 已經飽和成 1.0，資訊已經遺失。
"""

import math
import os

import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

RERANK_PATHS = frozenset({"/rerank", "/v1/rerank", "/v2/rerank"})
SCORE_PATHS = frozenset({"/score", "/v1/score"})
TARGET_PATHS = RERANK_PATHS | SCORE_PATHS
DEFAULT_TRUNCATE_PROMPT_TOKENS = -1  # -1 代表截斷到 max_model_len
DEFAULT_LOGIT_SIGMA = 4.0


def _is_target(scope: Scope) -> bool:
    return scope["type"] == "http" and scope["method"] == "POST" and scope["path"] in TARGET_PATHS


async def _read_body(receive: Receive) -> bytes | None:
    """讀完整個 request body；client 中途斷線時回傳 None。"""
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


def _replace_content_length(headers: list[tuple[bytes, bytes]], body: bytes) -> list:
    kept = [(k, v) for k, v in headers if k.lower() != b"content-length"]
    kept.append((b"content-length", str(len(body)).encode()))
    return kept


async def _forward(app: ASGIApp, scope: Scope, body: bytes, receive: Receive, send: Send) -> None:
    """以改寫後的 body 呼叫下游 app。"""
    scope = {**scope, "headers": _replace_content_length(scope["headers"], body)}
    sent = False

    async def replay() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    await app(scope, replay, send)


class DefaultTruncationMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_target(scope):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        if body is None:
            return
        await _forward(self.app, scope, _inject_default(body), receive, send)


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


def sigmoid(x: float) -> float:
    """分兩段計算，極端 logit 也不會 overflow。"""
    if x < 0:
        e = math.exp(x)
        return e / (1 + e)
    return 1 / (1 + math.exp(-x))


class ScoreCalibrationMiddleware:
    def __init__(self, app: ASGIApp, logit_sigma: float | None = None) -> None:
        if logit_sigma is None:
            logit_sigma = float(os.environ.get("LOGIT_SIGMA") or DEFAULT_LOGIT_SIGMA)
        if logit_sigma <= 0:
            raise ValueError(f"LOGIT_SIGMA must be positive, got {logit_sigma}")
        self.app = app
        self.logit_sigma = logit_sigma

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_target(scope):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        if body is None:
            return

        try:
            payload = orjson.loads(body)
        except orjson.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            # 無效請求交給 vLLM 回錯誤
            await _forward(self.app, scope, body, receive, send)
            return

        use_activation = payload.get("use_activation") is not False
        payload["use_activation"] = False
        await _forward(
            self.app, scope, orjson.dumps(payload), receive,
            self._calibrating_send(send, use_activation),
        )

    def _calibrating_send(self, send: Send, use_activation: bool) -> Send:
        start: Message | None = None
        chunks: list[bytes] = []

        async def wrapped(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            chunks.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            body = b"".join(chunks)
            if start["status"] == 200:
                body = self._rewrite(body, use_activation)
            headers = _replace_content_length(start["headers"], body)
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})

        return wrapped

    def _rewrite(self, body: bytes, use_activation: bool) -> bytes:
        try:
            payload = orjson.loads(body)
        except orjson.JSONDecodeError:
            return body
        if not isinstance(payload, dict):
            return body

        for field, key in (("results", "relevance_score"), ("data", "score")):
            for item in payload.get(field) or []:
                scaled = item[key] / self.logit_sigma
                item[key] = sigmoid(scaled) if use_activation else scaled
        return orjson.dumps(payload)
