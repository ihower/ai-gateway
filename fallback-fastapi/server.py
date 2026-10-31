"""ai-gateway 的 fallback server：各 service 的主要後端全部失效時，由 gateway 改送到這裡。

- `POST /v1/embeddings`：CPU 跑 voyage-4-nano（Sentence Transformers），行為與 vLLM 後端對齊：
  - `input_type` 未指定時套用 document prompt；`query` 套用 query prompt；`none` 不加 prompt（可送 token ids）
  - `dimensions` 只接受 256 / 512 / 1024 / 2048（Matryoshka），截斷後重新正規化
  - `encoding_format` 支援 `float` 與 `base64`（little-endian float32）
  - `usage.prompt_tokens` 含 prompt 前綴的 token 數
- `POST /v2/rerank`：轉送 Cohere Rerank API（`rerank-v4.0-fast`），回應改成 vLLM 的形狀
- `POST /v1/systemone`：轉送 TypeSafe 的 Jev API（SystemOne 格式，`jev-1.14.0`）

雲端 API 的 key 讀自同目錄的 `.env`（`COHERE_API_KEY`、`JEV_API_KEY`）；client 帶的 Authorization 不會轉送。

用法：uv run server.py [--host 127.0.0.1] [--port 12359]
"""

import argparse
import base64
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, Response
from sentence_transformers import SentenceTransformer

# .env 的值優先於已存在的環境變數
load_dotenv(Path(__file__).parent / ".env", override=True)

MODEL_ID = os.environ.get("FALLBACK_MODEL_ID", "voyageai/voyage-4-nano")
SERVED_NAME = os.environ.get("FALLBACK_SERVED_NAME", "voyage-4-nano")
# CPU 上 bf16 實測比 fp32 快約 1.8 倍，與 GPU 後端的 cosine > 0.9999
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("FALLBACK_DTYPE", "bf16")]
DIMENSIONS = [256, 512, 1024, 2048]
INPUT_TYPES = ["query", "document", "none"]

COHERE_URL = os.environ.get("COHERE_RERANK_URL", "https://api.cohere.com/v2/rerank")
COHERE_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v4.0-fast")
JEV_URL = os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
UPSTREAM_TIMEOUT = float(os.environ.get("FALLBACK_UPSTREAM_TIMEOUT", "60"))

model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device="cpu", model_kwargs={"dtype": DTYPE})
tokenizer = model.tokenizer
# torch 本身已用多執行緒推論，一次只跑一個 request 避免互搶 CPU
lock = threading.Lock()
http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        http = client
        yield


app = FastAPI(lifespan=lifespan)


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "BadRequestError"):
        self.status, self.message, self.kind = status, message, kind


@app.exception_handler(ApiError)
async def api_error(_: Request, e: ApiError):
    return JSONResponse({"error": {"message": e.message, "type": e.kind, "param": None, "code": e.status}}, e.status)


def is_token_ids(x) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(isinstance(t, int) and not isinstance(t, bool) for t in x)


def parse_inputs(raw, input_type: str) -> tuple[list[str] | None, list[list[int]] | None]:
    """回傳 (texts, token_ids) 其中之一。"""
    if isinstance(raw, str):
        return [raw], None
    if isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
        return raw, None
    if input_type == "none":
        if is_token_ids(raw):
            return None, [raw]
        if isinstance(raw, list) and raw and all(is_token_ids(x) for x in raw):
            return None, raw
        raise ApiError(400, "input must be a string, a list of strings, a list of token ids, or a list of token id lists")
    raise ApiError(400, "input must be a string or a list of strings when input_type adds a prompt; use input_type='none' for token ids")


def embed_token_ids(ids: list[list[int]]) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for seq in ids:
            t = torch.tensor([seq])
            features = model({"input_ids": t, "attention_mask": torch.ones_like(t)})
            out.append(features["sentence_embedding"][0].float().numpy())
    return np.stack(out)


def truncate(v: np.ndarray, dim: int | None) -> np.ndarray:
    v = v.astype(np.float32)
    if dim is not None and dim < v.shape[1]:
        v = v[:, :dim]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@app.post("/v1/embeddings")
def embeddings(body: dict):
    model_name = body.get("model")
    if model_name is not None and model_name not in (SERVED_NAME, MODEL_ID):
        raise ApiError(404, f"The model `{model_name}` does not exist.", "NotFoundError")
    input_type = body.get("input_type") or "document"
    if input_type not in INPUT_TYPES:
        raise ApiError(400, f"input_type must be one of {INPUT_TYPES}, got {input_type!r}")
    dim = body.get("dimensions")
    if dim is not None and dim not in DIMENSIONS:
        raise ApiError(
            400,
            f"Model '{SERVED_NAME}' only supports {DIMENSIONS} matryoshka dimensions, "
            "use other output dimensions will lead to poor results.",
        )
    fmt = body.get("encoding_format") or "float"
    if fmt not in ("float", "base64"):
        raise ApiError(400, f"encoding_format must be 'float' or 'base64', got {fmt!r}")
    if body.get("truncate_prompt_tokens") is not None:
        raise ApiError(400, "truncate_prompt_tokens is not supported by the CPU fallback")
    texts, ids = parse_inputs(body.get("input"), input_type)

    with lock:
        if texts is not None:
            prompt = model.prompts[input_type] if input_type != "none" else ""
            tokens = sum(len(x) for x in tokenizer([prompt + t for t in texts])["input_ids"])
            vectors = model.encode(texts, prompt=prompt, convert_to_numpy=True)
        else:
            tokens = sum(len(x) for x in ids)
            vectors = embed_token_ids(ids)
    vectors = truncate(vectors, dim)

    def encode(v: np.ndarray):
        if fmt == "base64":
            return base64.b64encode(v.astype("<f4").tobytes()).decode()
        return v.tolist()

    return {
        "id": f"embd-{uuid.uuid4().hex[:16]}",
        "object": "list",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "data": [{"index": i, "object": "embedding", "embedding": encode(v)} for i, v in enumerate(vectors)],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "completion_tokens": 0, "prompt_tokens_details": None},
    }


async def call_upstream(name: str, url: str, key_env: str, payload: dict) -> httpx.Response:
    """呼叫雲端 API。4xx 原樣交給呼叫端；連不上、逾時、5xx 轉成 502 / 504，讓 gateway 視為後端失敗。"""
    key = os.environ.get(key_env)
    if not key:
        raise ApiError(503, f"{key_env} is not set", "ServiceUnavailableError")
    try:
        resp = await http.post(url, json=payload, headers={"Authorization": f"Bearer {key}"})
    except httpx.TimeoutException:
        raise ApiError(504, f"{name} timed out", "UpstreamError")
    except httpx.HTTPError as e:
        raise ApiError(502, f"{name} request failed: {e}", "UpstreamError")
    if resp.status_code >= 500:
        raise ApiError(502, f"{name} returned HTTP {resp.status_code}", "UpstreamError")
    return resp


def passthrough(resp: httpx.Response) -> Response:
    return Response(resp.content, resp.status_code, media_type=resp.headers.get("content-type", "application/json"))


@app.post("/v2/rerank")
async def rerank(body: dict):
    """轉送 Cohere v2 rerank。回應與 vLLM 對齊：results 一律帶 `document.text`、`model` 為實際模型；
    另保留 Cohere 的 `meta`。Cohere 不回 token 數，因此沒有 `usage`。"""
    documents = body.get("documents")
    payload = {"model": COHERE_MODEL, "query": body.get("query"), "documents": documents}
    # 與 vLLM 一致：top_n 為 0 或未指定時回傳全部
    if body.get("top_n"):
        payload["top_n"] = body["top_n"]
    if body.get("max_tokens_per_doc") is not None:
        payload["max_tokens_per_doc"] = body["max_tokens_per_doc"]
    resp = await call_upstream("Cohere", COHERE_URL, "COHERE_API_KEY", payload)
    if resp.status_code != 200:
        return passthrough(resp)
    data = resp.json()
    results = [
        {"index": r["index"], "document": {"text": documents[r["index"]]}, "relevance_score": r["relevance_score"]}
        for r in data["results"]
    ]
    out = {"id": data.get("id") or f"rerank-{uuid.uuid4().hex[:16]}", "model": COHERE_MODEL, "results": results}
    if "meta" in data:
        out["meta"] = data["meta"]
    return out


@app.post("/v1/systemone")
async def systemone(body: dict):
    """轉送 Jev（同為 SystemOne 格式）；`model` 一律改為 Jev 的模型，回應原樣傳回。"""
    resp = await call_upstream("Jev", JEV_URL, "JEV_API_KEY", {**body, "model": JEV_MODEL})
    return passthrough(resp)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "cpu-fallback", "root": MODEL_ID}]}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12359)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
