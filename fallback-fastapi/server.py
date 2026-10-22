"""voyage-4-nano 的 CPU fallback embedding server（OpenAI `/v1/embeddings` 格式）。

主要 embedding 後端（vLLM）全部失效時，由 ai-gateway 改送到這裡。
行為與 vLLM 後端對齊：
- `input_type` 未指定時套用 document prompt；`query` 套用 query prompt；`none` 不加 prompt（可送 token ids）
- `dimensions` 只接受 256 / 512 / 1024 / 2048（Matryoshka），截斷後重新正規化
- `encoding_format` 支援 `float` 與 `base64`（little-endian float32）
- `usage.prompt_tokens` 含 prompt 前綴的 token 數

用法：uv run server.py [--host 127.0.0.1] [--port 12359]
"""

import argparse
import base64
import os
import threading
import time
import uuid

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("FALLBACK_MODEL_ID", "voyageai/voyage-4-nano")
SERVED_NAME = os.environ.get("FALLBACK_SERVED_NAME", "voyage-4-nano")
# CPU 上 bf16 實測比 fp32 快約 1.8 倍，與 GPU 後端的 cosine > 0.9999
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("FALLBACK_DTYPE", "bf16")]
DIMENSIONS = [256, 512, 1024, 2048]
INPUT_TYPES = ["query", "document", "none"]

model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device="cpu", model_kwargs={"dtype": DTYPE})
tokenizer = model.tokenizer
# torch 本身已用多執行緒推論，一次只跑一個 request 避免互搶 CPU
lock = threading.Lock()
app = FastAPI()


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
