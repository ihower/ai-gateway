"""SystemOne request ⇄ vLLM classify 輸入／輸出的轉換（pplx-decider v1.3，protocol v4、separate mode）。

- 編碼完全沿用官方 protocol.Encoder：每題一個 sequence（context + 該題 + <|fim_suffix|>），
  readout marker 一定是最後一個 token，所以 vLLM 的 LAST pooling 剛好取到 marker 的 hidden state。
- 回應沿用官方 protocol.answer：只取該題有效選項的 logits，除以校準溫度後 softmax。
- 只支援文字；images 與 locate 回 400（與 llama.cpp 版相同，未載入 mmproj / locate adapter）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from pplx_decider.protocol import PROTOCOL, Encoded, Encoder, LengthError, RequestError, answer, load_json

__all__ = ["DecisionCodec", "RequestError", "LengthError"]


class DecisionCodec:
    def __init__(self, model_dir: str | Path, max_length: int | None) -> None:
        model_dir = Path(model_dir)
        if load_json(model_dir / "joint_config.json") != PROTOCOL:
            raise ValueError("checkpoint 不是 protocol v4（joint-chunked-readout-v1）")
        config = load_json(model_dir / "decision_config.json")
        self.temperature = float(config["temperature"])
        self.max_length = max_length
        processor = AutoProcessor.from_pretrained(str(model_dir))
        self.encoder = Encoder(processor, config["codes"])

    def plan(self, request: Any) -> list[Encoded]:
        """驗證 request 並編成 separate mode 的 sequences（每題一個，順序同 questions）。"""
        if not isinstance(request, dict):
            raise RequestError("request body 必須是 JSON object")
        if request.get("images"):
            raise RequestError("此後端只支援文字，不接受 images")
        questions = request.get("questions")
        if isinstance(questions, dict) and any(
            isinstance(q, dict) and q.get("type") == "locate" for q in questions.values()
        ):
            raise RequestError("此後端不支援 locate 題型")
        # Encoder.plan 只看 state / questions；其他欄位（model 等）忽略
        chunks = self.encoder.plan(
            {"state": request.get("state"), "questions": questions} if "state" in request else request,
            self.max_length,
            mode="separate",
        )
        for chunk in chunks:
            if int(chunk.input_ids[-1]) != self.encoder.marker_id:
                raise ValueError("readout marker 不在 sequence 最後")
        return chunks

    def temperature_of(self, request: dict[str, Any]) -> float:
        value = request.get("temperature")
        scale = self.temperature if value is None else value
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise RequestError("temperature 必須是正的有限數")
        return float(scale)

    def respond(
        self, request: dict[str, Any], chunks: list[Encoded], logits: list[list[float]], model: str | None
    ) -> dict[str, Any]:
        """每個 sequence 的 readout logits（255 維）→ SystemOne response。"""
        if len(logits) != len(chunks):
            raise ValueError(f"logits 數量 {len(logits)} 與 sequence 數 {len(chunks)} 不符")
        scale = self.temperature_of(request)
        questions = request["questions"]
        answers: dict[str, Any] = {}
        for chunk, row in zip(chunks, logits, strict=True):
            (key,), (count,) = chunk.keys, chunk.counts
            values = torch.tensor(row[:count], dtype=torch.float32)
            answers[key] = answer(questions[key], (values / scale).softmax(-1).tolist())
        body: dict[str, Any] = {}
        if model is not None:
            body["model"] = model
        body["answers"] = {key: answers[key] for key in questions}
        body["usage"] = {
            "input_tokens": sum(chunk.length for chunk in chunks),
            "output_tokens": 0,
            "sequences": len(chunks),
        }
        return body
