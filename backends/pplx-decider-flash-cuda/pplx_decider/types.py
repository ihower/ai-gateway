"""Request and response schema (the Decisions API shape: one state, several named questions)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NotRequired, TypedDict, Union

from PIL import Image

JSONValue = Union[str, int, float, bool, None, list["JSONValue"], dict[str, "JSONValue"]]
#: State, instructions and option descriptions: plain text, or any JSON value (serialized with ``json.dumps``).
Content = Union[str, dict[str, JSONValue], list[JSONValue]]
#: A local path, a ``data:image/...;base64,`` URL, or a PIL image.
ImageInput = Union[str, Path, Image.Image]


class ChoiceQuestion(TypedDict):
    """Pick one of 1 to 255 named options. ``criteria`` maps each option name to its description (or ``None``)."""

    type: Literal["choice"]
    criteria: dict[str, Content | None]
    instructions: NotRequired[Content | None]


class NoulQuestion(TypedDict):
    """Yes/no question; the answer is P(true). ``criteria`` may describe the ``"true"`` and ``"false"`` outcomes."""

    type: Literal["noul"]
    criteria: NotRequired[dict[Literal["true", "false"], Content | None] | None]
    instructions: NotRequired[Content | None]


class ScoreQuestion(TypedDict):
    """Ordered scale of 2 to 255 levels; ``criteria[i]`` describes level ``i``. The answer is the expected level."""

    type: Literal["score"]
    criteria: list[Content]
    instructions: NotRequired[Content | None]


class LocateQuestion(TypedDict):
    """Extract spans matching the question key; optional ``instructions`` replaces that default label."""

    type: Literal["locate"]
    instructions: NotRequired[str | None]


Question = Union[ChoiceQuestion, NoulQuestion, ScoreQuestion, LocateQuestion]


class Request(TypedDict):
    """One decision context: every question is answered from the same state (and images)."""

    state: Content
    questions: dict[str, Question]
    images: NotRequired[list[ImageInput]]


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    legend: dict[str, Content | None]
    probabilities: dict[str, float]
    confidence: float


class Span(TypedDict):
    #: Character offsets into ``state`` (``end`` exclusive): ``state[start:end] == text``.
    start: int
    end: int
    text: str
    score: float


class LocateAnswer(TypedDict):
    type: Literal["locate"]
    #: Sorted by position; empty if nothing matches.
    spans: list[Span]


Answer = Union[ChoiceAnswer, NoulAnswer, ScoreAnswer, LocateAnswer]


class Usage(TypedDict):
    #: Non-padding tokens of every forward pass used for the request (text, template and image positions).
    input_tokens: int
    output_tokens: int
    #: Number of sequences run: one per question with ``mode="separate"``; one per request (more only for
    #: over-length requests) with ``mode="isolated"``.
    sequences: int


class Response(TypedDict):
    answers: dict[str, Answer]
    usage: Usage
