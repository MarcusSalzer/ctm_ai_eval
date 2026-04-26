from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ctm_ai_eval.rag.datamodels import RetrievalResult


class QaQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str  # stable ids for reproducibility
    author: str  # person or model name
    question: str
    context: str  # additional information to show with the question
    answer: str
    source: str  # document path etc?

    def to_question_string(self) -> str:
        if self.context:
            return self.context + "\n" + self.question
        return self.question


class ApiEvalResponse(BaseModel):
    """response and timing."""

    raw: dict[str, Any]
    latency_ms: int
    text: str | None
    # optional, for rag targets
    retrieved: list[RetrievalResult] | None = None


class EvalTrace(BaseModel):
    """Config and output of a single ezxample in a run."""

    run_id: str
    dataset_name: str
    example_id: str
    server_url: str
    route: str
    answer: str | None
    latency_ms: int
    target_cfg: dict[str, object]
    rag_cfg: dict[str, str] | None
    local_host: str  # Where did the eval run


class FloatTraceMetric(BaseModel):
    name: str
    run_id: str
    example_id: str
    score: float
    metric_config: dict[str, int | str | bool]
    details: dict[str, int | str | bool] = Field(default_factory=dict)

    @property
    def fingerprint(self) -> tuple[str, str, str]:
        """Defining inputs. Used to decide if it is already computed."""
        return (self.run_id, self.example_id, self.name)


@dataclass
class EvalCase:
    """What can be passed to any judge."""

    trace: EvalTrace
    question: QaQuestion
