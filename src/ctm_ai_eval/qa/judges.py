import getpass
import hashlib
import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import requests

from ctm_ai_eval.qa.datamodels import EvalCase, FloatTraceMetric, QaQuestion
from ctm_ai_eval.utils import hashing


class Judge(Protocol):
    name: str

    def evaluate(self, case: EvalCase) -> FloatTraceMetric: ...


@dataclass
class IsConcise:
    name: str = "concise"
    max_words: int = 20

    def evaluate(self, case: EvalCase) -> FloatTraceMetric:
        trace = case.trace
        words = len(trace.answer.split()) if trace.answer else 0

        return FloatTraceMetric(
            name=self.name,
            run_id=trace.run_id,
            example_id=trace.example_id,
            # 1 for ok, 0 for too long, -1 for error
            score=-1 if words == 0 else int(words <= self.max_words),
            metric_config={"max_words": self.max_words},
            details={"words": words},
        )


@dataclass
class HumanRatingJudge:
    """Interactively ask for feedback, or load from cache."""

    name: str = "human_rating"
    cache_path: Path = Path("tmp/human_rating_cache.json")
    _cache: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.cache_path.exists():
            # init empty cache
            self.cache_path.write_text("{}")
        # load cache
        self._cache = json.loads(self.cache_path.read_text())

    def evaluate(self, case: EvalCase) -> FloatTraceMetric:
        trace = case.trace
        question = case.question
        assert trace.answer is not None, "missing answer!"

        key = self._fingerprint(question, trace.answer)

        if key in self._cache:
            rating = self._cache[key]
        else:
            rating = self._ask_user(question, trace.answer)
            self._cache[key] = rating
            self.cache_path.write_text(json.dumps(self._cache, indent=2))

        return FloatTraceMetric(
            name=self.name,
            run_id=trace.run_id,
            example_id=trace.example_id,
            score=rating,
            metric_config={"user": getpass.getuser()},
        )

    def _ask_user(self, question: QaQuestion, answer: str) -> float:
        _ = subprocess.run(
            "cls" if platform.system == "Windows" else "clear",
            check=True,
        )

        print(f"\n=== Question ===\n{question.to_question_string()}\n")
        print(f"\n=== Expected answer ===\n{question.answer}\n")
        print(f"\n=== Answer ===\n{answer}\n")

        v = float(input("-> rating? [0,1] "))
        assert 0 <= v <= 1, "range!"
        return v

    def _fingerprint(self, question: QaQuestion, answer: str) -> str:
        s = (
            question.to_question_string()
            + "\nEXPECTED:\n"
            + question.answer
            + "\nANSWER:\n"
            + (answer or "")
        )
        return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class LLMJudge:
    model: str
    sys_prompt: str
    msg_template: str
    # where to send requests
    server_url: str = "http://127.0.0.1:11434/v1"
    route: str = "chat/completions"
    api_key: str = "APIKEY"
    # constant
    name: str = "llm_rating"

    def evaluate(self, case: EvalCase) -> FloatTraceMetric:

        trace = case.trace
        question = case.question

        message = self.msg_template.format(
            question=question.to_question_string(),
            expected_answer=question.answer,
            answer=trace.answer,
        )

        payload = {
            "model": self.model,
            # NOTE: zero temp for consistent grading
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": message},
            ],
        }

        r = requests.post(
            f"{self.server_url}/{self.route}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )

        r.raise_for_status()

        text = r.json()["choices"][0]["message"]["content"]

        score = self._extract_score(text)

        return FloatTraceMetric(
            name=self.name,
            run_id=trace.run_id,
            example_id=trace.example_id,
            score=score,
            metric_config={
                "model": self.model,
                "prompts_hash": hashing.stable_hash(
                    {"sys": self.sys_prompt, "msg": self.msg_template},
                ),
            },
            details={"raw_response": text},
        )

    def _extract_score(self, text: str) -> float:
        m = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", text)
        if not m:
            return -1
        return float(m.group())
