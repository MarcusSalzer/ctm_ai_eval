import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import requests
from pydantic import BaseModel

from ctm_ai_eval.qa.datamodels import ApiEvalResponse


class ChatTargetConfig(BaseModel):
    model: str
    system_prompt_id: str
    temperature: float = 0.0
    max_tokens: int | None = 512


@dataclass
class ApiTarget(ABC):
    chat_config: ChatTargetConfig
    server_url: str
    route: str

    @override
    def __str__(self) -> str:
        return f"ApiTarget({self.server_url}, {self.route})"

    def _ask(
        self,
        payload: dict[str, object],
        headers: dict[str, str | bytes],
    ) -> ApiEvalResponse:
        t0 = time.time()

        r = requests.post(
            f"{self.server_url}/{self.route}",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        latency_ms = int((time.time() - t0) * 1000)
        raw = r.json()
        return ApiEvalResponse(
            raw=raw,
            latency_ms=latency_ms,
            text=raw["choices"][0]["message"]["content"],
        )

    @abstractmethod
    def ask(self, prompt: str, system_prompt: str | None) -> ApiEvalResponse: ...


@dataclass
class OpenAIChatTarget(ApiTarget):
    api_key: str = "APIKEY"
    route: str = "chat/completions"

    @override
    def ask(self, prompt: str, system_prompt: str | None) -> ApiEvalResponse:

        messages: list[dict[str, str]] = []

        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self._ask(
            {
                "model": self.chat_config.model,
                "messages": messages,
                "temperature": self.chat_config.temperature,
                "max_tokens": self.chat_config.max_tokens,
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )


@dataclass
class OllamaChatTarget(OpenAIChatTarget):
    server_url: str = "http://127.0.0.1:11434/v1"
