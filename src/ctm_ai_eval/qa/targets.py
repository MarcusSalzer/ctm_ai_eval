import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import requests
from pydantic import BaseModel

from ctm_ai_eval.qa.datamodels import ApiEvalResponse
from ctm_ai_eval.rag.datamodels import HaystackTarget


class ChatTargetConfig(BaseModel):
    model: str
    system_prompt_id: str
    temperature: float = 0.0
    max_tokens: int | None = 512


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
        headers: Mapping[str, str | bytes],
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

    def _build_messages(self, prompt: str, system_prompt: str | None):
        messages: list[Mapping[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(self, prompt: str, system_prompt: str | None) -> dict[str, object]:
        return {
            "model": self.chat_config.model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": self.chat_config.temperature,
            "max_tokens": self.chat_config.max_tokens,
        }

    @abstractmethod
    def ask(self, prompt: str, system_prompt: str | None) -> ApiEvalResponse: ...


@dataclass
class OpenAIChatTarget(ApiTarget):
    chat_config: ChatTargetConfig
    api_key: str = "APIKEY"
    route: str = "chat/completions"
    server_url: str = "http://127.0.0.1:11434/v1"

    def _build_headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @override
    def ask(self, prompt: str, system_prompt: str | None) -> ApiEvalResponse:
        return self._ask(self._build_payload(prompt, system_prompt), self._build_headers())


@dataclass
class RagApiTarget(ApiTarget):
    chat_config: ChatTargetConfig
    haystack: HaystackTarget
    docs_dir: Path
    top_k: int = 5
    api_key: str = "APIKEY"
    route: str = "chat/completions"
    server_url: str = "http://127.0.0.1:11434/v1"

    _ingested: bool = field(default=False, init=False, repr=False)

    def _build_headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def ensure_ingested(self) -> None:
        if self._ingested:
            return
        docs = self.haystack.loader(self.docs_dir)
        chunks = self.haystack.chunker(docs)
        self.haystack.retriever.ingest(chunks)
        self._ingested = True

    @override
    def ask(self, prompt: str, system_prompt: str | None) -> ApiEvalResponse:
        self.ensure_ingested()
        chunks = self.haystack.retriever(prompt, k=self.top_k)
        context = "\n\n---\n\n".join(c.chunk.text for c in chunks)
        augmented = f"Context:\n{context}\n\nQuestion:\n{prompt}"
        res = self._ask(self._build_payload(augmented, system_prompt), self._build_headers())
        res.retrieved = chunks
        return res
