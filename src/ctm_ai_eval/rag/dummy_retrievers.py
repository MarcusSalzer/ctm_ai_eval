import random
from dataclasses import dataclass
from typing import override

from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.datamodels import RagChunk, RetrievalResult, Retriever


@dataclass
class DummyRetriever(Retriever):
    @property
    @override
    def fingerprint(self) -> str:
        return "dummy"

    @override
    def ingest(self, chunks: list[RagChunk]):
        self.chunks = chunks

    @override
    def _retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Retrieve a few chunks"""
        _ = query  # ignore
        assert self.chunks is not None
        chunks = random.sample(self.chunks, k)
        return [RetrievalResult(c, 0.5) for c in chunks]


@dataclass
class SimpleExactRetriever(Retriever):
    weight_lcs: float = 0.3

    @property
    @override
    def fingerprint(self) -> str:
        return f"exact{self.weight_lcs}".replace(".", "")

    @override
    def ingest(self, chunks: list[RagChunk]):
        self.chunks = chunks

    def score(self, query: str, text: str) -> float:
        q_norm = text_processing.normalize(query)
        t_norm = text_processing.normalize(text)

        # --- token overlap ---
        q_tokens = set(t.text for t in text_processing.tokenize_words(q_norm))
        t_tokens = set(t.text for t in text_processing.tokenize_words(t_norm))

        if not q_tokens:
            return 0.0

        overlap = len(q_tokens & t_tokens) / len(q_tokens)

        w = self.weight_lcs
        # --- longest substring bonus ---
        lcs = text_processing.longest_common_substring(q_norm, t_norm) if w > 0 else 0
        lcs_norm = lcs / max(len(q_norm), 1)

        # --- combine ---
        return (1 - w) * overlap + w * lcs_norm

    @override
    def _retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if self.chunks is None:
            raise RuntimeError("ingest first!")
        scored = [RetrievalResult(chunk=c, score=self.score(query, c.text)) for c in self.chunks]

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
