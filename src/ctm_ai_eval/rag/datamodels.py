from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ctm_ai_eval.utils.hashing import stable_hash


@dataclass
class SpanToken:
    """When splitting text, track where the token comes from."""

    text: str
    start: int
    end: int


@dataclass
class RagChunk:
    """Something from a document."""

    id: int
    text: str
    doc_id: int
    start: int

    @property
    def end(self):
        return self.start + len(self.text)


@dataclass
class RetrievalResult:
    """Retriever gives topk of these."""

    chunk: RagChunk
    score: float


@dataclass
class ChunkCoupledNeedle:
    """Needle extracted from a chunk: easy to evaluate, dependent on chunker."""

    chunk_id: int
    text: str


@dataclass
class SpanNeedle:
    """Needle extracted from a document: Might not exactly overlap with chunks."""

    doc_id: int
    # span in original text (not same lenght as "text" if paraphrased)
    start_char: int
    end_char: int
    # what is shown to retriever
    query: str


class Retriever(ABC):
    """Anything that can produce results given a query."""

    chunks: list[RagChunk] | None

    @property
    @abstractmethod
    def fingerprint(self) -> str: ...

    @abstractmethod
    def ingest(self, chunks: list[RagChunk]): ...

    @abstractmethod
    def _retrieve(self, query: str, k: int) -> list[RetrievalResult]: ...

    def __call__(self, query: str, k: int):
        return self._retrieve(query, k)


class Chunker(ABC):
    """Anything that can split up documents."""

    @abstractmethod
    def _chunk(self, docs: list[str]) -> list[RagChunk]: ...

    @property
    @abstractmethod
    def fingerprint(self) -> str: ...

    def __call__(self, docs: list[str]):
        return self._chunk(docs)


@dataclass
class HaystackTarget:
    loader: Callable[[Path], list[str]]
    chunker: Chunker
    retriever: Retriever

    @property
    def fingerprint_dict(self) -> dict[str, str]:
        return {
            "loader": self.loader.__name__,
            "chunker": self.chunker.fingerprint,
            "retriever": self.retriever.fingerprint,
        }

    @property
    def hash_id(self):
        return stable_hash(self.fingerprint_dict, length=16)

    @property
    def fingerprint_tuple(self):
        return tuple(self.fingerprint_dict.values())


@dataclass
class HaystackExperimentSetup:
    corpus_source: str
    setup_hash: str
