from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class SpanToken:
    """When splitting text, track where the token comes from."""

    text: str
    start: int
    end: int


class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[SpanToken]: ...


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
    def identifier(self) -> str: ...

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

    def __call__(self, docs: list[str]):
        return self._chunk(docs)


@dataclass
class HaystackTarget:
    corpus_loader: Callable[[Path], list[str]]
    chunker: Chunker
    retriever: Retriever

    @property
    def fingerprint(self):
        names = self.corpus_loader.__name__, type(self.chunker).__name__, self.retriever.identifier
        return "_".join(names)


@dataclass
class HaystackExperimentSetup:
    corpus_source: str
    setup_hash: str
