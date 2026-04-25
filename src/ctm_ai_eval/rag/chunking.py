from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from ctm_ai_eval.rag.datamodels import Chunker, RagChunk, SpanToken


@dataclass
class CharChunker(Chunker):
    """Simplest possible chunking."""

    len_chars: int = 200

    @property
    @override
    def fingerprint(self) -> str:
        return f"{type(self).__name__}({self.len_chars})"

    @override
    def _chunk(self, docs: list[str]):
        chunks: list[RagChunk] = []
        texts: set[str] = set()  # for deduplication
        for doc_id, d in enumerate(docs):
            for k in range(0, len(d), self.len_chars):
                candidate = d[k : k + self.len_chars]

                if candidate in texts:
                    continue

                texts.add(candidate)
                chunks.append(
                    RagChunk(
                        id=len(chunks),
                        text=candidate,
                        doc_id=doc_id,
                        start=k,
                    ),
                )
        return chunks


@dataclass
class TokenChunker(Chunker):
    """Simple chunking."""

    len_tokens: int  # for example 2 sentences, 70 words, 300 chars
    tokenizer: Callable[[str], list[SpanToken]]

    @property
    @override
    def fingerprint(self) -> str:
        return f"{(self.tokenizer.__name__)}({self.len_tokens})"

    @override
    def _chunk(self, docs: list[str]):
        chunks: list[RagChunk] = []
        texts: set[str] = set()  # for deduplication
        for doc_id, d in enumerate(docs):
            tks = [t.text for t in self.tokenizer(d)]

            for k in range(0, len(tks), self.len_tokens):
                candidate = " ".join(tks[k : k + self.len_tokens])

                # avoid duplicates
                if candidate in texts:
                    continue

                texts.add(candidate)
                chunks.append(
                    RagChunk(
                        id=len(chunks),
                        text=candidate,
                        doc_id=doc_id,
                        start=k,
                    ),
                )
        return chunks
