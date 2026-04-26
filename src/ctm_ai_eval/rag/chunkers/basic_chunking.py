from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from ctm_ai_eval.rag.chunkers.validation import validate_chunk_pos
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
                c = RagChunk(
                    id=len(chunks),
                    text=candidate,
                    doc_id=doc_id,
                    start=k,
                )
                validate_chunk_pos(d, c)
                chunks.append(c)
        return chunks


@dataclass
class TokenChunker(Chunker):
    """Simple chunking with optional overlap."""

    len_tokens: int
    overlap_tokens: int
    tokenizer: Callable[[str], list[SpanToken]]

    @property
    @override
    def fingerprint(self) -> str:
        return f"{self.tokenizer.__name__}(len={self.len_tokens}, overlap={self.overlap_tokens})"

    @override
    def _chunk(self, docs: list[str]):
        chunks: list[RagChunk] = []
        stride = self.len_tokens - self.overlap_tokens

        assert stride > 0, "overlap_tokens must be less than len_tokens"

        for doc_id, d in enumerate(docs):
            tks = self.tokenizer(d)

            for k in range(0, len(tks), stride):
                window = tks[k : k + self.len_tokens]

                start = window[0].start  # char offset in original doc
                end = window[-1].end
                c = RagChunk(
                    id=len(chunks),
                    text=d[start:end],
                    doc_id=doc_id,
                    start=start,
                )

                validate_chunk_pos(d, c)

                chunks.append(c)
        return chunks
