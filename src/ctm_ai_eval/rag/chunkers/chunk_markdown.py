import re
from dataclasses import dataclass
from typing import override

from ctm_ai_eval.rag.chunkers.validation import validate_chunk_pos
from ctm_ai_eval.rag.datamodels import Chunker, RagChunk


def _strip_yaml_front_matter(text: str) -> tuple[str, int]:
    """Returns (stripped_text, offset) where offset is how many chars were removed from start."""
    m = re.match(r"(?s)\A---\n.*?\n---\n", text)
    if m:
        offset = m.end()
        return text[offset:], offset
    return text, 0


@dataclass
class MarkdownChunker(Chunker):
    """~Original implementation from CTM project, with less metadata."""

    max_chars: int
    overlap_chars: int

    @property
    @override
    def fingerprint(self) -> str:
        return f"{type(self).__name__}(max={self.max_chars}, overlap={self.overlap_chars})"

    @override
    def _chunk(self, docs: list[str]) -> list[RagChunk]:
        all_chunks: list[RagChunk] = []
        for d_id, d in enumerate(docs):
            for text, start in self._chunk_single(d):
                # Correct any off-by-N in start by searching near the reported position
                search_window = 20  # how far to look around the reported start
                lo = max(0, start - search_window)
                actual_start = d.find(text, lo)
                if actual_start == -1:
                    # fallback: search whole doc
                    actual_start = d.find(text)
                c = RagChunk(len(all_chunks), text, doc_id=d_id, start=actual_start)
                validate_chunk_pos(d, c)
                all_chunks.append(c)

        return all_chunks

    def _chunk_single(self, text: str) -> list[tuple[str, int]]:
        stripped, front_offset = _strip_yaml_front_matter(text)

        # Find how many leading whitespace chars .strip() would remove
        lstrip_count = len(stripped) - len(stripped.lstrip())
        base_offset = front_offset + lstrip_count

        stripped = stripped.strip()
        if not stripped:
            return []

        # Split on headings; keep headings.
        # re.split with groups produces: [preamble, hlevel, htitle, body, hlevel, htitle, body, ...]
        parts = re.split(r"(?m)^(#{1,6})\s+(.+?)\s*$", stripped)
        chunks: list[tuple[str, int]] = []

        preamble = parts[0]
        preamble_offset = base_offset  # preamble starts right at base_offset
        if preamble.strip():
            chunks.extend(self._split_long(preamble, preamble_offset))

        # Walk through (hlevel, htitle, body) triples, tracking position in `stripped`
        cursor = len(parts[0])  # characters consumed so far in `stripped`
        i = 1
        while i + 2 < len(parts):
            hlevel = parts[i]
            htitle = parts[i + 1]
            body = parts[i + 2]

            # Each heading was reconstructed by re.split; we need to account for the
            # full heading line that was consumed.  The original line looked like:
            #   {hlevel} {htitle}\n
            heading_line_len = len(hlevel) + 1 + len(htitle) + 1  # "## Title\n"
            cursor += heading_line_len

            body_offset = base_offset + cursor
            if body.strip():
                chunks.extend(self._split_long(body, body_offset))

            cursor += len(body)
            i += 3

        return chunks

    def _split_long(self, body: str, body_offset: int) -> list[tuple[str, int]]:
        """Split body into windows of max_chars with overlap, yielding (text, original_start)."""
        out: list[tuple[str, int]] = []

        # Account for leading whitespace that .strip() removes
        lstrip_count = len(body) - len(body.lstrip())
        body = body.strip()

        if not body:
            return out

        adjusted_offset = body_offset + lstrip_count

        if len(body) <= self.max_chars:
            out.append((body, adjusted_offset))
            return out

        start = 0
        while start < len(body):
            end = min(len(body), start + self.max_chars)
            window = body[start:end].strip()
            if window:
                # Find where this window actually starts within body (after inner lstrip)
                window_lstrip = len(body[start:end]) - len(body[start:end].lstrip())
                out.append((window, adjusted_offset + start + window_lstrip))
            if end == len(body):
                break
            start = max(0, end - self.overlap_chars)

        return out
