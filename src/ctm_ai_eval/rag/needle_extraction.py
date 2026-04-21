import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import tqdm

from ctm_ai_eval.common_config import LlmConfig
from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.datamodels import (
    ChunkCoupledNeedle,
    RagChunk,
    SpanNeedle,
    SpanToken,
    Tokenizer,
)
from ctm_ai_eval.rich_print import CONS


def sample_chunk_needles_verbatim(
    chunks: list[RagChunk],
    length_words: int = 5,
    max_count: int | None = None,
) -> list[ChunkCoupledNeedle]:
    """Extract a few samples."""
    all_needles: list[ChunkCoupledNeedle] = []
    for c in chunks:
        words = [t.text for t in text_processing.tokenize_words(c.text)]
        for start in range(0, len(words) - length_words, length_words):
            candidate = " ".join(words[start : start + length_words])
            # only include if exact match to original
            if candidate in c.text:
                all_needles.append(ChunkCoupledNeedle(chunk_id=c.id, text=candidate))

    if max_count:
        assert max_count < len(all_needles)
        idxs: list[int] = np.linspace(0, len(all_needles) - 1, max_count, dtype=int).tolist()
        return [all_needles[i] for i in idxs]

    return all_needles


@dataclass
class LlmNeedleSampler:
    system_prompt: str
    user_template: str
    model: str = "gemma3:270m"
    chat_url: str = "http://localhost:11434/api/chat"

    # ---- public API ----

    def sample_all(
        self,
        chunks: list[RagChunk],
        max_count: int | None = None,
    ) -> Iterator[ChunkCoupledNeedle]:

        count = 0

        for c in chunks:
            query = self._generate(c.text)

            if query is None:
                continue

            yield ChunkCoupledNeedle(chunk_id=c.id, text=query)
            count += 1
            if max_count and count > max_count:
                break  # collected enough, stop

    # ---- LLM call ----

    def _generate(self, chunk_text: str) -> str | None:
        prompt = self.user_template.format(chunk=chunk_text)

        r = requests.post(
            self.chat_url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        r.raise_for_status()

        text = r.json()["message"]["content"]
        assert isinstance(text, str), f"unexpected message content: {type(text)}"
        return self._postprocess(text)

    # ---- output filtering ----

    def _postprocess(self, text: str) -> str | None:
        text = text.strip()

        if not text:
            return None

        # allow model to skip
        if text.lower() in {"none", "skip", "n/a", "skipping"}:
            return None

        return text


def sample_needles_llm(
    chunks: list[RagChunk],
    prompt_dir: Path,
    mode: Literal["paraphrase", "query"],
    *,
    max_count: int | None = None,
    verbose: bool = False,
):

    prompt_sys = (prompt_dir / f"needle_{mode}_sys.txt").read_text()
    prompt_user = (prompt_dir / f"needle_{mode}.jinja").read_text()

    sampler = LlmNeedleSampler(prompt_sys, prompt_user, model="rnj-1:8b")

    chunks_by_id = {c.id: c for c in chunks}
    all_needles: list[ChunkCoupledNeedle] = []
    pbar = tqdm.tqdm(desc="sampling needles...", total=max_count, disable=verbose, ncols=0)
    for n in sampler.sample_all(chunks, max_count):
        chunk = chunks_by_id[n.chunk_id]
        sim = text_processing.similarity(n.text, chunk.text)
        discard = sim > 0.25
        if verbose:
            CONS.print(f"\nchunk{chunk.id}", style="bold")
            CONS.print(chunk.text, style="dim")
            CONS.print("-->", n.text, style="italic", end=" ")
            CONS.print(f"{sim=:.1%}", style="red" if discard else "blue", highlight=False)

            if discard:
                print("skip (too similar)")

        if not discard:
            all_needles.append(n)

        pbar.update()

    return all_needles


@dataclass
class SpanRephraser:
    """Rephrase or skip."""

    system_prompt: str
    user_template: str
    llm: LlmConfig
    chat_url: str = "http://localhost:11434/api/chat"

    # ---- public API ----

    def sample_all(
        self,
        raw_needles: list[SpanNeedle],
        max_count: int | None = None,
    ) -> Iterator[SpanNeedle]:

        count = 0

        for n in raw_needles:
            query = self._generate(n.query)

            if query is None:
                continue

            yield SpanNeedle(
                doc_id=n.doc_id,
                start_char=n.start_char,
                end_char=n.end_char,
                query=query,
            )
            count += 1
            if max_count and count > max_count:
                break  # collected enough, stop

    # ---- LLM call ----

    def _generate(self, chunk_text: str) -> str | None:
        prompt = self.user_template.format(chunk=chunk_text)

        r = requests.post(
            self.chat_url,
            json={
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "think": self.llm.think,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        r.raise_for_status()

        text = r.json()["message"]["content"]
        assert isinstance(text, str), f"unexpected message content: {type(text)}"
        return self._postprocess(text)

    # ---- output filtering ----

    def _postprocess(self, text: str) -> str | None:
        text = text.strip()

        if not text:
            return None

        # allow model to skip
        if text.lower() in {"none", "skip", "n/a", "skipping"}:
            return None

        return text


def sample_span_needles_verbatim(
    docs: list[str],
    tokenizer: Tokenizer,
    *,
    max_count: int,
    min_tokens: int = 10,
    max_tokens: int = 60,
    rng: random.Random | None = None,
) -> list[SpanNeedle]:
    if rng is None:
        rng = random.Random(0)

    needles: list[SpanNeedle] = []

    # Pre-tokenize all docs
    tokenized_docs: list[tuple[int, str, list[SpanToken]]] = []
    for i, doc in enumerate(docs):
        tokens = tokenizer(doc)
        if len(tokens) >= min_tokens:
            tokenized_docs.append((i, doc, tokens))

    if not tokenized_docs:
        raise ValueError("No documents with enough tokens")

    attempts = 0
    max_attempts = max_count * 20

    while len(needles) < max_count and attempts < max_attempts:
        attempts += 1
        # pick a random doc
        doc_id, doc, tokens = rng.choice(tokenized_docs)

        # pick a start and end (in tokens)
        span_len = rng.randint(min_tokens, max_tokens)
        if len(tokens) < span_len:
            continue
        start_idx = 0 if len(tokens) <= span_len else rng.randint(0, len(tokens) - span_len)
        end_idx = start_idx + span_len

        # convert tokens to chars
        start_char = tokens[start_idx].start
        end_char = tokens[end_idx - 1].end

        # get text from original source
        text = doc[start_char:end_char]

        # basic filtering
        if len(text.strip()) < 20:
            continue

        needles.append(
            SpanNeedle(
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                query=text,
            )
        )

    return needles
