from dataclasses import dataclass
from typing import Any, cast, override

import faiss
import numpy as np
import requests
import tqdm

from ctm_ai_eval.rag.datamodels import RagChunk, RetrievalResult, Retriever

# ---- embedding helpers ----


@dataclass
class Embedder:
    model: str = "nomic-embed-text"
    embeddings_url: str = "http://localhost:11434/v1/embeddings"
    api_key: str = "ignored"
    time_out: int = 60

    def __call__(
        self,
        texts: list[str],
        batch_size: int = 256,
        *,
        verbose: bool = False,
    ) -> np.ndarray:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_vecs = []
        batch_idxs = list(range(0, len(texts), batch_size))
        for i in tqdm.tqdm(
            batch_idxs, desc="embedding", unit="batch", ncols=0, disable=not verbose
        ):
            batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": batch}

            r = requests.post(
                self.embeddings_url,
                headers=headers,
                json=payload,
                timeout=self.time_out,
            )
            r.raise_for_status()

            j = r.json()
            data = j.get("data", [])
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Unexpected embeddings response: {len(data)=}, expected {len(batch)}"
                )

            all_vecs.extend([item["embedding"] for item in data])

        arr = np.array(all_vecs, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Embeddings array has unexpected shape: {arr.shape}")
        return arr


# ---- FAISS retriever ----


class FaissRetriever(Retriever):
    embedder: Embedder

    def __init__(self, embedder: Embedder | None = None) -> None:
        super().__init__()
        # default if no embedder
        self.embedder = embedder if embedder is not None else Embedder()

    @property
    @override
    def identifier(self) -> str:
        return f"faiss_{self.embedder.model}".replace(".", "")

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.embedder.model})"

    @override
    def ingest(self, chunks: list[RagChunk]):
        self.chunks = chunks

        texts = [c.text for c in chunks]
        X = self.embedder(texts, verbose=True)

        # normalize for cosine similarity
        faiss.normalize_L2(X)

        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product
        # Faiss typing 😥...
        self.index.add(X)  # pyright: ignore[reportCallIssue]

    @override
    def _retrieve(self, query: str, k: int):
        assert self.index is not None
        assert self.chunks is not None

        q = self.embedder([query])
        faiss.normalize_L2(q)

        # Faiss typing 😥...
        scores, indices = cast(Any, self.index.search(q, k))  # pyright: ignore[reportCallIssue]

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(idx)],
                    score=float(score),
                )
            )

        return results
