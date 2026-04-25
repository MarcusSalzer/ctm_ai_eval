"""
Pick out "needles" from chunks, and evaluate retrieval.

Three levels:

- Verbatim
- Paraphrased
- Implicit (conceptual similarity, mor of a question?)


Evaluation

- Quantitative: recall@k, Mean Reciprocal Rank (how early does correct chunk appear)
- Qualitative: LLM judge for relevance

"""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Literal

import polars as pl
import tqdm

from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.ai_retriever import Embedder, FaissRetriever
from ctm_ai_eval.rag.chunking import TokenChunker
from ctm_ai_eval.rag.datamodels import ChunkCoupledNeedle, RagChunk, Retriever
from ctm_ai_eval.rag.dummy_retrievers import DummyRetriever, SimpleExactRetriever
from ctm_ai_eval.rag.metrics import (
    recall_at_k,
    recall_at_k_doc,
    reciprocal_rank,
    reciprocal_rank_doc,
)
from ctm_ai_eval.rag.needle_extraction import sample_chunk_needles_verbatim, sample_needles_llm
from ctm_ai_eval.rich_print import CONS
from ctm_ai_eval.utils.hashing import stable_hash
from ctm_ai_eval.utils.io_util import load_all_md

PROMPT_DIR = Path("./assets/prompts")
MAX_NEEDLES = 300
TARGETS: list[Retriever] = [
    DummyRetriever(),
    FaissRetriever(Embedder("nomic-embed-text")),
    FaissRetriever(Embedder("qwen3-embedding:4b")),
    FaissRetriever(Embedder("qwen3-embedding:0.6b")),
    FaissRetriever(Embedder("all-minilm:33m")),
    FaissRetriever(Embedder("embeddinggemma")),
    FaissRetriever(Embedder("nomic-embed-text-v2-moe")),
    SimpleExactRetriever(weight_lcs=0.3),
    SimpleExactRetriever(weight_lcs=0.0),
]
CORPUS_PATH = Path("~/projects/ctm_project/convertingappendixdontpanik/").expanduser()
assert CORPUS_PATH.exists()


def _run_target(
    retr: Retriever, chunks: list[RagChunk], needles: list[ChunkCoupledNeedle], k_max: int
):
    retr.ingest(chunks)

    chunks_by_id = {c.id: c for c in chunks}
    all_res = []
    p = tqdm.tqdm(needles, ncols=0, desc="needles")
    for n in p:
        t0 = perf_counter()
        retrieved = retr(n.text, k_max)
        t_retr = perf_counter() - t0
        r = {
            "chunk_id": n.chunk_id,
            "t_retr": t_retr,
            "recall@1_chunk": recall_at_k(retrieved, n.chunk_id, k=1),
            "recall@5_chunk": recall_at_k(retrieved, n.chunk_id, k=5),
            "recall@10_chunk": recall_at_k(retrieved, n.chunk_id, k=10),
            "recall@1_doc": recall_at_k_doc(retrieved, chunks_by_id[n.chunk_id].doc_id, k=1),
            "recall@5_doc": recall_at_k_doc(retrieved, chunks_by_id[n.chunk_id].doc_id, k=5),
            "rr_chunk": reciprocal_rank(retrieved, n.chunk_id),
            "rr_doc": reciprocal_rank_doc(retrieved, chunks_by_id[n.chunk_id].doc_id),
        }

        all_res.append(r)

    result_df = pl.DataFrame(all_res)
    return result_df


def _experiment(mode: Literal["query", "paraphrase", "verbatim"]):

    docs = load_all_md(CORPUS_PATH)
    chunks = TokenChunker(80, text_processing.tokenize_words)(docs)
    if mode == "verbatim":
        needles = sample_chunk_needles_verbatim(chunks, max_count=MAX_NEEDLES)
    else:
        needles = sample_needles_llm(chunks, PROMPT_DIR, mode, max_count=MAX_NEEDLES)

    setup_hash = stable_hash(
        {
            "corpus_path_stem": CORPUS_PATH.stem,
            "chunks": [r.text for r in chunks],
            "needles": [r.text for r in needles],
        },
        length=8,
    )
    exp_id = f"{mode}_{setup_hash}"

    CONS.print(
        f"running experiment '{exp_id}' "
        + f"({len(TARGETS)} targets, {len(chunks)} chunks, {len(needles)} needles)",
        style="bold",
        justify="center",
    )

    # store setup
    pl.DataFrame([asdict(r) for r in chunks]).write_parquet("tmp/haystack_chunks.parquet")
    pl.DataFrame([asdict(r) for r in needles]).write_parquet("tmp/haystack_chunk_needles.parquet")

    averages: list[Mapping[str, object]] = []
    for i, t in enumerate(TARGETS):
        print(f"\n target {i + 1}/{len(TARGETS)}  {t}")

        df = _run_target(t, chunks, needles, k_max=10)
        df.write_parquet(f"tmp/haystack_res{i}.cparquet")

        mean_metrics = df.drop("chunk_id").mean().row(0, named=True)
        averages.append({"target": t.fingerprint} | mean_metrics)

    df_avg = pl.DataFrame(averages)
    print(df_avg)


if __name__ == "__main__":
    _experiment("verbatim")
    _experiment("query")
    # _experiment("paraphrase")
