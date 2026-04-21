from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import polars as pl
import tqdm

from ctm_ai_eval.rag.datamodels import HaystackTarget, Retriever, SpanNeedle
from ctm_ai_eval.rag.metrics import recall_at_k_span, reciprocal_rank_span
from ctm_ai_eval.rich_print import CONS


def _compute_metrics(retr: Retriever, needles: list[SpanNeedle], k_max: int):

    all_res = []
    p = tqdm.tqdm(needles, ncols=0, desc="needles")
    for n in p:
        t0 = perf_counter()
        retrieved = retr(n.query, k_max)
        t_retr = perf_counter() - t0
        r = {
            "doc_id": n.doc_id,
            "start": n.start_char,
            "t_retr": t_retr,
            "recall@1": recall_at_k_span(retrieved, n, k=1),
            "recall@5": recall_at_k_span(retrieved, n, k=5),
            "recall@10": recall_at_k_span(retrieved, n, k=10),
            "rr": reciprocal_rank_span(retrieved, n),
        }

        all_res.append(r)

    result_df = pl.DataFrame(all_res)
    return result_df


def run_haystack_experiment(
    needles: list[SpanNeedle],
    targets: list[HaystackTarget],
    corpus_root: Path,
    *,
    max_needles: int | None,
):
    """Evaluate chunkers and retrievers on finding needles."""
    if max_needles is not None:
        needles = needles[:max_needles]

    averages: list[Mapping[str, object]] = []

    for i, t in enumerate(targets):
        CONS.print(
            f"\nTarget {i + 1}/{len(targets)} {t.fingerprint}. Loading docs",
            justify="center",
            style="bold",
        )
        docs = t.corpus_loader(corpus_root)
        chunks = t.chunker(docs)
        print(f"ingesting {len(chunks)} chunks")
        t.retriever.ingest(chunks)

        result = _compute_metrics(t.retriever, needles, k_max=10)
        print("TODO Saver results")

        mean_metrics = result.drop("doc_id", "start").mean().row(0, named=True)
        averages.append({"target": t.fingerprint} | mean_metrics)

    df_avg = pl.DataFrame(averages)
    print(df_avg)
