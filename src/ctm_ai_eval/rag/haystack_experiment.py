import itertools
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter

import polars as pl
import tqdm

from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.ai_retriever import Embedder, FaissRetriever
from ctm_ai_eval.rag.chunkers.basic_chunking import TokenChunker
from ctm_ai_eval.rag.chunkers.chunk_markdown import MarkdownChunker
from ctm_ai_eval.rag.config import HaystackMetricCfg, load_experiment_config
from ctm_ai_eval.rag.datamodels import Chunker, HaystackTarget, Retriever, SpanNeedle
from ctm_ai_eval.rag.dummy_retrievers import DummyRetriever, SimpleExactRetriever
from ctm_ai_eval.rag.metrics import recall_at_k_span, reciprocal_rank_span, soft_reciprocal_rank
from ctm_ai_eval.rich_print import CONS
from ctm_ai_eval.utils.hashing import stable_hash
from ctm_ai_eval.utils.io_util import load_all_md, load_ndjson_generic

NEEDLE_DIR = Path("./tmp/needles")


def _compute_metrics(
    retr: Retriever,
    needles: list[SpanNeedle],
    k_vals: Sequence[int] = (1, 5, 10),
):

    all_res = []
    p = tqdm.tqdm(needles, ncols=0, desc="needles")
    for n in p:
        t0 = perf_counter()
        retrieved = retr(n.query, max(k_vals))
        t_retr = perf_counter() - t0
        r = {
            "doc_id": n.doc_id,
            "start": n.start_char,
            "t_retr": t_retr,
            "rr": reciprocal_rank_span(retrieved, n),
            "soft_rr": soft_reciprocal_rank(retrieved, n),
        }
        # compute each recall value
        for k in k_vals:
            r[f"recall@{k}"] = recall_at_k_span(retrieved, n, k=k)

        all_res.append(r)

    result_df = pl.DataFrame(all_res)
    return result_df


def _run_haystack(
    needle_sets: Iterable[tuple[str, list[SpanNeedle]]],
    targets: list[HaystackTarget],
    corpus_root: Path,
    metrics: HaystackMetricCfg,
    *,
    result_dir: Path | str,
):
    """Evaluate a complete ingest-retrieve-pipeline.

    NOTE: for interpretable results it is probably good keep 2 of the 3 things constant.
          this can be either by separate experiment configurations,
          or group/filter the result after.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    averages: list[Mapping[str, object]] = []

    avg_file = result_dir / "averages.ndjson"
    details_dir = result_dir / "details"
    details_dir.mkdir(exist_ok=True)

    done: set[str]
    if avg_file.exists():
        prev_results = pl.read_ndjson(avg_file)
        done = set(prev_results["run_id"])
    else:
        prev_results = None
        done = set()

    for needle_name, needles in needle_sets:
        for i, t in enumerate(targets):
            run_id = stable_hash({"needle": needle_name} | t.fingerprint_dict)

            if run_id in done:
                CONS.print(f"{run_id} ({t.fingerprint_tuple}) done, skip.", style="yellow")
                continue
            CONS.print(
                f"\nTarget {i + 1}/{len(targets)} {t.fingerprint_tuple}, {needle_name}",
                style="bold",
            )
            # prepare data
            docs = t.loader(corpus_root)
            chunks = t.chunker(docs)
            print(f"ingesting {len(chunks)} chunks")
            t.retriever.ingest(chunks)

            result = _compute_metrics(t.retriever, needles, metrics.k_vals)
            mean_metrics = result.drop("doc_id", "start").mean().row(0, named=True)

            averages.append(
                {
                    "run_id": run_id,
                    "needle": needle_name,
                }
                | t.fingerprint_dict
                | mean_metrics
            )

            # store results
            result.write_parquet(details_dir / f"{run_id}.parquet")

    if averages:
        df_avg = pl.DataFrame(averages)
        print(df_avg)
        if prev_results is not None:
            df_avg = pl.concat([prev_results, df_avg], how="vertical")

        df_avg.write_ndjson(avg_file)


def _targets_prod(chunkers: Iterable[Chunker], retrievers: Iterable[Retriever]):
    return [
        HaystackTarget(loader=load_all_md, chunker=c, retriever=r)
        for (c, r) in itertools.product(chunkers, retrievers)
    ]


def haystack_chunk_size():
    """Compare chunk sizes.
    - This isn't really informative in terms of rr/recall, since larger chunks are always better.
    """
    cfg = load_experiment_config()
    retrievers = [DummyRetriever(), FaissRetriever(Embedder("nomic-embed-text"))]

    chunkers = [
        TokenChunker(k, 0, text_processing.tokenize_words) for k in [60, 100, 140, 180, 220]
    ]

    needle_files = NEEDLE_DIR.glob("*/*.ndjson")
    if not needle_files:
        print(f"no needles found in {NEEDLE_DIR}, please generate first")
        sys.exit(1)

    _run_haystack(
        ((f.stem, load_ndjson_generic(f, SpanNeedle)) for f in needle_files),
        _targets_prod(chunkers, retrievers),
        cfg.dataset.corpus_path,
        cfg.metrics,
        result_dir="./tmp/results/haystack-chunksize",
    )


def haystack_chunkers():
    """Compare chunk sizes.
    - This isn't really informative in terms of rr/recall, since larger chunks are always(?) better.
    """
    cfg = load_experiment_config()
    retriever = FaissRetriever(Embedder("nomic-embed-text"))

    chunkers = [
        MarkdownChunker(100, 0),
        MarkdownChunker(100, 50),
        MarkdownChunker(200, 0),
        MarkdownChunker(200, 100),
        TokenChunker(20, 0, text_processing.tokenize_words),
        TokenChunker(20, 10, text_processing.tokenize_words),
        TokenChunker(40, 0, text_processing.tokenize_words),
        TokenChunker(40, 20, text_processing.tokenize_words),
    ]

    targets = [HaystackTarget(load_all_md, c, retriever) for c in chunkers]

    needle_files = NEEDLE_DIR.glob("*/*.ndjson")
    if not needle_files:
        print(f"no needles found in {NEEDLE_DIR}, please generate first")
        sys.exit(1)

    _run_haystack(
        ((f.stem, load_ndjson_generic(f, SpanNeedle)) for f in needle_files),
        targets,
        cfg.dataset.corpus_path,
        cfg.metrics,
        result_dir="./tmp/results/haystack-chunkers",
    )


def haystack_retrievers():
    """Compare retrievers.

    - Possibly informative, as long asthe same chunking is used for all.
    """
    cfg = load_experiment_config()

    # include baselines and specified embedding models.
    retrievers = [
        DummyRetriever(),
        SimpleExactRetriever(weight_lcs=0.3),
    ] + [FaissRetriever(Embedder(k)) for k in cfg.targets.embedders]

    # A single chunker
    chunker = TokenChunker(40, 20, text_processing.tokenize_words)

    targets = [HaystackTarget(load_all_md, chunker, r) for r in retrievers]

    needle_files = NEEDLE_DIR.glob("*/*.ndjson")
    if not needle_files:
        print(f"no needles found in {NEEDLE_DIR}, please generate first")
        sys.exit(1)

    _run_haystack(
        ((f.stem, load_ndjson_generic(f, SpanNeedle)) for f in needle_files),
        targets,
        cfg.dataset.corpus_path,
        cfg.metrics,
        result_dir="./tmp/results/haystack_retrievers",
    )
