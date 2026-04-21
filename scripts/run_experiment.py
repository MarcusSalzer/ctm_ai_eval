import itertools
from pathlib import Path

import polars as pl

from ctm_ai_eval.io_util import load_all_md
from ctm_ai_eval.rag import text_processing
from ctm_ai_eval.rag.ai_retriever import Embedder, FaissRetriever
from ctm_ai_eval.rag.chunking import TokenChunker
from ctm_ai_eval.rag.config import load_experiment_config
from ctm_ai_eval.rag.datamodels import HaystackTarget, SpanNeedle
from ctm_ai_eval.rag.dummy_retrievers import DummyRetriever, SimpleExactRetriever
from ctm_ai_eval.rag.haystack_experiment import run_haystack_experiment

NEEDLE_DIR = Path("./tmp/needles/")


def _main():
    cfg = load_experiment_config()

    retrievers = [
        DummyRetriever(),
        SimpleExactRetriever(weight_lcs=0.3),
        SimpleExactRetriever(weight_lcs=0.0),
    ] + [FaissRetriever(Embedder(k)) for k in cfg.embedders]

    chunkers = [
        TokenChunker(100, text_processing.tokenize_words),
        TokenChunker(10, text_processing.tokenize_sentences),
    ]

    targets = [
        HaystackTarget(load_all_md, c, r) for (c, r) in itertools.product(chunkers, retrievers)
    ]

    for needle_set in NEEDLE_DIR.glob("*/*.parquet"):
        needles = [SpanNeedle(**r) for r in pl.read_parquet(needle_set).iter_rows(named=True)]

        run_haystack_experiment(needles, targets, cfg.dataset.corpus_path, max_needles=5)


if __name__ == "__main__":
    _main()
