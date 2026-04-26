"""Main entry point for running some experiment."""

import sys

from ctm_ai_eval.qa.eval_runs import qa_compute_metrics
from ctm_ai_eval.qa.qa_experiment import qa_trace
from ctm_ai_eval.rag.haystack_experiment import (
    haystack_chunk_size,
    haystack_chunkers,
    haystack_retrievers,
)
from ctm_ai_eval.rich_print import CONS

EXPERIMENTS = {
    # rag
    "retrievers": haystack_retrievers,
    "chunkers": haystack_chunkers,
    "chunksize": haystack_chunk_size,
    # qa
    "qa_trace": qa_trace,
    "qa_metrics": qa_compute_metrics,
}


def _main():
    if len(sys.argv) < 2:
        print(f"please specify an experiment: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    key = sys.argv[1]
    CONS.print(f"Running experiment: {key}", style="bold black on white", justify="center")

    EXPERIMENTS[key]()


if __name__ == "__main__":
    _main()
