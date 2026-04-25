"""Evaluate each run."""

import sys
from pathlib import Path

from ctm_ai_eval.utils.io_util import append_ndjson, load_list_json_generic, load_ndjson_generic

sys.path.append(".")

from ctm_ai_eval.qa import judges
from ctm_ai_eval.qa.datamodels import EvalCase, EvalTrace, FloatTraceMetric, QaQuestion

judge_sys_prompt = Path("./assets/prompts/judge_qa_sys.txt").read_text()
judge_msg_template = Path("./assets/prompts/judge_qa_msg.txt").read_text()

DATASET_NAME = "general_qa_python"

JUDGES: list[judges.Judge] = [
    judges.IsConcise(),
    judges.HumanRatingJudge(),
    judges.LLMJudge("rnj-1:8b", judge_sys_prompt, judge_msg_template),
]


def qa_compute_metrics() -> None:
    """Compute metrics for each run."""
    # load runs and dataset
    traces_file = Path(f"./tmp/traces/{DATASET_NAME}.ndjson")
    data_file = Path(f"./assets/data/{DATASET_NAME}.json")
    traces = load_ndjson_generic(traces_file, EvalTrace)
    examples_by_id = {e.example_id: e for e in load_list_json_generic(data_file, QaQuestion)}
    print(f"loaded {len(traces)} runs, {len(examples_by_id)} examples")

    # where to store results
    metrics_file = Path(f"./tmp/metrics/{DATASET_NAME}.ndjson")
    metrics_file.parent.mkdir(exist_ok=True)

    # avoid recomputing done results
    done_metrics: set[tuple[str, str, str]] = (
        {m.fingerprint for m in load_ndjson_generic(metrics_file, FloatTraceMetric)}
        if metrics_file.exists()
        else set()
    )

    for judge in JUDGES:
        print(f" --- Judge: {judge.name} ---")
        for i, trace in enumerate(traces):
            fingerprint = (trace.run_id, trace.example_id, judge.name)
            # skip already computed
            if fingerprint in done_metrics:
                print(f"ALREADY DONE {fingerprint}")
                continue

            print(f"{i + 1:3d}/{len(traces)} -> {fingerprint}")
            case = EvalCase(trace, examples_by_id[trace.example_id])

            result = judge.evaluate(case)
            # store result
            append_ndjson(metrics_file, [result])


if __name__ == "__main__":
    qa_compute_metrics()
