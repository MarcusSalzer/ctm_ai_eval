import datetime
import platform
import sys
from pathlib import Path

import tqdm

from ctm_ai_eval.qa.datamodels import EvalTrace, QaQuestion
from ctm_ai_eval.rag.ai_retriever import FaissRetriever
from ctm_ai_eval.rag.chunkers.chunk_markdown import MarkdownChunker
from ctm_ai_eval.rag.config import load_experiment_config
from ctm_ai_eval.rag.datamodels import HaystackTarget
from ctm_ai_eval.rich_print import CONS
from ctm_ai_eval.utils import io_util

sys.path.append(".")


from ctm_ai_eval.qa import targets

SERVER_URL = "http://localhost:5000"
DATASET_NAME = "general_qa_python"
SYS_PROMPT_DIR = Path("./assets/prompts/chat_system_prompts")
cfg = load_experiment_config()

TARGETS: list[targets.ApiTarget] = [
    # targets.RagApiTarget(
    #     targets.ChatTargetConfig(
    #         model="gemma3:1b-it-qat",
    #         temperature=0.0,
    #         system_prompt_id="concise",
    #     ),
    #     haystack=HaystackTarget(io_util.load_all_md, MarkdownChunker(200, 100), FaissRetriever()),
    #     docs_dir=cfg.dataset.corpus_path,
    # ),
    # targets.OpenAIChatTarget(
    #     targets.ChatTargetConfig(
    #         model="gemma3:1b-it-qat",
    #         temperature=0.0,
    #         system_prompt_id="concise",
    #     ),
    # ),
    targets.RagApiTarget(
        targets.ChatTargetConfig(
            model="gemma4:e2b",
            temperature=0.0,
            system_prompt_id="concise",
        ),
        haystack=HaystackTarget(io_util.load_all_md, MarkdownChunker(200, 100), FaissRetriever()),
        docs_dir=cfg.dataset.corpus_path,
    ),
    targets.OpenAIChatTarget(
        targets.ChatTargetConfig(
            model="gemma4:e2b",
            temperature=0.0,
            system_prompt_id="concise",
        ),
    ),
]


def trace_one_target(
    dataset_name: str,
    target: targets.ApiTarget,
) -> None:
    # load system prompts
    sys_prompts = {p.stem: p.read_text() for p in SYS_PROMPT_DIR.glob("*.txt")}
    # load dataset
    data_file = Path(f"./assets/data/{dataset_name}.json")
    examples = io_util.load_list_json_generic(data_file, QaQuestion)

    # where to store results
    run_id = datetime.datetime.now(datetime.UTC).isoformat()
    traces_file = Path(f"./tmp/traces/{dataset_name}.ndjson")
    traces_file.parent.mkdir(exist_ok=True, parents=True)

    # send an initial request without measuring,
    # to avoid latency spike when loading model
    print("pinging target...")
    target.ask("hello", None)

    print(f"running {run_id} ({dataset_name=})")
    for ex in tqdm.tqdm(examples):
        result = target.ask(
            ex.to_question_string(),
            sys_prompts[target.chat_config.system_prompt_id],
        )
        rag_config = (
            target.haystack.fingerprint_dict if isinstance(target, targets.RagApiTarget) else None
        )
        r = EvalTrace(
            run_id=run_id,
            dataset_name=dataset_name,
            example_id=ex.example_id,
            server_url=target.server_url,
            route=target.route,
            answer=result.text,
            latency_ms=result.latency_ms,
            target_cfg=target.chat_config.model_dump(),
            rag_cfg=rag_config,
            local_host=platform.node(),
        )
        io_util.append_ndjson(traces_file, [r])


def qa_trace() -> None:
    """Main function to run the QA trace collection."""

    for targ in TARGETS:
        CONS.print(targ)
        if isinstance(targ, targets.RagApiTarget):
            print("RAG-target: ingesting...")
            targ.ensure_ingested()
            print("RAG-target: ingest ok!")
        trace_one_target(DATASET_NAME, targ)


if __name__ == "__main__":
    qa_trace()
