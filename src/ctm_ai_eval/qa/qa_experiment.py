import datetime
import platform
import sys
from pathlib import Path

import tqdm

from ctm_ai_eval.qa.datamodels import EvalTrace, QaQuestion
from ctm_ai_eval.rich_print import CONS
from ctm_ai_eval.utils import io_util

sys.path.append(".")


from ctm_ai_eval.qa import targets

SERVER_URL = "http://localhost:5000"
DATASET_NAME = "general_qa_python"
SYS_PROMPT_DIR = Path("./assets/prompts/chat_system_prompts")

TARGETS = [
    targets.OllamaChatTarget(
        targets.ChatTargetConfig(
            model="gemma3:1b-it-qat",
            temperature=0.0,
            system_prompt_id="concise",
        ),
    ),
    targets.OllamaChatTarget(
        targets.ChatTargetConfig(
            model="qwen2.5-coder:0.5b",
            temperature=0.0,
            system_prompt_id="concise",
        ),
    ),
]


def run_eval(
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
        r = EvalTrace(
            run_id=run_id,
            dataset_name=dataset_name,
            example_id=ex.example_id,
            server_url=target.server_url,
            route=target.route,
            answer=result.text,
            latency_ms=result.latency_ms,
            target_cfg=target.chat_config.model_dump(),
            local_host=platform.node(),
        )
        io_util.append_ndjson(traces_file, [r])


def qa_trace() -> None:
    """Main function to run the QA trace collection."""

    for targ in TARGETS:
        CONS.print(targ)
        run_eval(DATASET_NAME, targ)


if __name__ == "__main__":
    qa_trace()
