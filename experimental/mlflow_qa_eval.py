import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.genai as mlflow_genai
import pandas as pd
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from openai import OpenAI

sys.path.append(".")
from ai_eval import util
from material_prep import io_util

os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"

# Your agent implementation
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="IGNORED")

DATASET_NAME = "general_qa_python"

MODEL = "gemma3:1b"
# MODEL = "qwen2.5-coder:1.5b"

# Ideally a big model?
MODEL_EVAL = "gemma3:1b"


qa_questions = io_util.load_qas_list_json(Path(f"ai_eval/data/{DATASET_NAME}.json"))

data_df = pd.DataFrame(
    [
        {
            "inputs": {"question": q.to_question_string()},
            "expectations": {"expected_response": q.answer},
        }
        for q in qa_questions
    ],
)
print(data_df.head())

_ = mlflow.set_experment("Python QA eval")
mlflow.log_param("model", MODEL)
mlflow.log_param("model_size_M", util.infer_model_size(MODEL))
mlflow.log_param("eval-model", MODEL_EVAL)

prompt = mlflow_genai.register_prompt(
    "qa-simple",
    [
        {
            "role": "system",
            "content": "Answer questions concisely and correctly.",
        },
        {"role": "user", "content": "{{question}}"},
    ],
)


def my_agent(question: str):
    msgs = prompt.format(question=question)
    assert isinstance(msgs, list)

    start = time.perf_counter()

    response = client.chat.completions.create(model=MODEL, messages=cast("Any", msgs))
    resp = response.choices[0].message.content
    latency = time.perf_counter() - start

    assert resp is not None
    return resp, latency


# Wrapper function for evaluation
def qa_predict_fn(question: str) -> str:
    resp, latency = my_agent(question)

    return resp


# Scorers
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 20


scorers = [
    Correctness(model=f"openai:/{MODEL_EVAL}"),
    Guidelines(
        name="is_english",
        guidelines="The answer must be in English",
        model=f"openai:/{MODEL_EVAL}",
    ),
    is_concise,
]

# Run evaluation
if __name__ == "__main__":
    print("DUMMY MESSAGE")
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "warmup"}]
    )
    print("MODEL OK")

    # evaluate
    results = mlflow_genai.evaluate(
        data=data_df,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )
