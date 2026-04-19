"""
example from: https://mlflow.org/docs/3.9.0/genai/getting-started/
"""

import mlflow
import mlflow.genai as mlflow_genai
import openai
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

mlflow.set_experiment("PROMPT OPT")

# Register an initial prompt
prompt = mlflow_genai.register_prompt(
    name="math_tutor",
    template="Answer this math question: {{question}}. Provide a concise explanation.",
)


MODEL = "gemma3:270m"
MODEL_EVAL = "gemma3:1b"


# Define prediction function that includes prompt.format() call
def predict_fn(question: str) -> str:
    prompt = mlflow_genai.load_prompt("prompts:/math_tutor@latest")
    completion = openai.OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="OLLAMA",
    ).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": str(prompt.format(question=question))}],
    )
    resp = completion.choices[0].message.content
    assert resp is not None
    return resp


# Prepare training data with inputs and expectations
train_data = [
    {
        "inputs": {"question": "What is 15 + 27?"},
        "expectations": {"expected_response": "42"},
    },
    {
        "inputs": {"question": "Calculate 8 × 9"},
        "expectations": {"expected_response": "72"},
    },
    {
        "inputs": {"question": "What is 100 - 37?"},
        "expectations": {"expected_response": "63"},
    },
    # ... more examples
]

# Automatically optimize the prompt using MLflow + GEPA
result = mlflow_genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=f"openai:/{MODEL_EVAL}",
        max_metric_calls=8,  # speed/performance tradeoff
        display_progress_bar=True,
    ),
    scorers=[Correctness(model=f"openai:/{MODEL_EVAL}")],
)

# The optimized prompt is automatically registered as a new version
optimized_prompt = result.optimized_prompts[0]
print(f"Optimized prompt registered as version {optimized_prompt.version}")
print(f"Template: {optimized_prompt.template}")
print(f"Score: {result.final_eval_score}")
