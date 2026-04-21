from pathlib import Path

import polars as pl

from ctm_ai_eval.util import infer_model_size


def load_all_md(root: Path):
    """Load all .md/.qmd recursively."""
    docs = sorted(root.rglob("*.md")) + sorted(root.rglob("*.qmd"))
    print(f"{len(docs)=}")

    return [p.read_text() for p in docs]


def load_traces_df(traces_file: Path | str) -> pl.DataFrame:
    return (
        pl.read_ndjson(traces_file)
        .unnest("target_cfg")
        .with_columns(
            model_size=pl.col("model").map_elements(
                infer_model_size,
                return_dtype=pl.Int32,
            ),
            ans_length=pl.col("answer").str.len_chars(),
            # string type fingerprint for model
            model_spec=pl.col("model")
            + "_"
            + pl.col("temperature").cast(pl.String)
            + "_"
            + pl.col("system_prompt_id"),
        )
    )
