from pathlib import Path

import polars as pl


def infer_model_size(name: str) -> int:
    """Get a number (M params) from the name."""

    part = name.lower().split(":")[1].split("-")[0]

    if part.endswith("m"):
        return int(part.removesuffix("m"))
    if part.endswith("b"):
        return int(1000 * float(part.removesuffix("b")))

    msg = "unknown suffix"
    raise ValueError(msg)


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
