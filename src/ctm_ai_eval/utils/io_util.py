import json
import re
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from ctm_ai_eval.util import infer_model_size


def load_all_md(root: Path):
    """Load all .md/.qmd recursively. Sorted by extension and filename"""
    docs = sorted(root.rglob("*.md")) + sorted(root.rglob("*.qmd"))
    print(f"{len(docs)=}")

    return [p.read_text() for p in docs]


def load_ndjson_generic[T](path: Path, typ: type[T], *, max_count: int | None = None) -> list[T]:
    """Load NDJSON of records."""
    lines = path.read_text().splitlines()
    if max_count is not None:
        lines = lines[:max_count]
    return [typ(**json.loads(s)) for s in lines]


def load_list_json_generic[T](path: Path, typ: type[T]) -> list[T]:
    """Load list json of records."""

    raw_list = json.loads(path.read_text())
    return [typ(**r) for r in raw_list]


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


def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]

    # trash before json
    text = re.sub(r"^[\s\S]+{", "{", text)
    return text.strip()


def save_records_list_json(path: Path, records: Sequence[BaseModel]) -> None:
    _ = path.write_text(json.dumps([r.model_dump(mode="json") for r in records]))


def append_ndjson(file: Path, records: Sequence[BaseModel]) -> None:
    with file.open("a") as fout:
        for r in records:
            fout.write(r.model_dump_json() + "\n")
