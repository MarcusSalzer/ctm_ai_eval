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
