import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DatasetConfig:
    corpus_path: Path
    max_needles: int


@dataclass
class TokenizationConfig:
    type: Literal["words", "sentences", "sentence_windows"]


# @dataclass
# class ChunkingConfig:
#     type: Literal["token"]
#     chunk_size: int
#     overlap: int


@dataclass
class EmbeddingConfig:
    models: list[str]


@dataclass
class ExperimentConfig:
    mode: Literal["verbatim", "paraphrase", "query"]
    top_k: int


@dataclass
class RagExperimentConfig:
    dataset: DatasetConfig


def load_config(path: str = "config.toml") -> RagExperimentConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return RagExperimentConfig(
        dataset=DatasetConfig(
            corpus_path=Path(raw["dataset"]["corpus_path"]).expanduser(),
            max_needles=raw["dataset"]["max_needles"],
        ),
    )
