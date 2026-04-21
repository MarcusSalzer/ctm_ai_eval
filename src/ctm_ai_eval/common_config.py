from dataclasses import dataclass


@dataclass
class LlmConfig:
    model: str
    temperature: float
    think: bool
