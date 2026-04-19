from dataclasses import dataclass


@dataclass
class LlmConfig:
    model: str
    temperature: float
    sys_prompt_name: str
