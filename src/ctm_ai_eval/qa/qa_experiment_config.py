from dataclasses import dataclass

from ctm_ai_eval.common_config import LlmConfig


@dataclass
class QaExperimentConfig:
    chat_models: list[LlmConfig]
