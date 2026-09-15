"""Configuration objects.

RULE 3 from the sprint brief: Ollama is an inference/runtime layer, not a training
framework. Everything here therefore describes *inference*, never training.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# Sprint-01 target model (production). Documented exact model id from
# https://huggingface.co/Qwen/Qwen3-14B -> Ollama tag ``qwen3:14b``.
TARGET_MODEL = "qwen3:14b"

# Model used for CI / laptop smoke runs, same family & chat template, small weights.
SMOKE_MODEL = "qwen3:0.6b"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class OllamaConfig:
    """Connection + *frozen* inference mode.

    The inference mode is pinned on purpose: the sprint brief requires that a
    benchmark compares architectures, not accidentally different decoding modes.
    """

    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.environ.get("SMARTGATE_MODEL", SMOKE_MODEL))
    # Qwen3 exposes an explicit thinking / non-thinking switch in its chat template.
    # Sprint 01 fixes non-thinking mode so latency and outputs stay comparable.
    think: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    num_ctx: int = field(default_factory=lambda: _env_int("SMARTGATE_NUM_CTX", 4096))
    num_predict: int = field(default_factory=lambda: _env_int("SMARTGATE_NUM_PREDICT", 256))
    timeout: float = field(default_factory=lambda: _env_float("SMARTGATE_TIMEOUT", 120.0))
    retries: int = 2

    @property
    def api_base(self) -> str:
        return self.host.rstrip("/") + "/api"

    def options(self) -> dict:
        """Deterministic generation parameters passed to the Ollama API."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["options"] = self.options()
        return d


@dataclass(frozen=True)
class DetectorConfig:
    """Hyper-parameters of the statistical stage.

    RULE 2: these are *experimental points*, not copied defaults — they are swept by
    ``smartgate sweep`` and the chosen values are recorded in Report.md.
    """

    ewma_alpha: float = 0.3
    ewma_k: float = 3.0
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    warmup: int = 14
    min_sigma: float = 1e-6


@dataclass(frozen=True)
class PipelineConfig:
    detector: str = "ewma"
    use_llm_gate: bool = True
    top_k: int = 10
    # Days we are allowed to observe after the first alarm before committing to a verdict.
    decision_horizon: int = 7
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    detectors: DetectorConfig = field(default_factory=DetectorConfig)
