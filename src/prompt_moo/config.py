"""
Global configuration for PromptMOO.

Provides a single source of truth for all tunable defaults (timeouts, temperatures,
max_tokens, rate limits, retry config, etc.).

Pattern follows SlowBurn/Concurry: a mutable singleton ``promptmoo_config``
whose ``defaults`` field can be mutated at runtime or scoped via ``temp_config()``.
"""

from contextlib import contextmanager
from typing import Any, Generator

from morphic import MutableTyped
from pydantic import ConfigDict, Field, confloat, conint


class PromptMOODefaults(MutableTyped):
    """All tunable defaults for PromptMOO, in one place.

    Call sites read from ``promptmoo_config.defaults.<field>`` at call time.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Batch-level timeout for .result() calls in pipeline components
    batch_invocation_timeout: confloat(gt=0.0) = 3600.0

    # Per-LLM-role timeouts (used in create_*_llm factory functions)
    task_llm_timeout: confloat(gt=0.0) = 60.0
    optimizer_llm_timeout: confloat(gt=0.0) = 600.0
    gradient_llm_timeout: confloat(gt=0.0) = 600.0
    loss_llm_timeout: confloat(gt=0.0) = 60.0

    # Per-LLM-role max_tokens
    task_llm_max_tokens: conint(ge=1) = 256
    optimizer_llm_max_tokens: conint(ge=1) = 4096
    gradient_llm_max_tokens: conint(ge=1) = 2048
    loss_llm_max_tokens: conint(ge=1) = 256

    # Temperature defaults per role
    task_llm_temperature: confloat(ge=0.0, le=2.0) = 0.9
    optimizer_llm_temperature: confloat(ge=0.0, le=2.0) = 0.9
    gradient_llm_temperature: confloat(ge=0.0, le=2.0) = 0.9
    loss_llm_temperature: confloat(ge=0.0, le=2.0) = 0.9

    # Shared rate limits (across all 4 LLM workers per experiment)
    max_parallel_calls: conint(ge=1) = 20
    max_rpm: conint(ge=1) = 1_000
    max_input_tpm: conint(ge=1) = 10_000_000
    max_output_tpm: conint(ge=1) = 1_000_000

    # Retry
    num_retries: conint(ge=0) = 10
    retry_algorithm: str = "Fibonacci"
    retry_wait: confloat(ge=0.0) = 2.0
    retry_jitter: confloat(ge=0.0) = 0.7

    # Training loop
    substep_delay: confloat(ge=0.0) = 1.5

    # Verbosity
    verbosity: conint(ge=0) = 1


class PromptMOOConfig(MutableTyped):
    """Top-level config object. The global singleton is ``promptmoo_config``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    defaults: PromptMOODefaults = Field(default_factory=PromptMOODefaults)

    def reset_to_defaults(self) -> None:
        """Restore all defaults to their original values."""
        object.__setattr__(self, "defaults", PromptMOODefaults())


promptmoo_config = PromptMOOConfig()


@contextmanager
def temp_config(**overrides: Any) -> Generator[PromptMOOConfig, None, None]:
    """Temporarily override config defaults within a ``with`` block.

    All keyword arguments must be valid field names on ``PromptMOODefaults``.
    On exit (normal or exception), the original values are restored.

    Usage::

        from prompt_moo.config import temp_config

        with temp_config(batch_invocation_timeout=300, verbosity=0):
            results = algo.train(dataset, initial_prompt)

    Raises:
        ValueError: If any key is not a recognized PromptMOODefaults field.
    """
    defaults = promptmoo_config.defaults
    valid_fields = set(PromptMOODefaults.model_fields.keys())
    unknown = set(overrides.keys()) - valid_fields
    if len(unknown) > 0:
        raise ValueError(
            f"Unknown config key(s): {sorted(unknown)}. Valid: {sorted(valid_fields)}"
        )
    saved = {k: getattr(defaults, k) for k in overrides}
    try:
        for key, value in overrides.items():
            setattr(defaults, key, value)
        yield promptmoo_config
    finally:
        for key, value in saved.items():
            setattr(defaults, key, value)
