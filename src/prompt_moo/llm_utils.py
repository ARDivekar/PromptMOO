"""Utilities for working with LLM worker instances in pipeline components."""

from typing import List

from .types import LLMPool


def get_prompt_suffix(llm_pool: LLMPool) -> str:
    """Read the prompt suffix stored on an LLM worker (empty string if none).

    The suffix is set by ``runner.py`` at worker creation time via
    ``object.__setattr__(llm, "_prompt_suffix", ...)``. For Qwen models
    with reasoning disabled, this contains the ``/no_think`` token.
    Non-Qwen workers (or Qwen with reasoning enabled) have an empty suffix.
    """
    return getattr(llm_pool, "_prompt_suffix", "")


def apply_prompt_suffix(prompts: List[str], llm_pool: LLMPool) -> List[str]:
    """Append the worker's prompt suffix to every prompt string.

    Returns the original list unchanged if the suffix is empty.
    """
    suffix = get_prompt_suffix(llm_pool)
    if len(suffix) == 0:
        return prompts
    return [p + suffix for p in prompts]
