"""
Task Predictor: Transforms dataset samples into predictions using LLM.

This is Step 1 of the optimization pipeline.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from morphic import Registry, Typed, validate
from morphic.typed import format_exception_msg

from .config import promptmoo_config
from .data_structures import Batch, PredictionResult
from .llm_utils import apply_prompt_suffix
from .prompt_template import PromptTemplate
from .types import strip_smart_quotes

# Export validator for use when creating LLM pools
__all__ = ["TaskPredictor", "StandardTaskPredictor", "validate_task_response"]


@validate
def parse_task_response(response: str, **context) -> dict:
    """Parse JSON from LLM response.

    Handles clean JSON, preamble/postamble text, code fences, and
    escaped ``{{``/``}}`` braces from f-string template echo.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If no valid JSON found or parsing fails
    """
    if response is None:
        raise ValueError("Response from LLM was None.")

    if isinstance(response, Exception):
        raise ValueError(
            f"LLM call resulted in an exception: {format_exception_msg(response)}"
        )

    if not isinstance(response, str):
        response = str(response)

    text: str = response.strip()

    fence_match: Optional[re.Match] = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
    )
    if fence_match is not None:
        inner: str = fence_match.group(1).strip()
        if len(inner) > 0 and "{" in inner:
            text = inner

    start: int = text.find("{")
    end: int = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response:\n{response}")

    json_str: str = strip_smart_quotes(text[start:end])

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fallback: try replacing escaped braces from template echo
    fallback: str = json_str.replace("{{", "{").replace("}}", "}")
    try:
        return json.loads(fallback)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(
            f"Failed to parse JSON: {format_exception_msg(e)}.\nResponse:\n{response}"
        )


def validate_task_response(result: str, **context) -> bool:
    """Validator for task predictor responses - ensures valid JSON.

    Args:
        result: Raw LLM response text
        **context: Additional context (unused)

    Returns:
        True if response contains valid JSON, False otherwise
    """
    if result is None or not isinstance(result, str):
        return False

    try:
        parse_task_response(result)
        return True
    except (ValueError, json.JSONDecodeError):
        return False


def _extract_numeric_outputs(parsed: dict) -> dict:
    """Extract numeric task scores from a parsed JSON dict.

    Handles:
    - int/float values (normal case): kept as-is
    - String integers ("5"): coerced to int
    - String floats ("4.5"): coerced to float
    - Booleans: excluded (bool is subclass of int in Python)
    - Strings, nulls, dicts: excluded
    """
    result: Dict[str, Union[int, float]] = {}
    for k, v in parsed.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            result[k] = v
            continue
        if isinstance(v, str):
            v_stripped: str = v.strip()
            try:
                result[k] = int(v_stripped)
                continue
            except ValueError:
                pass
            try:
                result[k] = float(v_stripped)
                continue
            except ValueError:
                pass
    return result


class TaskPredictor(Typed, Registry, ABC):
    """Transforms dataset samples into predictions.

    This is a transformer component that takes a batch of samples and produces predictions.
    """

    _allow_subclass_override = True

    @abstractmethod
    def predict(
        self,
        batch: Batch,
        prompt_template: PromptTemplate,
        llm_pool: Any,  # LLMPool protocol; see types.py
        verbosity: int = 1,
        **kwargs,
    ) -> List[PredictionResult]:
        """Generate predictions for batch.

        Args:
            batch: Batch of dataset samples to predict on
            prompt_template: Current prompt template to use
            llm_pool: LLM worker pool for making predictions
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Algorithm-specific context (e.g., trajectory)

        Returns:
            List of PredictionResult objects, one per sample
        """
        pass


class StandardTaskPredictor(TaskPredictor):
    """Standard implementation: format prompts and call LLM in parallel."""

    aliases = ["standard", "default"]

    @validate
    def predict(
        self,
        batch: Batch,
        prompt_template: PromptTemplate,
        llm_pool: Any,  # LLMPool protocol; see types.py
        verbosity: int = 1,
        failure_tolerance: float = 0.05,
        **kwargs,
    ) -> List[PredictionResult]:
        """Generate predictions for batch using standard prompt formatting.

        Args:
            batch: Batch of dataset samples
            prompt_template: Current prompt template (carries ``input_col_labels``
                for sample rendering)
            llm_pool: LLM worker pool
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)

        Returns:
            List of PredictionResult objects
        """
        prompts: List[str] = []
        for sample in batch.samples:
            prompts.append(prompt_template.render_task_prompt(sample=sample))

        prompts = apply_prompt_suffix(prompts, llm_pool)
        responses = llm_pool.call_llm_batch(
            prompts=prompts,
            verbosity=verbosity,
        ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)

        # Parse responses into PredictionResult
        results: List[PredictionResult] = []
        num_failed_parsing: int = 0
        for sample, prompt, response in zip(batch.samples, prompts, responses):
            try:
                outputs: dict = parse_task_response(response)
            except ValueError as e:
                num_failed_parsing += 1
                error_message: str = format_exception_msg(e)
                if verbosity >= 1:
                    print(
                        f"Failed to parse task response for sample {sample.sample_id}:\n{error_message}"
                    )
                results.append(
                    PredictionResult(
                        sample_id=sample.sample_id,
                        prompt=prompt,
                        task_outputs={},
                        raw_response=response
                        if isinstance(response, str)
                        else str(response),
                        parser_error=error_message,
                    )
                )
                continue
            results.append(
                PredictionResult(
                    sample_id=sample.sample_id,
                    prompt=prompt,
                    task_outputs=_extract_numeric_outputs(outputs),
                    raw_response=response,
                )
            )
        if num_failed_parsing >= len(batch.samples) * failure_tolerance:
            raise ValueError(
                f"Too many task responses failed to parse: {num_failed_parsing} out of {len(batch.samples)}"
            )
        return results
