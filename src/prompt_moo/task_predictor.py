"""
Task Predictor: Transforms dataset samples into predictions using LLM.

This is Step 1 of the optimization pipeline.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, List

from morphic import Registry, Typed
from morphic.typed import format_exception_msg

from .data_structures import Batch, PredictionResult
from .prompt_template_utils import PromptTemplate
from .llm_workers import BATCH_INVOCATION_TIMEOUT

# Export validator for use when creating LLM pools
__all__ = ["TaskPredictor", "StandardTaskPredictor", "validate_task_response"]


def parse_task_response(response: str, **context) -> dict:
    """Parse JSON from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If no valid JSON found or parsing fails
    """
    # Extract JSON block from response
    response = response.strip().replace("{{", "{").replace("}}", "}")
    start = response.find("{")
    end = response.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response:\n{response}")

    json_str = (
        response[start:end]
        .removeprefix('"')
        .removesuffix('"')
        .removeprefix('"')
        .removesuffix('"')
        .removeprefix('"')
        .removesuffix('"')
    )

    try:
        return json.loads(json_str)
    except Exception as e:
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
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start == -1 or end == 0:
            return False
        json_str = result[start:end].replace("{{", "{").replace("}}", "}")
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


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
        llm_pool: Any,
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

    def predict(
        self,
        batch: Batch,
        prompt_template: PromptTemplate,
        llm_pool: Any,
        verbosity: int = 1,
        failure_tolerance: float = 0.05,
        **kwargs,
    ) -> List[PredictionResult]:
        """Generate predictions for batch using standard prompt formatting.

        Args:
            batch: Batch of dataset samples
            prompt_template: Current prompt template
            llm_pool: LLM worker pool
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Unused, for compatibility

        Returns:
            List of PredictionResult objects
        """
        # Build prompts for each sample
        prompts = []
        for sample in batch.samples:
            prompt = prompt_template.to_str() + "\n\n"
            prompt += "## Sample Point\n"
            for col, val in sample.inputs.items():
                prompt += f"{col}: {val}\n"
            prompts.append(prompt)

        # Parallel LLM calls
        responses = llm_pool.call_llm_batch(
            prompts=prompts, 
            verbosity=verbosity,
        ).result(
            timeout=BATCH_INVOCATION_TIMEOUT
        )

        # Parse responses into PredictionResult
        results = []
        num_failed_parsing = 0
        for sample, prompt, response in zip(batch.samples, prompts, responses):
            try:
                outputs = parse_task_response(response)
            except ValueError as e:
                num_failed_parsing += 1
                if verbosity >= 1:
                    print(
                        f"Failed to parse task response for sample {sample.sample_id}:\n{format_exception_msg(e)}"
                    )
                continue
            results.append(
                PredictionResult(
                    sample_id=sample.sample_id,
                    # task_outputs=outputs.get("scores", {})
                    # if isinstance(outputs, dict)
                    # else {},
                    task_outputs={
                        k: v for k, v in outputs.items() if isinstance(v, (int, float))
                    },
                    raw_response=response,
                    prompt=prompt,  # Store the LLM input for observability
                )
            )
        if num_failed_parsing >= len(batch.samples) * failure_tolerance:
            raise ValueError(
                f"Too many task responses failed to parse: {num_failed_parsing} out of {len(batch.samples)}"
            )
        return results
