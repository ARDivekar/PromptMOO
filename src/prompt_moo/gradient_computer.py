"""
Gradient Computer: Transforms feedback into text gradients.

This is Step 3 of the optimization pipeline.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, ClassVar, Dict, List, Optional, Union

from morphic import Registry, Typed, validate
from morphic.typed import format_exception_msg

from .config import promptmoo_config
from .data_structures import (
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from .llm_utils import apply_prompt_suffix
from .prompt_template import PromptTemplate

# Export validator for use when creating LLM pools
__all__ = [
    "GradientComputer",
    "StandardGradientComputer",
    "OPROGradientComputer",
    "GPOGradientComputer",
    "TextGradGradientComputer",
    "validate_gradient_response",
]


def validate_gradient_response(result: str, **context) -> bool:
    """Validator for gradient computer responses - ensures non-empty text.

    Args:
        result: Raw LLM response text
        **context: Additional context (unused)

    Returns:
        True if response contains non-empty text, False otherwise
    """
    # Gradient responses should be non-empty text (not necessarily JSON)
    return isinstance(result, str) and len(result.strip()) > 0


class GradientComputer(Typed, Registry, ABC):
    """Transforms feedback into text gradients.

    This is a transformer component that generates improvement suggestions from feedback.
    """

    _allow_subclass_override: ClassVar[bool] = True

    @abstractmethod
    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        gradient_batch_size: int,
        verbosity: int = 1,
        *,
        use_textual_feedback: bool = False,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        gradient_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Task, List[TextGradient]]:
        """Compute text gradients from feedbacks.

        Args:
            feedbacks: Dict of feedbacks from loss computer
            prompt_template: Current prompt template
            tasks: List of tasks
            llm_pool: LLM pool for gradient generation
            gradient_batch_size: Batch size for grouping feedbacks
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            use_textual_feedback: Whether to use textual feedback for gradient generation
            predictions: List of prediction results (used by TextGrad, PE2)
            ground_truths: List of ground truth samples (used by TextGrad, PE2)
            input_col_labels: Mapping of column names to display labels (used by TextGrad, PE2)
            gradient_task_strategy: Multi-task gradient strategy (used by TextGrad)
            **kwargs: Algorithm-specific context (e.g. PE2 conversation state)

        Returns:
            Dict with:
            - Keys: task_name or combined task tuple
            - Values: List of TextGradient objects
        """
        pass

    def _batch_feedbacks(
        self,
        *,
        feedbacks: List[Union[NumericFeedback, TextualFeedback]],
        batch_size: int,
    ) -> List[List[Union[NumericFeedback, TextualFeedback]]]:
        """Batch feedbacks into groups.

        Args:
            feedbacks: List of feedback objects
            batch_size: Size of each batch

        Returns:
            List of feedback batches
        """
        if batch_size <= 0:
            return [feedbacks]

        batches: List[List[Union[NumericFeedback, TextualFeedback]]] = []
        for i in range(0, len(feedbacks), batch_size):
            batches.append(feedbacks[i : i + batch_size])
        return batches


class StandardGradientComputer(GradientComputer):
    """Standard: Use LLM to generate improvement suggestions."""

    aliases = ["standard", "default"]

    @validate
    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        gradient_batch_size: int,
        verbosity: int = 1,
        *,
        use_textual_feedback: bool = False,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        gradient_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Task, List[TextGradient]]:
        """Compute gradients using LLM for improvement suggestions.

        Args:
            feedbacks: Feedbacks from loss computer
            prompt_template: Current prompt template
            tasks: List of tasks
            llm_pool: LLM pool
            gradient_batch_size: Gradient batch size
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            use_textual_feedback: Whether to use textual feedback (unused)
            predictions: Prediction results (unused)
            ground_truths: Ground truth samples (unused)
            input_col_labels: Column label mapping (unused)
            gradient_task_strategy: Multi-task strategy (unused)
            **kwargs: Absorbed for interface compatibility

        Returns:
            Dict mapping task keys to text gradients
        """
        result: Dict[Task, List[TextGradient]] = {}

        for task, feedback_list in feedbacks.items():
            if len(feedback_list) == 0:
                continue

            # Batch feedbacks for gradient computation
            feedback_batches: List[List[Union[NumericFeedback, TextualFeedback]]] = (
                self._batch_feedbacks(
                    feedbacks=feedback_list,
                    batch_size=gradient_batch_size,
                )
            )

            # Build all gradient prompts first
            prompts: List[str] = []
            for fb_batch in feedback_batches:
                grad_prompt: str = self._build_gradient_prompt(
                    feedbacks=fb_batch,
                    task=task,
                    prompt_template=prompt_template,
                    tasks=tasks,
                )
                prompts.append(grad_prompt)

            # Call LLM with all prompts in a single batch
            gradients: List[TextGradient] = []
            if len(prompts) > 0:
                try:
                    prompts = apply_prompt_suffix(prompts, llm_pool)
                    responses: List[str] = llm_pool.call_llm_batch(
                        prompts=prompts, verbosity=verbosity
                    ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)

                    for fb_batch, prompt, response in zip(
                        feedback_batches, prompts, responses
                    ):
                        feedback_ids: List[str] = []
                        for fb in fb_batch:
                            feedback_ids.extend(fb.aggregated_from_samples)

                        gradient: TextGradient = TextGradient(
                            task_name=task.task_name,
                            gradient_text=response,
                            based_on_feedbacks=feedback_ids,
                            gradient_prompt=prompt,
                        )
                        gradients.append(gradient)
                except (RuntimeError, TimeoutError, ValueError) as e:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: gradient LLM call "
                        f"failed for task {task.task_name}:\n{format_exception_msg(e)}"
                    ) from e

            result[task] = gradients

        return result

    def _build_gradient_prompt(
        self,
        *,
        feedbacks: List[Union[NumericFeedback, TextualFeedback]],
        task: Task,
        prompt_template: PromptTemplate,
        tasks: List[Task],
    ) -> str:
        """Build prompt for gradient computation.

        Groups feedback by sample batch so the gradient LLM can see which
        metrics and critiques correspond to which examples.

        Args:
            feedbacks: Batch of feedbacks
            task: Task object
            prompt_template: Current prompt template
            tasks: List of all tasks

        Returns:
            Prompt string for gradient generation
        """
        groups: OrderedDict[frozenset, Dict[str, list]] = OrderedDict()
        for fb in feedbacks:
            key: frozenset = frozenset(fb.aggregated_from_samples)
            if key not in groups:
                groups[key] = {"samples": list(key), "items": []}
            if isinstance(fb, NumericFeedback):
                groups[key]["items"].append(
                    f"  - {fb.metric_name}: {fb.value:.4f} ({fb.optimization_direction})"
                )
            elif isinstance(fb, TextualFeedback):
                groups[key]["items"].append(f"  - Textual: {fb.feedback_text}")

        feedback_sections: List[str] = []
        for i, (key, group) in enumerate(groups.items(), 1):
            header: str = f"Batch {i} (samples: {', '.join(group['samples'])}):"
            feedback_sections.append(header + "\n" + "\n".join(group["items"]))

        prompt: str = f"""You are given feedback on the performance of a task: {task.task_name}

Current prompt template:
{prompt_template.render_instructions()}

Feedback on performance (grouped by sample batch):
{chr(10).join(feedback_sections) if len(feedback_sections) > 0 else "None"}

Based on this feedback, write the specific sentences or phrases that should be added to, removed from, or changed in the current task instruction.
Do not give abstract advice (e.g. "add more detail"). Instead, write the concrete instruction text that would fix the identified issues.
Provide 2-3 specific, ready-to-use instruction edits:
"""
        return prompt
