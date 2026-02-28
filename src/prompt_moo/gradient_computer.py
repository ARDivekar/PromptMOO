"""
Gradient Computer: Transforms feedback into text gradients.

This is Step 3 of the optimization pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Union

from concurry import BaseFuture
from morphic import Registry, Typed

from .data_structures import (
    CombinedFeedback,
    NumericFeedback,
    Task,
    TextGradient,
    TextualFeedback,
)
from .llm_workers import LLM, BATCH_INVOCATION_TIMEOUT
from .prompt_template_utils import PromptTemplate

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
        llm_pool: Optional[LLM],
        gradient_batch_size: int,
        verbosity: int = 1,
        **kwargs: Dict,
    ) -> Dict[Task, List[TextGradient]]:
        """Compute text gradients from feedbacks.

        Args:
            feedbacks: Dict of feedbacks from loss computer
            prompt_template: Current prompt template
            tasks: List of tasks
            llm_pool: LLM pool for gradient generation
            gradient_batch_size: Batch size for grouping feedbacks
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Algorithm-specific context

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

        batches = []
        for i in range(0, len(feedbacks), batch_size):
            batches.append(feedbacks[i : i + batch_size])
        return batches


class StandardGradientComputer(GradientComputer):
    """Standard: Use LLM to generate improvement suggestions."""

    aliases = ["standard", "default"]

    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,
        gradient_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[TextGradient]]:
        """Compute gradients using LLM for improvement suggestions.

        Args:
            feedbacks: Feedbacks from loss computer
            prompt_template: Current prompt template
            tasks: List of tasks
            llm_pool: LLM pool
            gradient_batch_size: Gradient batch size
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Unused

        Returns:
            Dict mapping task keys to text gradients
        """
        result = {}

        for task, feedback_list in feedbacks.items():
            if len(feedback_list) == 0:
                continue

            # Batch feedbacks for gradient computation
            feedback_batches = self._batch_feedbacks(
                feedbacks=feedback_list,
                batch_size=gradient_batch_size,
            )

            # Build all gradient prompts first
            prompts = []
            for fb_batch in feedback_batches:
                grad_prompt = self._build_gradient_prompt(
                    feedbacks=fb_batch,
                    task=task,
                    prompt_template=prompt_template,
                    tasks=tasks,
                )
                prompts.append(grad_prompt)

            # Call LLM with all prompts in a single batch
            gradients = []
            if len(prompts) > 0:
                try:
                    responses = llm_pool.call_llm_batch(
                        prompts=prompts, verbosity=verbosity
                    ).result(timeout=BATCH_INVOCATION_TIMEOUT)

                    for fb_batch, response in zip(feedback_batches, responses):
                        # Create TextGradient
                        feedback_ids = []
                        for fb in fb_batch:
                            feedback_ids.extend(fb.aggregated_from_samples)

                        gradient = TextGradient(
                            task_name=task.task_name,
                            gradient_text=response,
                            based_on_feedbacks=feedback_ids,
                            gradient_prompt=None,  # Not storing prompt for brevity
                        )
                        gradients.append(gradient)
                except Exception:
                    # Skip all batches on error
                    pass

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

        Args:
            feedbacks: Batch of feedbacks
            task: Task object
            prompt_template: Current prompt template
            tasks: List of all tasks

        Returns:
            Prompt string for gradient generation
        """
        prompt = f"""You are given feedback on the performance of a task: {task.task_name}

Current prompt template instructions:
{prompt_template.to_str()}

Feedback on performance:
"""

        for fb in feedbacks:
            if isinstance(fb, NumericFeedback):
                prompt += f"- {fb.metric_name}: {fb.value:.4f} ({fb.optimization_direction})\n"
            elif isinstance(fb, TextualFeedback):
                prompt += f"- {fb.feedback_text}\n"

        prompt += """
Based on this feedback, suggest how to improve the instructions for this task.
Provide 2-3 specific suggestions for improvement:
"""
        return prompt


class OPROGradientComputer(GradientComputer):
    """OPRO: No gradients, just pass through numeric scores as summary."""

    aliases = ["opro"]

    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,
        gradient_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[TextGradient]]:
        """Convert numeric feedbacks to summary gradients (no LLM calls).

        OPRO doesn't use text gradients in the traditional sense - it uses
        the trajectory of prompt-score pairs. This just creates a summary.

        Args:
            feedbacks: Feedbacks from loss computer
            prompt_template: Current prompt (unused)
            tasks: List of tasks (unused)
            llm_pool: LLM pool (unused)
            gradient_batch_size: Batch size (unused - OPRO aggregates all feedbacks)
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Unused

        Returns:
            Dict mapping task keys to summary gradients
        """
        result = {}

        for task, feedback_list in feedbacks.items():
            # Extract numeric scores from all feedbacks
            scores = []
            for fb in feedback_list:
                if isinstance(fb, NumericFeedback):
                    scores.append(fb.value)

            if len(scores) > 0:
                avg_score = sum(scores) / len(scores)

                gradient = TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Average score: {avg_score:.4f}",
                    based_on_feedbacks=[],
                    gradient_prompt=None,
                )
                result[task] = [gradient]
            else:
                result[task] = []

        return result


class GPOGradientComputer(GradientComputer):
    """GPO GradientComputer: Generate textual gradients for candidate prompts using feedbacks."""

    aliases = ["gpo"]

    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,
        gradient_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[TextGradient]]:
        """Compute GPO-style gradients from feedbacks.

        Args:
            feedbacks: Feedbacks from loss computer
            prompt_template: Current prompt
            tasks: Task definitions
            llm_pool: LLM pool to generate textual gradients
            gradient_batch_size: Number of feedbacks per gradient computation
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Algorithm-specific context (e.g., trajectory)

        Returns:
            Dict mapping task keys to lists of TextGradient
        """
        result = {}

        for task, feedback_list in feedbacks.items():
            if len(feedback_list) == 0:
                result[task] = []
                continue

            # Combine numeric and textual feedbacks with the same samples
            combined_feedbacks = self._combine_feedbacks_by_samples(
                feedbacks=feedback_list, task=task
            )

            # Batch combined feedbacks for gradient computation
            feedback_batches = self._batch_combined_feedbacks(
                feedbacks=combined_feedbacks, batch_size=gradient_batch_size
            )

            # Build all gradient prompts first
            prompts = []
            for fb_batch in feedback_batches:
                grad_prompt = self._build_gradient_prompt(
                    feedbacks=fb_batch,
                    task=task,
                    prompt_template=prompt_template,
                    tasks=tasks,
                )
                prompts.append(grad_prompt)

            # Call LLM with all prompts in a single batch
            gradients = []
            if len(prompts) > 0:
                try:
                    responses = llm_pool.call_llm_batch(
                        prompts=prompts, verbosity=verbosity
                    ).result(timeout=BATCH_INVOCATION_TIMEOUT)

                    for fb_batch, response in zip(feedback_batches, responses):
                        # Aggregate feedback IDs from all combined feedbacks
                        feedback_ids = []
                        for combined_fb in fb_batch:
                            if len(combined_fb.aggregated_from_samples) > 0:
                                feedback_ids.extend(combined_fb.aggregated_from_samples)

                        # Create and store the gradient
                        gradient = TextGradient(
                            task_name=task.task_name,
                            gradient_text=response,
                            based_on_feedbacks=feedback_ids,
                            gradient_prompt=None,  # Not storing prompt for brevity
                        )
                        gradients.append(gradient)
                except Exception as e:
                    print(
                        f"[WARN] Failed to compute gradient for {task.task_name}: {e}"
                    )

            result[task] = gradients

        return result

    def _combine_feedbacks_by_samples(
        self,
        *,
        feedbacks: List[Union[NumericFeedback, TextualFeedback]],
        task: Task,
    ) -> List[CombinedFeedback]:
        """Combine numeric and textual feedbacks that share the same samples.

        GPO computes both numeric metrics and textual feedback on the same batches
        of samples. This method groups them together so they can be used to generate
        a single coherent gradient.

        Args:
            feedbacks: List of numeric and textual feedbacks
            task: Task object

        Returns:
            List of CombinedFeedback objects, each representing feedbacks for a
            unique set of samples
        """
        # Group feedbacks by their sample sets (using frozenset for hashability)
        sample_groups: Dict[frozenset, Dict[str, List]] = {}

        for fb in feedbacks:
            sample_key = frozenset(fb.aggregated_from_samples)

            if sample_key not in sample_groups:
                sample_groups[sample_key] = {
                    "numeric": [],
                    "textual": [],
                    "samples": list(sample_key),
                }

            if isinstance(fb, NumericFeedback):
                sample_groups[sample_key]["numeric"].append(fb)
            elif isinstance(fb, TextualFeedback):
                sample_groups[sample_key]["textual"].append(fb)

        # Create CombinedFeedback objects
        combined = []
        for sample_key, group in sample_groups.items():
            combined.append(
                CombinedFeedback(
                    task_name=task.task_name,
                    numeric_feedbacks=group["numeric"],
                    textual_feedbacks=group["textual"],
                    aggregated_from_samples=group["samples"],
                )
            )

        return combined

    def _batch_combined_feedbacks(
        self,
        *,
        feedbacks: List[CombinedFeedback],
        batch_size: int,
    ) -> List[List[CombinedFeedback]]:
        """Batch combined feedbacks into groups.

        Args:
            feedbacks: List of CombinedFeedback objects
            batch_size: Size of each batch

        Returns:
            List of CombinedFeedback batches
        """
        if batch_size <= 0:
            return [feedbacks]

        batches = []
        for i in range(0, len(feedbacks), batch_size):
            batches.append(feedbacks[i : i + batch_size])
        return batches

    def _build_gradient_prompt(
        self,
        *,
        feedbacks: List[CombinedFeedback],
        task: Task,
        prompt_template: PromptTemplate,
        tasks: List[Task],
    ) -> str:
        """
        Build a prompt for the LLM to generate text gradients for a task.

        Args:
            feedbacks: Batch of CombinedFeedback objects (numeric + textual)
            task: Task object
            prompt_template: Current prompt template
            tasks: List of all tasks

        Returns:
            A string prompt for gradient generation
        """
        # Extract numeric and textual feedback from combined feedbacks
        numeric_summary = []
        textual_summary = []

        for combined_fb in feedbacks:
            # Add all numeric feedbacks
            for nf in combined_fb.numeric_feedbacks:
                numeric_summary.append(
                    f"- {nf.metric_name}: {nf.value:.4f} ({nf.optimization_direction})"
                )
            # Add all textual feedbacks
            for tf in combined_fb.textual_feedbacks:
                textual_summary.append(f"- {tf.feedback_text}")

        prompt = f"""You are optimizing the task: {task.task_name}

Current prompt instructions:
{prompt_template.to_str()}

Feedback on performance:
Numeric feedback:
{chr(10).join(numeric_summary) if len(numeric_summary) > 0 else "None"}

Textual feedback:
{chr(10).join(textual_summary) if len(textual_summary) > 0 else "None"}

Based on this feedback, suggest how to update or improve the instructions for this task.
Focus on clarity, correctness, and alignment with the intended goal.
Return a concise improvement suggestion in plain text.
"""
        return prompt


class TextGradGradientComputer(GradientComputer):
    """
    TextGrad-specific Gradient Computer:
    - Generates textual gradients from textual feedback only
    - Uses LLM to refine and structure feedback into actionable improvement suggestions
    """

    aliases = ["textgrad"]

    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,
        gradient_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[TextGradient]]:
        """
        Compute TextGrad-style gradients from textual feedbacks.

        Args:
            feedbacks: Textual feedbacks from loss computer
            prompt_template: Current prompt
            tasks: Task definitions
            llm_pool: LLM pool to generate textual gradients
            gradient_batch_size: Number of feedbacks per gradient computation
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Algorithm-specific context

        Returns:
            Dict mapping task keys to lists of TextGradient
        """
        result = {}

        for task, feedback_list in feedbacks.items():
            if len(feedback_list) == 0:
                result[task] = []
                continue

            # Batch feedbacks for gradient computation
            feedback_batches = self._batch_feedbacks(
                feedbacks=feedback_list,
                batch_size=gradient_batch_size,
            )

            # Build all gradient prompts first
            prompts = []
            for fb_batch in feedback_batches:
                grad_prompt = self._build_gradient_prompt(
                    feedbacks=fb_batch,
                    task=task,
                    prompt_template=prompt_template,
                    tasks=tasks,
                )
                prompts.append(grad_prompt)

            # Call LLM with all prompts in a single batch
            gradients = []
            if len(prompts) > 0:
                try:
                    responses = llm_pool.call_llm_batch(
                        prompts=prompts, verbosity=verbosity
                    ).result(timeout=BATCH_INVOCATION_TIMEOUT)

                    for fb_batch, response in zip(feedback_batches, responses):
                        # Aggregate feedback IDs
                        feedback_ids = []
                        for fb in fb_batch:
                            if len(fb.aggregated_from_samples) > 0:
                                feedback_ids.extend(fb.aggregated_from_samples)

                        # Create and store the gradient
                        gradient = TextGradient(
                            task_name=task.task_name,
                            gradient_text=response,
                            based_on_feedbacks=feedback_ids,
                            gradient_prompt=None,  # Not storing prompt for brevity
                        )
                        gradients.append(gradient)
                except Exception as e:
                    print(
                        f"[WARN] Failed to compute gradient for {task.task_name}: {e}"
                    )

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
        """
        Build a prompt for the LLM to generate text gradients for a task.

        For TextGrad, we only use textual feedback (no numeric metrics).

        Args:
            feedbacks: Batch of feedbacks (should be textual only)
            task: Task object
            prompt_template: Current prompt template
            tasks: List of all tasks

        Returns:
            A string prompt for gradient generation
        """
        # Extract only textual feedback (TextGrad doesn't use numeric)
        textual_summary = [
            f"- {fb.feedback_text}"
            for fb in feedbacks
            if isinstance(fb, TextualFeedback)
        ]

        prompt = f"""You are optimizing the task: {task.task_name}

Current prompt instructions:
{prompt_template.to_str()}

Textual feedback on performance:
{chr(10).join(textual_summary) if textual_summary else "No feedback available"}

Based on this textual feedback, suggest how to update or improve the instructions for this task.
Focus on addressing the specific issues mentioned in the feedback.
Provide clear, actionable suggestions for improving clarity, correctness, and effectiveness.
Return a concise improvement suggestion in plain text.
"""
        return prompt
