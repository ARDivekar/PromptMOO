"""
Loss Computer: Transforms predictions into feedback (numeric and/or textual).

This is Step 2 of the optimization pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

from morphic import Registry, Typed, validate
from morphic.typed import format_exception_msg

from .config import promptmoo_config
from .data_structures import (
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextualFeedback,
)
from .llm_utils import apply_prompt_suffix
from .metrics import Metric
from .prompt_template import PromptTemplate

# Export validator for use when creating LLM pools
__all__ = [
    "LossComputer",
    "TaskLevelLossComputer",
    "validate_loss_feedback_response",
]


def validate_loss_feedback_response(result: str, **context) -> bool:
    """Validator for loss computer textual feedback - ensures non-empty text.

    Args:
        result: Raw LLM response text
        **context: Additional context (unused)

    Returns:
        True if response contains non-empty text, False otherwise
    """
    # Textual feedback should be non-empty text
    return isinstance(result, str) and len(result.strip()) > 0


class LossComputer(Typed, Registry, ABC):
    """Transforms predictions into feedback (numeric and/or textual).

    This is a transformer component that computes losses/feedback for predictions.
    """

    _allow_subclass_override: ClassVar[bool] = True

    @abstractmethod
    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        llm_pool: Optional[Any],  # LLMPool protocol; see types.py
        loss_batch_size: int,
        verbosity: int = 1,
        *,
        loss_functions: Optional[Dict[str, Dict[str, Any]]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        loss_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Compute feedback for predictions.

        Args:
            predictions: List of prediction results from task predictor
            ground_truths: List of dataset samples with ground truth values
            tasks: List of tasks to compute losses for
            prompt_template: Current prompt template (used for the actual optimized
                instruction text in textual feedback prompts)
            llm_pool: Optional LLM pool for computing textual feedback
            loss_batch_size: Batch size for grouping predictions/samples
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            loss_functions: Dict mapping task_name to loss function config
            input_col_labels: Dict mapping column names to display labels
            loss_task_strategy: Multi-task loss strategy (e.g., "separate_tasks",
                "combine_all_tasks")
            **kwargs: Algorithm-specific context

        Returns:
            Dict with:
            - Keys: task_name (str) or combined tasks (tuple of sorted task names)
            - Values: List of feedback objects (numeric or textual)
        """
        pass

    @staticmethod
    @validate
    def batch_predictions(
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task: Task,
        batch_size: int,
    ) -> List[Tuple[List[PredictionResult], List[DatasetSample]]]:
        """Split predictions and ground truths into batches for a specific task.

        Args:
            predictions: Full list of prediction results.
            ground_truths: Full list of ground truth samples.
            task: Task object (unused currently, reserved for future filtering).
            batch_size: Number of samples per batch.

        Returns:
            List of (prediction_batch, ground_truth_batch) tuples.
        """
        batches: List[Tuple[List[PredictionResult], List[DatasetSample]]] = []
        for i in range(0, len(predictions), batch_size):
            pred_batch = predictions[i : i + batch_size]
            gt_batch = ground_truths[i : i + batch_size]
            batches.append((pred_batch, gt_batch))
        return batches

    def _compute_numeric_loss(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task: Task,
        loss_fn_config: dict,
    ) -> Optional[NumericFeedback]:
        """Compute numeric loss for a batch using the Metric registry.

        Resolves the metric name from ``loss_fn_config["metric"]`` to a
        ``Metric`` subclass via ``Metric.get_subclass()``, extracts
        y_true/y_pred arrays, computes the value, and wraps the result in
        a ``NumericFeedback`` containing the ``Metric`` instance.

        Metric-specific kwargs (e.g. ``num_classes`` for LCE) are extracted
        from ``loss_fn_config`` and forwarded to both ``compute()`` and
        the ``Metric`` constructor.

        Args:
            predictions: Batch of predictions.
            ground_truths: Batch of ground truths.
            task: Task to compute loss for.
            loss_fn_config: Must contain ``"metric"`` key (e.g. ``"accuracy"``).
                May contain additional metric-specific keys (e.g. ``"num_classes"``
                for LCE).

        Returns:
            NumericFeedback object, or None if no valid pairs found.
        """
        if "metric" not in loss_fn_config:
            return None
        metric_name: str = loss_fn_config["metric"]

        try:
            metric_cls: Type[Metric] = Metric.get_subclass(metric_name)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"{self.__class__.__name__}: unknown metric {metric_name!r}. "
                f"Register it as a Metric subclass in metrics.py."
            ) from e

        metric_kwargs: Dict[str, Any] = {
            k: v
            for k, v in loss_fn_config.items()
            if k not in ("metric", "use_textual")
        }

        y_true: List[Any]
        y_pred: List[Any]
        y_true, y_pred = self._extract_task_arrays(
            predictions=predictions,
            ground_truths=ground_truths,
            task_name=task.task_name,
        )
        if len(y_true) == 0:
            return None

        try:
            value: float = metric_cls.compute(
                y_true=y_true, y_pred=y_pred, **metric_kwargs
            )
            metric_instance: Metric = metric_cls(value=value, **metric_kwargs)
            sample_ids: List[str] = [p.sample_id for p in predictions]
            return NumericFeedback(
                task_name=task.task_name,
                metric=metric_instance,
                aggregated_from_samples=sample_ids,
            )
        except (KeyError, ValueError, ZeroDivisionError) as e:
            raise RuntimeError(
                f"{self.__class__.__name__}: numeric loss computation "
                f"failed for task {task.task_name} ({metric_name}):\n{format_exception_msg(e)}"
            ) from e

    @staticmethod
    def _extract_task_arrays(
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task_name: str,
    ) -> Tuple[List[Any], List[Any]]:
        """Extract aligned (y_true, y_pred) lists for a single task.

        Skips samples where either the prediction or the ground truth is
        missing for the given task.

        Args:
            predictions: Batch of prediction results.
            ground_truths: Batch of ground truth samples.
            task_name: Task to extract values for.

        Returns:
            Tuple of (y_true, y_pred) lists with matching lengths.
        """
        y_true: List[Any] = []
        y_pred: List[Any] = []
        for pred, gt in zip(predictions, ground_truths):
            if task_name in pred.task_outputs and task_name in gt.ground_truths:
                y_true.append(gt.ground_truths[task_name])
                y_pred.append(pred.task_outputs[task_name])
        return y_true, y_pred


class TaskLevelLossComputer(LossComputer):
    """Compute losses independently per task."""

    aliases = ["task-level", "independent"]

    @validate
    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        llm_pool: Optional[Any],  # LLMPool protocol; see types.py
        loss_batch_size: int,
        verbosity: int = 1,
        *,
        loss_functions: Optional[Dict[str, Dict[str, Any]]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        loss_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Compute losses independently for each task.

        Args:
            predictions: Prediction results
            ground_truths: Ground truth samples
            tasks: Tasks to compute losses for
            prompt_template: Current prompt template (carries the optimized instruction)
            llm_pool: Optional LLM for textual feedback
            loss_batch_size: Loss batch size
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            loss_functions: Dict mapping task_name to loss function config
            input_col_labels: Dict mapping column names to display labels (unused)
            loss_task_strategy: Multi-task loss strategy (unused)
            **kwargs: Additional algorithm-specific context

        Returns:
            Dict mapping task_name to list of feedback objects
        """
        if loss_functions is None:
            raise ValueError(
                f"{self.__class__.__name__}.compute() requires 'loss_functions'"
            )
        result: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = {}

        for task in tasks:
            if task.task_name not in loss_functions:
                raise ValueError(
                    f"{self.__class__.__name__}: task {task.task_name!r} not found "
                    f"in loss_functions. Available: {list(loss_functions.keys())}"
                )

            loss_fn_config: Dict[str, Any] = loss_functions[task.task_name]

            task_batches: List[Tuple[List[PredictionResult], List[DatasetSample]]] = (
                LossComputer.batch_predictions(
                    predictions=predictions,
                    ground_truths=ground_truths,
                    task=task,
                    batch_size=loss_batch_size,
                )
            )

            feedbacks: List[Union[NumericFeedback, TextualFeedback]] = []

            # First compute all numeric losses
            for pred_batch, gt_batch in task_batches:
                numeric: Optional[NumericFeedback] = self._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    feedbacks.append(numeric)

            # Optionally compute textual feedback via LLM (batched)
            if llm_pool is not None and loss_fn_config.get("use_textual") is True:
                # Build all prompts first
                prompts: List[str] = []
                for pred_batch, gt_batch in task_batches:
                    feedback_prompt: str = self._build_feedback_prompt(
                        predictions=pred_batch,
                        ground_truths=gt_batch,
                        task=task,
                        prompt_template=prompt_template,
                        loss_fn_config=loss_fn_config,
                    )
                    prompts.append(feedback_prompt)

                # Call LLM with all prompts in a single batch
                if len(prompts) > 0:
                    try:
                        prompts = apply_prompt_suffix(prompts, llm_pool)
                        responses = llm_pool.call_llm_batch(
                            prompts=prompts, verbosity=verbosity
                        ).result(
                            timeout=promptmoo_config.defaults.batch_invocation_timeout
                        )

                        for (pred_batch, _gt_batch), prompt, response in zip(
                            task_batches, prompts, responses
                        ):
                            sample_ids: List[str] = [p.sample_id for p in pred_batch]
                            textual: TextualFeedback = TextualFeedback(
                                task_name=task.task_name,
                                feedback_text=response,
                                aggregated_from_samples=sample_ids,
                                feedback_prompt=prompt,
                            )
                            feedbacks.append(textual)
                    except (RuntimeError, TimeoutError, ValueError) as e:
                        raise RuntimeError(
                            f"{self.__class__.__name__}: textual feedback LLM call "
                            f"failed for task {task.task_name}:\n{format_exception_msg(e)}"
                        ) from e

            result[task] = feedbacks

        return result

    def _build_feedback_prompt(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task: Task,
        prompt_template: PromptTemplate,
        loss_fn_config: dict,
    ) -> str:
        """Build prompt for textual feedback generation.

        Uses the current optimized instruction from ``prompt_template`` rather
        than the frozen ``task.task_instruction`` so the loss LLM evaluates
        predictions against the instruction the task LLM actually received.
        """
        current_instruction: str = self._get_current_instruction(
            task=task, prompt_template=prompt_template
        )
        prompt = f"""You are given evaluation results for the task: {task.task_name}

Task Description: {task.task_description}
Task Instruction: {current_instruction}

Analyze the following predictions and ground truths, and provide feedback on what's wrong:

"""
        for pred, gt in zip(predictions, ground_truths):
            if (
                task.task_name in pred.task_outputs
                and task.task_name in gt.ground_truths
            ):
                prompt += f"Predicted: {pred.task_outputs[task.task_name]}, Ground Truth: {gt.ground_truths[task.task_name]}\n"

        prompt += "\nProvide 2-3 sentences of feedback on what's wrong with these predictions:"
        return prompt

    @staticmethod
    def _get_current_instruction(
        *,
        task: Task,
        prompt_template: PromptTemplate,
    ) -> str:
        """Return the current optimized instruction for *task*.

        Reads from the prompt template's ``instruction`` dict.  Raises
        ``KeyError`` if the task is missing — this indicates a broken
        prompt template, not a condition to fall back from.
        """
        return prompt_template.instruction[task.task_name]
