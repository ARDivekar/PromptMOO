"""
Loss Computer: Transforms predictions into feedback (numeric and/or textual).

This is Step 2 of the optimization pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from morphic import Registry, Typed
from sklearn.metrics import f1_score

from .data_structures import (
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextualFeedback,
)
from .llm_workers import LLM, BATCH_INVOCATION_TIMEOUT

# Export validator for use when creating LLM pools
__all__ = [
    "LossComputer",
    "TaskLevelLossComputer",
    "OPROLossComputer",
    "GPOLossComputer",
    "TextGradLossComputer",
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
        llm_pool: Optional[LLM],
        loss_batch_size: int,
        verbosity: int = 1,
        **kwargs: Dict[str, Any],
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Compute feedback for predictions.

        Args:
            predictions: List of prediction results from task predictor
            ground_truths: List of dataset samples with ground truth values
            tasks: List of tasks to compute losses for
            llm_pool: Optional LLM pool for computing textual feedback
            loss_batch_size: Batch size for grouping predictions/samples
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Algorithm-specific context (e.g., loss_functions config)

        Returns:
            Dict with:
            - Keys: task_name (str) or combined tasks (tuple of sorted task names)
            - Values: List of feedback objects (numeric or textual)
        """
        pass


class TaskLevelLossComputer(LossComputer):
    """Compute losses independently per task."""

    aliases = ["task-level", "independent"]

    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        llm_pool: Optional[Any],
        loss_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Compute losses independently for each task.

        Args:
            predictions: Prediction results
            ground_truths: Ground truth samples
            tasks: Tasks to compute losses for
            llm_pool: Optional LLM for textual feedback
            loss_batch_size: Loss batch size
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Must contain 'loss_functions' dict mapping task_name to config

        Returns:
            Dict mapping task_name to list of feedback objects
        """
        loss_functions = kwargs.get("loss_functions", {})
        result = {}

        for task in tasks:
            if task.task_name not in loss_functions:
                continue

            loss_fn_config = loss_functions[task.task_name]

            # Batch predictions for this task
            task_batches = self._batch_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                task=task,
                batch_size=loss_batch_size,
            )

            feedbacks = []

            # First compute all numeric losses
            for pred_batch, gt_batch in task_batches:
                numeric = self._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    feedbacks.append(numeric)

            # Optionally compute textual feedback via LLM (batched)
            if llm_pool is not None and loss_fn_config.get("use_textual", False):
                # Build all prompts first
                prompts = []
                for pred_batch, gt_batch in task_batches:
                    feedback_prompt = self._build_feedback_prompt(
                        predictions=pred_batch,
                        ground_truths=gt_batch,
                        task=task,
                        loss_fn_config=loss_fn_config,
                    )
                    prompts.append(feedback_prompt)

                # Call LLM with all prompts in a single batch
                if len(prompts) > 0:
                    try:
                        responses = llm_pool.call_llm_batch(
                            prompts=prompts, verbosity=verbosity
                        ).result(timeout=BATCH_INVOCATION_TIMEOUT)

                        for (pred_batch, _gt_batch), response in zip(
                            task_batches, responses
                        ):
                            sample_ids = [p.sample_id for p in pred_batch]
                            textual = TextualFeedback(
                                task_name=task.task_name,
                                feedback_text=response,
                                aggregated_from_samples=sample_ids,
                                feedback_prompt=None,  # Not storing prompt for brevity
                            )
                            feedbacks.append(textual)
                    except Exception:
                        # Skip all textual feedback on error
                        pass

            result[task] = feedbacks

        return result

    def _batch_predictions(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task: Task,
        batch_size: int,
    ) -> List[Tuple[List[PredictionResult], List[DatasetSample]]]:
        """Batch predictions for a specific task.

        Returns:
            List of (prediction_batch, ground_truth_batch) tuples
        """
        batches = []
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
        """Compute numeric loss for a batch.

        Args:
            predictions: Batch of predictions
            ground_truths: Batch of ground truths
            task: Task to compute loss for
            loss_fn_config: Loss function configuration

        Returns:
            NumericFeedback object or None if computation fails
        """
        metric_name = loss_fn_config.get("metric", "accuracy")
        metric_name_lower = metric_name.lower()

        # Determine optimization direction
        if metric_name_lower in ["accuracy", "f1"]:
            direction = "maximize"
        else:
            direction = "minimize"

        try:
            if metric_name_lower in ["accuracy", "acc"]:
                value = self._compute_accuracy(
                    predictions, ground_truths, task.task_name
                )
            elif metric_name_lower in ["f1"]:
                value = self._compute_f1(predictions, ground_truths, task.task_name)
            elif metric_name_lower in ["lce", "ce"]:
                value = self._compute_lce(predictions, ground_truths, task.task_name)
            else:
                return None

            sample_ids = [p.sample_id for p in predictions]
            return NumericFeedback(
                task_name=task.task_name,
                metric_name=metric_name,
                value=value,
                optimization_direction=direction,
                aggregated_from_samples=sample_ids,
            )
        except Exception:
            return None

    def _compute_accuracy(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task_name: str,
    ) -> float:
        """Compute accuracy."""
        correct = 0
        total = 0
        for pred, gt in zip(predictions, ground_truths):
            if task_name in pred.task_outputs and task_name in gt.ground_truths:
                if pred.task_outputs[task_name] == gt.ground_truths[task_name]:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.0

    def _compute_f1(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task_name: str,
    ) -> float:
        """Compute F1 score."""
        y_true = []
        y_pred = []
        for pred, gt in zip(predictions, ground_truths):
            if task_name in pred.task_outputs and task_name in gt.ground_truths:
                y_true.append(gt.ground_truths[task_name])
                y_pred.append(pred.task_outputs[task_name])

        if len(y_true) == 0:
            return 0.0

        return float(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

    def _compute_lce(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task_name: str,
    ) -> float:
        """Compute log cross-entropy."""
        losses = []
        for pred, gt in zip(predictions, ground_truths):
            if task_name in pred.task_outputs and task_name in gt.ground_truths:
                pred_val = pred.task_outputs[task_name]
                # Normalize to probability
                prob = pred_val / 5.0
                losses.append(-np.log(prob + 1e-12))

        return float(np.mean(losses)) if len(losses) > 0 else 0.0

    def _build_feedback_prompt(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task: Task,
        loss_fn_config: dict,
    ) -> str:
        """Build prompt for textual feedback generation."""
        prompt = f"""You are given evaluation results for the task: {task.task_name}

Task Description: {task.task_description}
Task Instruction: {task.task_instruction}

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


class OPROLossComputer(LossComputer):
    """OPRO-specific: only numeric losses, no textual feedback."""

    aliases = ["opro"]

    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        llm_pool: Optional[Any],
        loss_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Compute only numeric losses for OPRO.

        Args:
            predictions: Prediction results
            ground_truths: Ground truth samples
            tasks: Tasks to compute losses for
            llm_pool: Unused (OPRO doesn't use textual feedback)
            loss_batch_size: Batch size for processing predictions
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Must contain 'loss_functions' dict

        Returns:
            Dict mapping task_name to list of NumericFeedback objects
        """
        # OPRO doesn't use textual feedback
        if llm_pool is not None:
            pass  # Ignore, don't assert to allow flexibility

        loss_functions = kwargs.get("loss_functions", {})
        result = {}

        # Use TaskLevelLossComputer's methods
        task_computer = TaskLevelLossComputer()

        for task in tasks:
            if task.task_name not in loss_functions:
                continue

            loss_fn_config = loss_functions[task.task_name]

            # Batch predictions for this task
            task_batches = task_computer._batch_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                task=task,
                batch_size=loss_batch_size,
            )

            feedbacks = []
            for pred_batch, gt_batch in task_batches:
                # Compute numeric loss for each batch
                numeric = task_computer._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    feedbacks.append(numeric)

            result[task] = feedbacks

        return result


class GPOLossComputer(LossComputer):
    """
    GPO-specific Loss Computer:
    - Computes numeric feedback
    - Optionally uses LLM to generate textual feedback (like TaskLevel)
    - Designed to be light-weight and multi-task aware
    """

    aliases = ["gpo"]

    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        llm_pool: Optional[Any],
        loss_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """
        Compute feedback for GPO
        - Numeric for Task (always)
        - Textual feedback if llm_pool is provided and enabled for that task
        """

        loss_functions = kwargs.get("loss_functions", {})
        if verbosity >= 2:
            print(loss_functions)
        result: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = {}

        task_computer = TaskLevelLossComputer()

        for task in tasks:
            if task.task_name not in loss_functions:
                continue

            loss_fn_config = loss_functions[task.task_name]

            # Batch predictions for this task
            task_batches = task_computer._batch_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                task=task,
                batch_size=loss_batch_size,
            )

            feedbacks = []

            # First compute all numeric losses
            for pred_batch, gt_batch in task_batches:
                numeric = task_computer._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    feedbacks.append(numeric)

            # Optionally compute textual feedback via LLM (batched)
            if llm_pool is not None and loss_fn_config.get("use_textual", False):
                # Build all prompts first
                prompts = []
                for pred_batch, gt_batch in task_batches:
                    feedback_prompt = task_computer._build_feedback_prompt(
                        predictions=pred_batch,
                        ground_truths=gt_batch,
                        task=task,
                        loss_fn_config=loss_fn_config,
                    )
                    prompts.append(feedback_prompt)

                # Call LLM with all prompts in a single batch
                if len(prompts) > 0:
                    try:
                        responses = llm_pool.call_llm_batch(
                            prompts=prompts, verbosity=verbosity
                        ).result(timeout=BATCH_INVOCATION_TIMEOUT)

                        for (pred_batch, _gt_batch), response in zip(
                            task_batches, responses
                        ):
                            sample_ids = [p.sample_id for p in pred_batch]
                            textual = TextualFeedback(
                                task_name=task.task_name,
                                feedback_text=response,
                                aggregated_from_samples=sample_ids,
                                feedback_prompt=None,  # Not storing prompt for brevity
                            )
                            feedbacks.append(textual)
                    except Exception:
                        # Skip all textual feedback on error
                        pass

            result[task] = feedbacks

        return result


class TextGradLossComputer(LossComputer):
    """
    TextGrad-specific Loss Computer:
    - Computes textual feedback only (no numeric metrics)
    - Uses LLM to generate rich textual feedback for gradient computation
    - Designed for natural language optimization signals
    """

    aliases = ["textgrad"]

    def compute(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        llm_pool: Optional[Any],
        loss_batch_size: int,
        verbosity: int = 1,
        **kwargs,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """
        Compute textual feedback for TextGrad.

        Args:
            predictions: Model predictions
            ground_truths: Ground truth samples
            tasks: List of tasks to compute feedback for
            llm_pool: LLM worker pool for generating textual feedback
            loss_batch_size: Batch size for processing predictions
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            **kwargs: Additional context (should contain 'loss_functions')

        Returns:
            Dict mapping task names to list of TextualFeedback objects
        """
        if llm_pool is None:
            raise ValueError(
                "TextGrad requires an LLM pool for textual feedback generation"
            )

        loss_functions = kwargs.get("loss_functions", {})
        result: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = {}

        task_computer = TaskLevelLossComputer()

        for task in tasks:
            task_name = task.task_name

            if task_name not in loss_functions:
                if verbosity >= 2:
                    print(f"Task {task_name} not found")
                continue

            loss_fn_config = loss_functions[task_name]

            # Batch predictions for this task
            task_batches = task_computer._batch_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                task=task,
                batch_size=loss_batch_size,
            )

            # TextGrad only uses textual feedback - build all prompts first
            prompts = []
            for pred_batch, gt_batch in task_batches:
                feedback_prompt = task_computer._build_feedback_prompt(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                prompts.append(feedback_prompt)

            # Call LLM with all prompts in a single batch
            feedbacks: List[Union[NumericFeedback, TextualFeedback]] = []
            if len(prompts) > 0:
                try:
                    responses = llm_pool.call_llm_batch(
                        prompts=prompts, verbosity=verbosity
                    ).result(timeout=BATCH_INVOCATION_TIMEOUT)

                    for (pred_batch, _gt_batch), response in zip(
                        task_batches, responses
                    ):
                        sample_ids = [p.sample_id for p in pred_batch]
                        textual = TextualFeedback(
                            task_name=task.task_name,
                            feedback_text=response,
                            aggregated_from_samples=sample_ids,
                            feedback_prompt=None,  # Not storing prompt for brevity
                        )
                        feedbacks.append(textual)
                except Exception:
                    # Skip all textual feedback on error
                    pass

            result[task] = feedbacks

        return result
