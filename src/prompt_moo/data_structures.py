"""
Immutable data structures for prompt optimization framework.

All classes use Morphic's Typed pattern for immutability and validation.

This module contains:
- Task: Represents an evaluation task
- DatasetSample: Single sample from dataset
- Batch: Collection of samples for one optimization step
- PredictionResult: LLM prediction result for a sample
- NumericFeedback: Numeric loss/score for evaluation
- TextualFeedback: Textual description of errors/issues
- CombinedFeedback: Combined numeric and textual feedback (for GPO)
- TextGradient: Suggested improvement direction for prompt
- OptimizerResult: Result from prompt optimizer
"""

from typing import Any, Dict, List, Literal, Optional

from morphic import Typed


class Task(Typed):  # Subclassing Typed makes Task immutable
    """Represents an evaluation task with name, description, and instruction.

    This is an immutable data container - once created, fields cannot be modified.

    Task equality and hashing is based on task_name only, so tasks with the
    same name are considered equal even if their instructions differ.

    Attributes:
        task_name: Human-readable name of the task (used for hashing)
        task_description: Description of what the task does
        task_instruction: Initial instruction for the task
        gt_col: Column name in the dataset DataFrame containing ground truth values
    """

    task_name: str
    task_description: str
    task_instruction: str
    gt_col: str  # Dataset column name for ground truth

    def __hash__(self) -> int:
        """Hash based on task_name only."""
        return hash(self.task_name)

    def __eq__(self, other: Any) -> bool:
        """Equality based on task_name only."""
        if not isinstance(other, Task):
            return False
        return self.task_name == other.task_name

    def __str__(self) -> str:
        return f"Task name: {self.task_name}\nTask description: {self.task_description}\nTask instruction: {self.task_instruction}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "task_instruction": self.task_instruction,
        }


class DatasetSample(Typed):
    """Single sample from dataset with inputs and ground truths.

    Attributes:
        sample_id: Unique identifier for this sample
        inputs: Dictionary of input features (input_cols from dataset)
        ground_truths: Dictionary of ground truth values (gt_cols from dataset)
    """

    sample_id: str
    inputs: Dict[str, Any]
    ground_truths: Dict[str, Any]


class Batch(Typed):
    """Collection of samples for one optimization step.

    Attributes:
        step: Step number in the optimization process
        samples: List of DatasetSample objects in this batch
    """

    step: int
    samples: List[DatasetSample]


class PredictionResult(Typed):
    """LLM prediction result for a single sample.

    Attributes:
        sample_id: ID of the sample that was predicted
        task_outputs: Dictionary mapping task names to predicted values
        raw_response: Raw text response from LLM before parsing
        prompt: The prompt that was sent to the LLM (for full observability)
    """

    sample_id: str
    task_outputs: Dict[str, Any]
    raw_response: str
    prompt: Optional[str] = None  # LLM input prompt


class NumericFeedback(Typed):
    """Numeric loss/score for evaluation.

    Attributes:
        task_name: Name of the task this feedback is for
        metric_name: Name of the metric (e.g., "accuracy", "f1", "lce")
        value: Numeric value of the metric
        optimization_direction: Whether to maximize or minimize this metric
        aggregated_from_samples: List of sample IDs this feedback was computed from
    """

    task_name: str
    metric_name: str
    value: float
    optimization_direction: Literal["maximize", "minimize"]
    aggregated_from_samples: List[str]


class TextualFeedback(Typed):
    """Textual description of errors/issues in predictions.

    Attributes:
        task_name: Name of the task this feedback is for
        feedback_text: Textual description of what's wrong
        aggregated_from_samples: List of sample IDs this feedback was based on
    """

    task_name: str
    feedback_text: str
    aggregated_from_samples: List[str]
    feedback_prompt: Optional[str]


class CombinedFeedback(Typed):
    """Combined numeric and textual feedback for the same samples (used in GPO).

    This represents both numeric metrics and textual feedback computed on the same
    set of samples, allowing GPO to generate a single coherent gradient from both
    types of information.

    Attributes:
        task_name: Name of the task this feedback is for
        numeric_feedbacks: List of NumericFeedback objects (may be empty)
        textual_feedbacks: List of TextualFeedback objects (may be empty)
        aggregated_from_samples: List of sample IDs (union of all feedback samples)
    """

    task_name: str
    numeric_feedbacks: List[NumericFeedback]
    textual_feedbacks: List[TextualFeedback]
    aggregated_from_samples: List[str]


class TextGradient(Typed):
    """Suggested improvement direction for prompt.

    Attributes:
        task_name: Name of the task (or combined task identifier)
        gradient_text: Text describing how to improve the prompt
        based_on_feedbacks: IDs of feedback objects this gradient was computed from
        gradient_prompt: The prompt sent to LLM to generate this gradient
    """

    task_name: str
    gradient_text: str
    based_on_feedbacks: List[str]
    gradient_prompt: Optional[str]


class OptimizerResult(Typed):
    """Result from prompt optimizer including LLM interaction details.

    Attributes:
        new_prompt: The updated prompt template (PromptTemplate instance)
        meta_prompt: The meta-prompt sent to the optimizer LLM
        raw_response: The raw response from the optimizer LLM
    """

    new_prompt: Any  # PromptTemplate - using Any to avoid circular import issues
    meta_prompt: str
    raw_response: str

# ---------------------------------------------------------------------------
# Typed metric output classes
# ---------------------------------------------------------------------------

class StepMetricResult(Typed):
    """Metric values for every task at a single evaluation step.

    Attributes:
        step: The optimisation step number.
        split: Dataset split this evaluation was run on (e.g. "eval").
        metric_name: Name of the metric (e.g. "accuracy", "f1").
        task_values: Mapping of task_name -> metric value (None when missing).
    """

    step: int
    split: str
    metric_name: str
    task_values: Dict[str, Optional[float]]


class AlgoMetricSeries(Typed):
    """Metric series across steps for one algorithm run.

    Attributes:
        algo_name: Name of the algorithm (directory basename).
        run_ctx: Reference to the full SingleRunContext for this run.
        split: Dataset split (e.g. "eval").
        metric_name: Name of the metric (e.g. "accuracy", "f1").
        steps: Ordered list of StepMetricResult, one per evaluated step.
    """

    algo_name: str
    run_ctx: Any  # SingleRunContext - using Any to avoid circular import
    split: str
    metric_name: str
    steps: List[StepMetricResult]


class ExptMetricReport(Typed):
    """Metric report aggregating all algorithm runs in one experiment.

    Attributes:
        expt_ctx: Reference to the full ExptRunContext.
        split: Dataset split (e.g. "eval").
        metric_name: Name of the metric (e.g. "accuracy", "f1").
        algo_reports: Mapping of algo_name -> AlgoMetricSeries.
    """

    expt_ctx: Any  # ExptRunContext - using Any to avoid circular import
    split: str
    metric_name: str
    algo_reports: Dict[str, AlgoMetricSeries]

class TaskMetricResult(Typed):
    """
    Metrics computed for a single task at a single step.
    """

    task_name: str
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]

class StepMultiMetricResult(Typed):
    """
    All task metrics for a single step.
    """

    unique_id: str
    algo_name: str
    step: int
    split: str
    task_metrics: List[Any] # TaskMetricResult - using Any to avoid circular import issues