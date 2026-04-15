"""OPRO algorithm, optimizer, gradient computer, and loss computer."""

import warnings
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from morphic import validate
from pydantic import Field, PrivateAttr

from ..data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from ..gradient_computer import GradientComputer
from ..metrics import Metric
from ..prompt_algorithm import PromptAlgorithm
from ..prompt_optimizer import PromptOptimizer
from ..prompt_template import PromptTemplate
from ..prompt_trajectory import (
    PromptTrajectory,
    TrajectoryElement,
)
from ..types import extract_instructions_json


class OPROTrajectoryElement(TrajectoryElement):
    """OPRO trajectory element: ``Instruction: {text} / Score: {score}``.

    Uses ``"Instruction"`` as the label, matching the OPRO paper's
    ``text: .../score: N`` format (application.tex Fig 2).
    """

    aliases = ["opro"]

    def __str__(self) -> str:
        lines = []
        if isinstance(self.instructions, str):
            lines.append(f"Instruction: {self.instructions}")
            for task_name, feedbacks in self.numeric_scores.items():
                for fb in feedbacks:
                    lines.append(f"Score: {fb.display_score}")
        else:
            for task_name, instruction_text in self.instructions.items():
                lines.append(f"{task_name}:")
                lines.append(f"- Instruction: {instruction_text}")
                if task_name in self.numeric_scores:
                    for fb in self.numeric_scores[task_name]:
                        lines.append(f"- Score: {fb.display_score}")
        return "\n".join(lines)


class OPRO(PromptAlgorithm):
    """OPRO algorithm implementation (Yang et al., ICLR 2024).

    OPRO uses a top-k trajectory of (prompt, score) pairs to guide
    optimization.  It has exactly two LLM roles: a scorer (task LLM) and
    an optimizer LLM.  There is no gradient LLM or loss LLM.

    Key paper hyperparameters:
        k: 20 (top instructions kept in trajectory)
        num_candidates: 8 (instructions generated per step)
        num_task_demonstrations: 3 (exemplars in meta-prompt)
        trajectory_order: worst_to_best (ascending, best last)
    """

    loss_computer: Dict[str, Any] = Field(
        default_factory=lambda: {"name": "task-level"}
    )
    gradient_computer: Dict[str, Any] = Field(default_factory=lambda: {"name": "opro"})
    prompt_optimizer: Dict[str, Any] = Field(default_factory=lambda: {"name": "opro"})

    k: int = 20  # Paper default: top 20 instructions in trajectory
    task_losses: Dict[str, str] = Field(default_factory=dict)
    num_task_demonstrations: int = 3
    num_candidates: int = 8  # Paper: 8 candidates per step
    trajectory_order: Literal["worst_to_best", "best_to_worst"] = "worst_to_best"
    use_fixed_training_batch: bool = True  # Paper: same subset across all steps

    # Paper (exp.tex L33-34): scorer LLM uses temperature 0 for greedy
    # decoding; optimizer LLM uses temperature 1.0 for diverse candidate
    # generation. Ablation (exp.tex L354-358) confirms 1.0 is optimal.
    task_llm_temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    optimizer_llm_temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        """Default loss_batch_size and gradient_batch_size to batch_size,
        and force unused LLM roles to None.

        OPRO evaluates the entire training batch as one unit (one aggregate
        score enters the trajectory).

        The OPRO paper has exactly two LLM roles: a scorer (task LLM) and
        an optimizer LLM.  There is no loss LLM (scoring is programmatic)
        and no gradient LLM (the optimizer infers improvement direction from
        the (prompt, score) trajectory, not from natural-language critiques).
        Force both to None so the runner does not waste resources creating
        workers that will never be called.
        """
        batch_size = data.get("batch_size")
        if batch_size is not None:
            if data.get("loss_batch_size") is None:
                data["loss_batch_size"] = batch_size
            if data.get("gradient_batch_size") is None:
                data["gradient_batch_size"] = batch_size

        if data.get("loss_llm") is not None:
            warnings.warn(
                "OPRO does not use a loss LLM (scoring is programmatic). "
                "The provided loss_llm will be ignored and set to None.",
                stacklevel=2,
            )
        data["loss_llm"] = None

        if data.get("gradient_llm") is not None:
            warnings.warn(
                "OPRO does not use a gradient LLM (no textual gradients). "
                "The provided gradient_llm will be ignored and set to None.",
                stacklevel=2,
            )
        data["gradient_llm"] = None

    def post_initialize(self) -> None:
        """Validate that loss and gradient batch sizes equal the training batch size.

        OPRO computes one aggregate score per instruction across the
        entire training batch.  If loss_batch_size != batch_size, the loss
        computer would produce multiple NumericFeedback objects per task,
        resulting in multiple score lines per trajectory element instead of
        one.  The paper's trajectory format is strictly one score per
        instruction (exp.tex L36, application.tex Fig 2).
        """
        if self.loss_batch_size != self.batch_size:
            raise ValueError(
                f"OPRO requires loss_batch_size == batch_size "
                f"(got loss_batch_size={self.loss_batch_size}, batch_size={self.batch_size}). "
                f"The OPRO paper computes one aggregate score per instruction "
                f"over the full training batch."
            )
        if self.gradient_batch_size != self.batch_size:
            raise ValueError(
                f"OPRO requires gradient_batch_size == batch_size "
                f"(got gradient_batch_size={self.gradient_batch_size}, batch_size={self.batch_size}). "
                f"The OPRO paper evaluates the full training batch as one unit."
            )

    _trajectory: Optional[PromptTrajectory] = PrivateAttr(default=None)

    @property
    def trajectory(self) -> PromptTrajectory:
        """Lazy-initialized trajectory heap.

        Uses ``task_losses`` to determine the metric priority and ``k``
        to set the bounded capacity.
        """
        if self._trajectory is None:
            metrics = list(dict.fromkeys(self.task_losses.values()))
            self._trajectory = PromptTrajectory(
                k=self.k,
                order=self.trajectory_order,
                metric_priority=metrics,
            )
        return self._trajectory

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """Build the algorithm context dict passed to all pipeline components.

        Returns:
            Dict with keys: trajectory, loss_functions, task_demonstrations.
        """
        loss_functions = {}
        for task in self.tasks:
            if task.task_name in self.task_losses:
                loss_functions[task.task_name] = self._build_loss_fn_config(
                    task_name=task.task_name,
                    use_textual=False,
                )
            else:
                raise ValueError(f"Unsupported task: {task.task_name}")
        return {
            "trajectory": self.trajectory,
            "loss_functions": loss_functions,
            "task_demonstrations": self._require_task_demonstrations(),
        }

    def _before_optimize(
        self,
        *,
        step: int,
        feedbacks: Dict,
        gradients: Dict,
        current_prompt: PromptTemplate,
    ) -> None:
        """Push the current prompt's (instruction, score) to the trajectory.

        Why this hook exists (paper reference: application.tex L98-101):

        The OPRO paper seeds the trajectory with the initial instruction
        and its score before optimization begins.  In the PromptMOO loop,
        the initial prompt is scored in Step 2 (loss computation) of step 0,
        but the optimizer runs in Step 4 of that same step.  By pushing the
        (instruction, score) pair here — after feedbacks are computed
        (Step 2-3) but before the optimizer runs (Step 4) — the meta-prompt
        always contains at least one (instruction, score) pair, even on the
        very first step.  This matches the paper's invariant that the
        optimizer always has trajectory context to work with.

        On subsequent steps, this pushes the *current* prompt's score
        (computed on that step's batch) before the optimizer proposes a
        replacement.  Candidate prompts generated in Step 4 are pushed
        separately in ``_update_state``.
        """
        numeric_scores: Dict[str, List[NumericFeedback]] = {}
        for task, feedback_list in feedbacks.items():
            task_fbs = [fb for fb in feedback_list if isinstance(fb, NumericFeedback)]
            if len(task_fbs) > 0:
                numeric_scores[task.task_name] = task_fbs

        element = OPROTrajectoryElement(
            instructions=current_prompt.instruction,
            numeric_scores=numeric_scores,
        )
        self.trajectory.push(element)

    def _update_state(
        self,
        step: int,
        feedbacks: Dict,
        gradients: Dict,
        current_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        all_candidates: Optional[List[PromptTemplate]] = None,
        all_candidate_scores: Optional[List[Dict[Task, List[NumericFeedback]]]] = None,
    ) -> None:
        """Push evaluated candidates to the trajectory.

        The current prompt's (instruction, score) pair was already pushed
        by ``_before_optimize`` before the optimizer ran.  Here we only
        add the candidate prompts generated and scored during Step 4.

        When multi-candidate generation is active (num_candidates > 1),
        ALL candidates and their evaluation scores are pushed, matching
        the OPRO paper where all 8 candidates per step enter the history.
        """
        if all_candidates is not None and all_candidate_scores is not None:
            for cand, cand_scores in zip(all_candidates, all_candidate_scores):
                cand_ns: Dict[str, List[NumericFeedback]] = {
                    task.task_name: fbs for task, fbs in cand_scores.items()
                }
                cand_element = OPROTrajectoryElement(
                    instructions=cand.instruction,
                    numeric_scores=cand_ns,
                )
                self.trajectory.push(cand_element)

        if self.verbosity >= 2:
            print(f"\n[OPRO Debug] Trajectory ({len(self.trajectory)}/{self.k}):")
            print(self.trajectory.get_top_k_str())

    def _serialize_algorithm_context(self, *, step: int) -> Dict[str, Any]:
        """Serialize algorithm context for observability logging.

        Returns:
            Dict with trajectory elements, trajectory capacity, loss function
            configs, and task demonstrations (when present).
        """
        ctx: Dict[str, Any] = {
            "trajectory": self.trajectory.to_serializable_list(),
            "trajectory_k": self.trajectory.k,
            "loss_functions": {
                task.task_name: self._build_loss_fn_config(
                    task_name=task.task_name,
                    use_textual=False,
                )
                for task in self.tasks
                if task.task_name in self.task_losses
            },
        }
        demos = self._require_task_demonstrations()
        if len(demos) > 0:
            ctx["task_demonstrations"] = [d.model_dump() for d in demos]
        return ctx

    def _serialize_algorithm_state(self) -> Dict[str, Any]:
        """Serialize algorithm state for observability logging.

        Returns:
            Dict with trajectory elements, capacity info, and current size.
        """
        return {
            "trajectory": self.trajectory.to_serializable_list(),
            "trajectory_k": self.trajectory.k,
            "trajectory_size": len(self.trajectory),
            "k": self.k,
        }


class OPROOptimizer(PromptOptimizer):
    """OPRO-specific optimizer (Yang et al., ICLR 2024).

    The meta-prompt follows the paper's three-component structure:
    1. Meta-instructions telling the optimizer what to do
    2. Optimization trajectory: past (instruction, score) pairs
    3. Task exemplars: (input, ground_truth) pairs with ``<INS_task>``
       placeholders showing where each instruction will be inserted
    """

    aliases: ClassVar[List[str]] = ["opro"]

    @validate
    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        trajectory: PromptTrajectory,
        loss_functions: Dict[str, Dict[str, Any]],
        batch: Optional[Batch] = None,
        task_demonstrations: List[DatasetSample],
        input_col_labels: Optional[Dict[str, str]] = None,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        optimizer_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Build the OPRO meta-prompt.

        The meta-prompt has three sections (no full prompt template shown):
        1. Meta-instructions + trajectory (with ``---`` delimiters)
        2. Task exemplars with ``<INS_taskname>`` placeholders
        3. Generation instruction + JSON output format

        The trajectory order description is dynamically generated from
        ``trajectory.order``.

        Args:
            gradients: Unused by OPRO (no gradient section in meta-prompt).
            current_prompt: Current prompt template (used to extract the
                evaluation directive for exemplar formatting).
            tasks: List of tasks to generate instructions for.
            trajectory: PromptTrajectory containing past (instruction, score)
                pairs.
            loss_functions: Mapping of task_name -> loss config dict with
                a ``"metric"`` key (e.g. ``{"metric": "accuracy"}``).
                Used to build the score direction description.
            task_demonstrations: List of DatasetSample exemplars.
            input_col_labels: Mapping of column names to display labels.
            **kwargs: Absorbed for interface compatibility.

        Returns:
            The complete meta-prompt string.
        """
        top_k_str: str = ""
        if len(trajectory) > 0:
            top_k_str = trajectory.get_top_k_str()

        task_names: List[str] = [t.task_name for t in tasks]

        evaluation_directive: str = current_prompt.skeleton.split("\n")[0].rstrip(".")
        demos_section: str = self._format_task_exemplars(
            demonstrations=task_demonstrations,
            tasks=tasks,
            input_col_labels=current_prompt.input_col_labels,
            evaluation_directive=evaluation_directive,
        )

        score_direction_description: str = self._build_score_direction_description(
            loss_functions=loss_functions,
        )

        if trajectory.order == "worst_to_best":
            order_description: str = (
                "The instructions are arranged from worst to best, "
                "where the best-performing instructions appear last."
            )
        elif trajectory.order == "best_to_worst":
            order_description: str = (
                "The instructions are arranged from best to worst, "
                "where the best-performing instructions appear first."
            )
        else:
            raise ValueError(
                f"Unknown trajectory order: {trajectory.order!r}. "
                f"Must be 'worst_to_best' or 'best_to_worst'."
            )

        example_instruction: str = ",\n".join(
            [f'    "{tn}": "improved instruction for {tn}"' for tn in task_names]
        )

        meta_prompt: str = f"""Your task is to generate improved instructions for each task listed below. Below are some previous instruction sets with their scores. {order_description}
{score_direction_description}
{top_k_str}
---
{demos_section}
Generate an instruction for each task that is different from all the instruction sets above and improves the performance scores. The instructions should be concise, effective, and generally applicable to all inputs.

Return ONLY a valid JSON object in this format:
{{
  "instructions": {{
{example_instruction}
  }}
}}

DO NOT include any explanations or text outside the JSON.
"""
        return meta_prompt

    @validate
    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Parse the optimizer LLM response into a task -> instruction dict."""
        task_names: List[str] = [t.task_name for t in tasks]
        return extract_instructions_json(response, task_names=task_names)

    @staticmethod
    def _format_task_exemplars(
        *,
        demonstrations: List[DatasetSample],
        tasks: List[Task],
        input_col_labels: Dict[str, str],
        evaluation_directive: str,
    ) -> str:
        """Format task exemplars for the OPRO meta-prompt.

        Follows the OPRO paper's exemplar structure (application.tex Fig 2)
        adapted for multi-task: each exemplar shows the input data, the
        evaluation directive with per-task ``<INS_taskname>`` placeholders
        indicating where each instruction will be inserted, and the
        expected output.

        The ``<INS_taskname>`` placeholders serve the same role as the
        paper's ``<INS>`` in ``A: <INS>``.

        Args:
            demonstrations: List of DatasetSample exemplars from training set.
            tasks: List of tasks (order is preserved in placeholder listing).
            input_col_labels: Mapping of column names to display labels.
            evaluation_directive: First line of the skeleton (e.g.
                "Evaluate the Summary of the Source Text").

        Returns:
            Formatted exemplar section string, or empty string if no
            demonstrations are provided.
        """
        if len(demonstrations) == 0:
            return ""

        task_names: List[str] = [t.task_name for t in tasks]
        ins_placeholders: str = ", ".join(f"<INS_{tn}>" for tn in task_names)

        lines: List[str] = [
            "",
            f"The following exemplars show how your instructions are applied. "
            f"You replace {ins_placeholders} in the prompt with your "
            f"instructions, then the task LLM reads the input and produces "
            f"an output. The output is correct if it matches the expected output.",
            "",
        ]
        for i, demo in enumerate(demonstrations, 1):
            lines.append(f"Exemplar {i}:")
            lines.append("  Input:")
            for col, val in demo.inputs.items():
                if col not in input_col_labels:
                    raise ValueError(
                        f"OPRO._format_task_demonstrations: column {col!r} has no label in "
                        f"input_col_labels. Available: {list(input_col_labels.keys())}."
                    )
                label: str = input_col_labels[col]
                lines.append(f"    {label}: {val}")
            lines.append(f"  Instructions: {evaluation_directive}")
            for tn in task_names:
                lines.append(f"    {tn}: <INS_{tn}>")
            gt_parts: List[str] = []
            for task in tasks:
                if task.task_name not in demo.ground_truths:
                    raise ValueError(
                        f"OPROOptimizer._format_task_exemplars: demonstration "
                        f"sample {demo.sample_id!r} has no ground truth for "
                        f"task {task.task_name!r}. "
                        f"Available ground truth keys: {list(demo.ground_truths.keys())}. "
                        f"The OPRO paper requires exemplars to show the expected "
                        f"output for every task (application.tex Fig 2)."
                    )
                gt_val: Any = demo.ground_truths[task.task_name]
                gt_parts.append(f"{task.task_name}: {gt_val}")
            lines.append(f"  Expected output: {', '.join(gt_parts)}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_score_direction_description(
        *,
        loss_functions: Dict[str, Dict[str, Any]],
    ) -> str:
        """Build a description of what each score means (higher/lower is better).

        Looks up each task's metric from the Metric registry to determine
        the optimization direction, then produces a human-readable summary.
        When all tasks use the same direction and range, produces a single
        sentence.  When mixed, lists each task's direction individually.

        Args:
            loss_functions: Mapping of task_name -> loss config dict with
                a ``"metric"`` key (e.g. ``{"metric": "accuracy"}``).

        Returns:
            Description string (may be empty if no loss_functions provided).
        """
        if len(loss_functions) == 0:
            return ""

        directions: Dict[str, str] = {}
        ranges: Dict[str, str] = {}
        for task_name, config in loss_functions.items():
            metric_name: str = config["metric"]
            metric_cls = Metric.get_subclass(metric_name)
            direction: str = metric_cls.display_direction
            directions[task_name] = direction
            low, high = metric_cls.display_score_range
            nc: Optional[int] = config.get("num_classes")
            if nc is not None and nc > 0 and high == float("inf"):
                high = float(nc - 1)
            high_str: str = (
                str(int(high))
                if high != float("inf") and high == float(int(high))
                else ("∞" if high == float("inf") else str(high))
            )
            low_str: str = str(int(low)) if low == float(int(low)) else str(low)
            ranges[task_name] = f"{low_str} to {high_str}"

        unique_directions = set(directions.values())
        unique_ranges = set(ranges.values())

        if len(unique_directions) == 1 and len(unique_ranges) == 1:
            direction = next(iter(unique_directions))
            score_range = next(iter(unique_ranges))
            return (
                f"Scores range from {score_range}; "
                f"{direction} scores indicate better performance."
            )

        parts: List[str] = []
        for task_name in loss_functions:
            parts.append(
                f"  - {task_name}: {ranges[task_name]}, "
                f"{directions[task_name]} is better"
            )
        return "Score interpretation:\n" + "\n".join(parts)


class OPROGradientComputer(GradientComputer):
    """OPRO gradient computer: deterministic score summary, no LLM calls.

    OPRO does not use textual gradients.  The optimizer LLM infers
    improvement direction entirely from the (prompt, score) trajectory.
    This component produces a lightweight summary for observability
    logging only; it is never included in the optimizer meta-prompt.
    """

    aliases = ["opro"]

    @validate
    def compute(
        self,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Optional[Any],  # LLMPool protocol; see types.py
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
        """Convert numeric feedbacks to summary gradients (no LLM calls).

        Produces one ``TextGradient`` per task containing the average
        numeric score.  The ``gradient_prompt`` is ``None`` (no LLM was
        called), which downstream components use to detect non-LLM
        gradients.

        Args:
            feedbacks: Per-task feedback lists from the loss computer.
            prompt_template: Current prompt (unused by OPRO).
            tasks: List of tasks (unused by OPRO).
            llm_pool: LLM pool (unused by OPRO; should be None).
            gradient_batch_size: Batch size (unused; OPRO aggregates all).
            verbosity: Logging verbosity level.

        Returns:
            Dict mapping each Task to a single-element list containing the
            score summary TextGradient.
        """
        result: Dict[Task, List[TextGradient]] = {}

        for task, feedback_list in feedbacks.items():
            scores: List[float] = [
                fb.value for fb in feedback_list if isinstance(fb, NumericFeedback)
            ]

            if len(scores) > 0:
                avg_score: float = sum(scores) / len(scores)
                gradient: TextGradient = TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Average score: {avg_score:.4f}",
                    based_on_feedbacks=[],
                    gradient_prompt=None,
                )
                result[task] = [gradient]
            else:
                result[task] = []

        return result
