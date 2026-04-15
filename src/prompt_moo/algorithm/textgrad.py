"""TextGrad algorithm, optimizer, gradient computer, and loss computer."""

from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Type, Union

from morphic import validate
from morphic.typed import format_exception_msg
import pandas as pd
from pydantic import Field, PrivateAttr, conint

from ..config import promptmoo_config
from ..data_input import Dataset
from ..data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    OptimizerResult,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from ..gradient_computer import GradientComputer
from ..llm_utils import apply_prompt_suffix
from ..loss_computer import LossComputer
from ..metrics import Metric
from ..observability import ObservabilityManager
from ..prompt_algorithm import PromptAlgorithm
from ..prompt_optimizer import PromptOptimizer
from ..prompt_template import PromptTemplate
from ..prompt_trajectory import PromptTrajectory
from ..types import (
    COMBINE_ALL_TASKS,
    SEPARATE_TASKS,
    extract_instructions_json,
    strip_smart_quotes,
)

BACKWARD_SYSTEM_PROMPT: str = (
    "You are part of an optimization system that improves a given text "
    "(i.e. the variable). You are the gradient (feedback) engine. "
    "Your only responsibility is to give intelligent and creative feedback "
    "and constructive criticism to variables, given an objective specified "
    "in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
    "The variables may be solutions to problems, prompts to language models, "
    "code, or any other text-based variable. "
    "Pay attention to the role description of the variable, and the context "
    "in which it is used. You should assume that the variable will be used "
    "in a similar context in the future. "
    "Only provide strategies, explanations, and methods to change in the "
    "variable. DO NOT propose a new version of the variable, that will be "
    "the job of the optimizer. Your only job is to send feedback and "
    "criticism (compute 'gradients'). "
    "For instance, feedback can be in the form of "
    "'Since language models have the X failure mode...', "
    "'Adding X can fix this error because...', "
    "'Removing X can improve the objective function because...', "
    "'Changing X to Y would fix the mistake ...', "
    "that gets at the downstream objective.\n"
    "If a variable is already working well (e.g. the objective function is "
    "perfect, an evaluation shows the response is accurate), describe what "
    "aspects of the variable are working and why they lead to correct "
    "results. This positive feedback is essential for the optimizer to know "
    "what to preserve.\n\n"
    "### Glossary of tags that will be sent to you:\n"
    "# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.\n"
    "# - <LM_INPUT>: The input to the language model.\n"
    "# - <LM_OUTPUT>: The output of the language model.\n"
    "# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.\n"
    "# - <VARIABLE>: Specifies the span of the variable.\n"
    "# - <ROLE>: The role description of the variable."
)


class TextGrad(PromptAlgorithm):
    """TextGrad algorithm implementation (Yuksekgonul et al., 2024).

    TextGrad backpropagates natural-language feedback through the computation
    graph. Each instance gets its own per-instance loss and gradient;
    gradients are concatenated (analogous to ``tg.sum``).

    Per the paper's minibatch SGD design, ``loss_batch_size`` and
    ``gradient_batch_size`` MUST both be 1.

    Multi-task strategy:
        Each of the three downstream stages (loss, gradient, optimizer) can
        independently operate in ``"separate_tasks"`` or ``"combine_all_tasks"``
        mode.  The constraint is: once a stage combines, all downstream stages
        must also combine (because combined output cannot be split back into
        per-task pieces).

        Valid configurations (enforced in ``post_initialize``)::

            Loss          Gradient      Optimizer
            -----------   -----------   -----------
            separate      separate      separate      (S/S/S)
            separate      separate      combine       (S/S/C) [default]
            separate      combine       combine       (S/C/C)
            combine       combine       combine       (C/C/C)

        Any other combination raises ``ValueError`` at construction time.

    Testing:
        - Unit tests for strategy validation (no LLM calls):
          ``tests/test_textgrad_strategies.py``
        - E2E tests for all 4 valid modes with real LLM calls and artifact
          inspection: ``tests/test_e2e_textgrad_modes.py``
    """

    aliases = ["text-g", "textgrad"]

    loss_computer: Dict[str, Any] = Field(default_factory=lambda: {"name": "textgrad"})
    gradient_computer: Dict[str, Any] = Field(
        default_factory=lambda: {"name": "textgrad"}
    )
    prompt_optimizer: Dict[str, Any] = Field(
        default_factory=lambda: {"name": "textgrad"}
    )

    loss_task_strategy: Literal["separate_tasks", "combine_all_tasks"] = (
        "separate_tasks"
    )
    gradient_task_strategy: Literal["separate_tasks", "combine_all_tasks"] = (
        "separate_tasks"
    )
    optimizer_task_strategy: Literal["separate_tasks", "combine_all_tasks"] = (
        "combine_all_tasks"
    )

    # TextGrad processes per-instance losses and gradients, so the training
    # batch should be small to keep API cost manageable.
    batch_size: int = 3

    # Paper: task LLM uses low temperature for deterministic predictions;
    # loss, gradient, and optimizer LLMs use higher temperature for
    # critique generation and creative prompt rewriting.
    task_llm_temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    optimizer_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    gradient_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    loss_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

    # Per-task metric names for numeric loss computation.
    # Maps task_name -> metric registry key (e.g. "accuracy", "f1", "lce").
    task_losses: Dict[str, str] = Field(default_factory=dict)

    # --- Validation gate (TextGrad paper Section 3.3) ---
    # After each optimization step, evaluate the new prompt on a held-out
    # validation batch.  Accept the update only if the validation score does
    # not regress.  This prevents the cascading-failure mode where a bad
    # update compounds across steps (TextGrad has no trajectory safety net).
    validation_gate_samples: conint(ge=1) = 100
    # Metric registry name (e.g. "accuracy", "f1", "lce"), or "none" to disable
    validation_metric: str

    _validation_metric_cls: Optional[Type[Metric]] = PrivateAttr(default=None)
    _current_validation_score: Optional[float] = PrivateAttr(default=None)
    _validation_batch: Optional[Batch] = PrivateAttr(default=None)

    _previous_instructions: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _last_predictions: Optional[List[PredictionResult]] = PrivateAttr(default=None)
    _last_ground_truths: Optional[List[DatasetSample]] = PrivateAttr(default=None)

    @property
    def validation_gate_enabled(self) -> bool:
        """Whether the validation gate is active (False when validation_metric='none')."""
        return self.validation_metric != "none"

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        """Auto-default loss_batch_size and gradient_batch_size to 1.

        TextGrad computes per-instance losses and per-instance backward-pass
        gradients that are concatenated via ``tg.sum``.  When the user omits
        these fields (``None``), they are filled with ``1``.
        """
        if data.get("loss_batch_size") is None:
            data["loss_batch_size"] = 1
        if data.get("gradient_batch_size") is None:
            data["gradient_batch_size"] = 1

    def post_initialize(self) -> None:
        if self.loss_batch_size != 1:
            raise ValueError(
                f"TextGrad requires loss_batch_size=1 (got {self.loss_batch_size})."
            )
        if self.gradient_batch_size != 1:
            raise ValueError(
                f"TextGrad requires gradient_batch_size=1 (got {self.gradient_batch_size})."
            )

        if self.loss_task_strategy == COMBINE_ALL_TASKS:
            if self.gradient_task_strategy != COMBINE_ALL_TASKS:
                raise ValueError(
                    f"loss_task_strategy='{COMBINE_ALL_TASKS}' produces combined "
                    f"multi-task feedback that cannot be split back into per-task "
                    f"pieces. gradient_task_strategy must also be '{COMBINE_ALL_TASKS}' "
                    f"(got '{self.gradient_task_strategy}')."
                )
        if self.gradient_task_strategy == COMBINE_ALL_TASKS:
            if self.optimizer_task_strategy != COMBINE_ALL_TASKS:
                raise ValueError(
                    f"gradient_task_strategy='{COMBINE_ALL_TASKS}' produces combined "
                    f"multi-task gradients that cannot be split back into per-task "
                    f"pieces. optimizer_task_strategy must also be '{COMBINE_ALL_TASKS}' "
                    f"(got '{self.optimizer_task_strategy}')."
                )

        if self.validation_gate_enabled:
            try:
                self._validation_metric_cls = Metric.get_subclass(
                    self.validation_metric
                )
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"TextGrad: unknown validation_metric={self.validation_metric!r}. "
                    f"Register it as a Metric subclass in metrics.py, or use one of "
                    f"the built-in metrics (e.g. 'accuracy', 'f1', 'precision', 'recall', 'lce')."
                ) from e

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        loss_functions: Dict[str, Dict[str, Any]] = {}
        for task in self.tasks:
            loss_fn_config: Dict[str, Any] = {"use_textual": True}
            if task.task_name in self.task_losses:
                loss_fn_config["metric"] = self.task_losses[task.task_name]
            loss_functions[task.task_name] = loss_fn_config

        context: Dict[str, Any] = {
            "algorithm": "textgrad",
            "step": step,
            "loss_functions": loss_functions,
            "loss_task_strategy": self.loss_task_strategy,
            "gradient_task_strategy": self.gradient_task_strategy,
            "optimizer_task_strategy": self.optimizer_task_strategy,
        }

        if self._previous_instructions is not None:
            context["previous_instructions"] = self._previous_instructions

        if batch is not None:
            context["batch"] = batch

        return context

    # -- Hooks: inject predictions/ground_truths for backward pass --

    def _before_train(
        self,
        *,
        dataset: Dataset,
        initial_prompt: PromptTemplate,
    ) -> None:
        """Set up the fixed validation batch from the training set."""
        if self.validation_gate_enabled:
            self._setup_validation_batch(dataset=dataset)

    def _after_predict(
        self,
        *,
        step: int,
        predictions: List[PredictionResult],
        batch: Batch,
    ) -> None:
        self._last_predictions = predictions
        self._last_ground_truths = batch.samples

    def _build_gradient_context(self, *, step: int) -> Dict[str, Any]:
        context: Dict[str, Any] = self._get_algorithm_context(step)
        context["predictions"] = self._last_predictions
        context["ground_truths"] = self._last_ground_truths
        return context

    def _build_optimizer_context(self, *, step: int, batch: Batch) -> Dict[str, Any]:
        context: Dict[str, Any] = self._get_algorithm_context(step=step, batch=batch)
        context["predictions"] = self._last_predictions
        context["ground_truths"] = self._last_ground_truths
        return context

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
        self._previous_instructions = new_prompt.instruction
        self._last_predictions = None
        self._last_ground_truths = None

    def _serialize_algorithm_context(self, *, step: int) -> Dict[str, Any]:
        return {
            "algorithm": "textgrad",
            "step": step,
            "loss_task_strategy": self.loss_task_strategy,
            "gradient_task_strategy": self.gradient_task_strategy,
            "optimizer_task_strategy": self.optimizer_task_strategy,
        }

    def _serialize_algorithm_state(self) -> Dict[str, Any]:
        validation_gate_state: Dict[str, Any]
        if self.validation_gate_enabled:
            validation_gate_state = {
                "validation_metric": self.validation_metric,
                "enabled": True,
                "validation_gate_samples": self.validation_gate_samples,
                "current_validation_score": self._current_validation_score,
            }
        else:
            validation_gate_state = {
                "validation_metric": self.validation_metric,
                "enabled": False,
            }
        return {
            "previous_instructions": self._previous_instructions,
            "validation_gate": validation_gate_state,
        }

    # ------------------------------------------------------------------
    # Validation gate: accept prompt updates only if validation improves
    # ------------------------------------------------------------------
    def _setup_validation_batch(
        self,
        *,
        dataset: Dataset,
    ) -> None:
        """Sample a fixed validation batch from the training set once before training.

        The paper (Section 3.3) runs "a validation loop with the validation
        set."  Since PromptMOO datasets only have train/test splits (no
        dedicated validation file), we carve the validation batch from the
        **tail** of the shuffled training data — rows that come after the
        ``train_size`` window the optimization loop samples from.  This
        ensures the validation gate never sees the test set and the
        validation samples are disjoint from training batches in practice
        (the training loop samples ``batch_size`` rows from the head of a
        per-step shuffle, while validation rows come from the tail region).

        The batch is held constant across all steps so the gate signal is
        stable.

        Args:
            dataset: Dataset instance providing ``train()`` and column info.
        """
        train_data: pd.DataFrame = dataset.train()
        shuffled: pd.DataFrame = train_data.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)
        n_validation: int = min(self.validation_gate_samples, len(shuffled))
        validation_rows: pd.DataFrame = shuffled.tail(n_validation).reset_index(
            drop=True
        )

        full_batch: Batch = self._sample_batch(
            data=validation_rows,
            dataset=dataset,
            step=-1,
            full=True,
        )
        self._validation_batch = Batch(step=-1, samples=full_batch.samples)
        self._current_validation_score = None

    def _compute_validation_score(
        self,
        *,
        prompt: PromptTemplate,
        step: int,
    ) -> float:
        """Evaluate a prompt on the validation batch and return a scalar score.

        Uses the task LLM to predict, then computes the validation metric
        programmatically (no LLM-based loss).  The score is direction-normalized
        via ``Metric.normalized_score`` so higher is always better.

        Args:
            prompt: The prompt template to evaluate.
            step: Current optimization step (for algorithm context).

        Returns:
            Direction-normalized scalar score (higher is always better).
        """
        if self._validation_batch is None:
            raise RuntimeError(
                "TextGrad._compute_validation_score called before "
                "_setup_validation_batch.  This should not happen."
            )

        from ..task_predictor import TaskPredictor

        predictor: TaskPredictor = TaskPredictor.of(self.task_predictor["name"])
        context_kwargs: Dict[str, Any] = self._get_algorithm_context(step=step)
        context_kwargs.pop("batch", None)

        predictions: List[PredictionResult] = predictor.predict(
            self._validation_batch,
            prompt,
            self.task_llm,
            verbosity=0,
            **context_kwargs,
        )

        metric_cls: Optional[Type[Metric]] = self._validation_metric_cls
        task_scores: List[float] = []
        for task in self.tasks:
            y_true: List[str] = []
            y_pred: List[str] = []
            for pred, gt_sample in zip(predictions, self._validation_batch.samples):
                if (
                    task.task_name in pred.task_outputs
                    and task.task_name in gt_sample.ground_truths
                ):
                    y_true.append(gt_sample.ground_truths[task.task_name])
                    y_pred.append(pred.task_outputs[task.task_name])
            if len(y_true) == 0:
                continue
            value: float = metric_cls.compute(y_true=y_true, y_pred=y_pred)
            metric_instance: Metric = metric_cls(value=value)
            task_scores.append(metric_instance.normalized_score)

        if len(task_scores) == 0:
            return float("-inf")
        return sum(task_scores) / len(task_scores)

    def _should_accept_prompt_update(
        self,
        *,
        current_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        step: int,
        observer: ObservabilityManager,
    ) -> bool:
        """Validation gate: accept only if the new prompt does not regress.

        Compares the new prompt's validation score against the cached current
        score (or computes the current score on first call).  Records the
        gate decision to the observer for post-hoc analysis.

        Args:
            current_prompt: The prompt before this step's optimization.
            new_prompt: The candidate prompt produced by Step 4.
            step: Current optimization step number.
            observer: ObservabilityManager for recording the gate decision.

        Returns:
            True if the new prompt should be accepted, False to roll back.
        """
        if not self.validation_gate_enabled:
            if self.verbosity >= 2:
                print(
                    "  Validation gate: DISABLED (validation_metric='none'), auto-accepting."
                )
            observer.record(
                key="validation_gate",
                value={
                    "validation_metric": self.validation_metric,
                    "skipped": True,
                    "accepted": True,
                },
            )
            return True

        if self._current_validation_score is None:
            self._current_validation_score = self._compute_validation_score(
                prompt=current_prompt,
                step=step,
            )

        new_score: float = self._compute_validation_score(
            prompt=new_prompt,
            step=step,
        )

        accepted: bool = new_score >= self._current_validation_score
        if self.verbosity >= 2:
            verdict: str = "ACCEPTED" if accepted else "REJECTED (rollback)"
            direction: str = self._validation_metric_cls.optimization_direction
            print(
                f"  Validation gate ({self.validation_metric}, {direction}): "
                f"current={self._current_validation_score:.4f}, "
                f"new={new_score:.4f} -> {verdict}"
            )

        observer.record(
            key="validation_gate",
            value={
                "validation_metric": self.validation_metric,
                "validation_gate_samples": self.validation_gate_samples,
                "current_score": self._current_validation_score,
                "new_score": new_score,
                "accepted": accepted,
            },
        )

        if accepted:
            self._current_validation_score = new_score

        return accepted


class TextGradOptimizer(PromptOptimizer):
    """TextGrad-Specific Prompt Optimizer (Yuksekgonul et al., 2024).

    Implements the TGD.step operator from the paper.  The optimizer sees:

    - ``<ROLE>``: the role of the variable being improved
    - ``<VARIABLE>``: the current instruction text for each task
    - ``<CONTEXT>``: concrete forward-pass examples (``<LM_SYSTEM_PROMPT>``,
      ``<LM_INPUT>``, ``<LM_OUTPUT>``) so the optimizer understands how the
      prompt is actually used
    - ``<FEEDBACK>``: the concatenated per-instance textual gradients

    Per-instance gradients are newline-concatenated (the textual analog of
    ``tg.sum``), matching the paper: "gradients propagating from multiple
    instances are concatenated together, thus the optimizer sees all of the
    feedback."

    Multi-task strategy (``optimizer_task_strategy``):
        ``"combine_all_tasks"`` (default): 1 optimizer call producing a dict
            of all task instructions simultaneously.
        ``"separate_tasks"``: K independent optimizer calls, each updating
            one task's instruction.  Each call sees only that task's
            gradients and produces a single instruction string.
    """

    aliases: ClassVar[List[str]] = ["textgrad"]

    @validate
    def optimize(
        self,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        verbosity: int = 1,
        *,
        trajectory: Optional[PromptTrajectory] = None,
        batch: Optional[Batch] = None,
        task_demonstrations: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        optimizer_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> OptimizerResult:
        """Override to support separate_tasks mode with K independent calls."""
        if optimizer_task_strategy is None:
            raise ValueError(
                "TextGradOptimizer.optimize: optimizer_task_strategy must be "
                "provided explicitly. Pass it from TextGrad._get_algorithm_context."
            )
        if optimizer_task_strategy not in (SEPARATE_TASKS, COMBINE_ALL_TASKS):
            raise ValueError(
                f"TextGradOptimizer: unknown optimizer_task_strategy={optimizer_task_strategy!r}. "
                f"Must be {SEPARATE_TASKS!r} or {COMBINE_ALL_TASKS!r}."
            )

        if optimizer_task_strategy == SEPARATE_TASKS:
            return self._optimize_separate(
                gradients=gradients,
                current_prompt=current_prompt,
                tasks=tasks,
                llm_pool=llm_pool,
                verbosity=verbosity,
                predictions=predictions,
                ground_truths=ground_truths,
                input_col_labels=input_col_labels,
                **kwargs,
            )
        return super().optimize(
            gradients=gradients,
            current_prompt=current_prompt,
            tasks=tasks,
            llm_pool=llm_pool,
            verbosity=verbosity,
            trajectory=trajectory,
            batch=batch,
            task_demonstrations=task_demonstrations,
            input_col_labels=input_col_labels,
            predictions=predictions,
            ground_truths=ground_truths,
            optimizer_task_strategy=optimizer_task_strategy,
            **kwargs,
        )

    def _optimize_separate(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        verbosity: int,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> OptimizerResult:
        """Separate-tasks mode: K independent optimizer calls.

        Each call receives only one task's gradients and the <CONTEXT> with
        that task's forward-pass data.  Produces one instruction per call.
        Results are merged into a single OptimizerResult with all tasks.
        """
        if predictions is None:
            raise ValueError(
                "TextGradOptimizer separate_tasks mode requires 'predictions' "
                "to build per-task <CONTEXT> blocks."
            )
        if ground_truths is None:
            raise ValueError(
                "TextGradOptimizer separate_tasks mode requires 'ground_truths' "
                "to build per-task <CONTEXT> blocks."
            )
        all_meta_prompts: List[str] = []
        all_new_instructions: Dict[str, str] = {}

        for task in tasks:
            if task not in gradients:
                raise ValueError(
                    f"TextGrad.create_meta_prompt: task {task.task_name!r} not found in "
                    f"gradients dict. Available: {[t.task_name for t in gradients.keys()]}."
                )
            all_meta_prompts.append(
                self._create_single_task_meta_prompt(
                    gradients=gradients[task],
                    task=task,
                    current_prompt=current_prompt,
                    predictions=predictions,
                    ground_truths=ground_truths,
                    input_col_labels=current_prompt.input_col_labels,
                )
            )

        prompts_to_send: List[str] = apply_prompt_suffix(all_meta_prompts, llm_pool)
        responses: List[str] = llm_pool.call_llm_batch(
            prompts=prompts_to_send,
            verbosity=verbosity,
        ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)

        if len(responses) != len(tasks):
            raise ValueError(
                f"TextGradOptimizer separate_tasks: expected {len(tasks)} responses, "
                f"got {len(responses)}"
            )

        for task, response in zip(tasks, responses):
            instruction: str = str(response).strip()
            instruction = strip_smart_quotes(instruction)
            if len(instruction) == 0:
                instruction = task.task_instruction
            all_new_instructions[task.task_name] = instruction

        updated_tasks: List[Task] = []
        for task in tasks:
            updated_tasks.append(
                Task(
                    task_name=task.task_name,
                    task_description=task.task_description,
                    task_instruction=all_new_instructions[task.task_name],
                    gt_col=task.gt_col,
                )
            )

        new_prompt: PromptTemplate = PromptTemplate(
            skeleton=current_prompt.skeleton,
            instruction=all_new_instructions,
            tasks=updated_tasks,
            input_col_labels=current_prompt.input_col_labels,
        )

        combined_meta: str = "\n\n---\n\n".join(all_meta_prompts)
        return OptimizerResult(
            new_prompt=new_prompt,
            meta_prompt=combined_meta,
            raw_response=str(all_new_instructions),
        )

    def _create_single_task_meta_prompt(
        self,
        *,
        gradients: List[TextGradient],
        task: Task,
        current_prompt: PromptTemplate,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        input_col_labels: Dict[str, str],
    ) -> str:
        """Build an optimizer prompt for a single task (separate_tasks mode).

        The prompt targets one task's instruction as the <VARIABLE>, includes
        only that task's gradients in <FEEDBACK>, and shows the full context.
        """
        if task.task_name not in current_prompt.instruction:
            raise ValueError(
                f"TextGrad._create_single_task_meta_prompt: task {task.task_name!r} "
                f"not found in current_prompt.instruction. "
                f"Available: {list(current_prompt.instruction.keys())}."
            )

        gradient_text: str = "\n".join(f"  - {g.gradient_text}" for g in gradients)

        context_block: str = self._build_context_block(
            predictions=predictions,
            ground_truths=ground_truths,
            prompt_template=current_prompt,
            input_col_labels=current_prompt.input_col_labels,
            task_name_filter=task.task_name,
            feedback_text=gradient_text,
        )

        return f"""You are part of an optimization system that improves text (i.e., the variable). You will receive some feedback, and use the feedback to improve the variable. The feedback may be noisy — identify what is important and what is correct. Pay attention to the role description of the variable, and the context in which it is used.

Here is the role of the variable you will improve:
<ROLE>instruction for the '{task.task_name}' task in the system prompt to a language model</ROLE>

The variable is the text within the following span:
<VARIABLE>{current_prompt.instruction[task.task_name]}</VARIABLE>

Here is the context and feedback we got for the variable:

{context_block}

Improve the instruction using the feedback provided in <FEEDBACK> tags.
Output ONLY the improved instruction text for '{task.task_name}'. Do not include JSON, quotes, or any other formatting.
""".strip()

    @validate
    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        trajectory: Optional[PromptTrajectory] = None,
        batch: Optional[Batch] = None,
        task_demonstrations: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        optimizer_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Build the combine_all_tasks meta-prompt (default mode)."""
        resolved_predictions: List[PredictionResult] = (
            predictions if predictions is not None else []
        )
        resolved_ground_truths: List[DatasetSample] = (
            ground_truths if ground_truths is not None else []
        )

        gradient_text: str = self._format_gradients_for_meta_prompt(gradients)

        current_instructions_text: str = ",\n".join(
            [
                f'  "{task.task_name}": "{current_prompt.instruction[task.task_name]}"'
                for task in tasks
            ]
        )

        context_block: str = self._build_context_block(
            predictions=resolved_predictions,
            ground_truths=resolved_ground_truths,
            prompt_template=current_prompt,
            input_col_labels=current_prompt.input_col_labels,
            feedback_text=gradient_text,
        )

        example_instruction: str = ",\n".join(
            [
                f'        "{task.task_name}": "Improved instruction for {task.task_name} based on the feedback provided."'
                for task in tasks
            ]
        )

        meta_prompt: str = f"""You are part of an optimization system that improves text (i.e., the variable). You will receive some feedback, and use the feedback to improve the variable. The feedback may be noisy — identify what is important and what is correct. Pay attention to the role description of the variable, and the context in which it is used.

Here is the role of the variable you will improve:
<ROLE>task instructions in the system prompt to a language model</ROLE>

The variable is the text within the following span:
<VARIABLE>
{{
{current_instructions_text}
}}
</VARIABLE>

Here is the context and feedback we got for the variable:

{context_block}

Improve the instructions using the feedback provided in <FEEDBACK> tags.

Generate exactly ONE improved instruction set **as a single JSON dictionary only**.
Do not include any text outside the JSON.

The JSON structure should be:
{{
  "instructions": {{
{example_instruction}
  }}
}}""".strip()

        return meta_prompt

    @validate
    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Any,
    ) -> Dict[str, str]:
        task_names: List[str] = [t.task_name for t in tasks]
        result: Dict[str, str] = extract_instructions_json(
            response, task_names=task_names
        )
        for task in tasks:
            if task.task_name not in result:
                result[task.task_name] = task.task_instruction
        return result

    @staticmethod
    def _format_gradients_for_meta_prompt(
        gradients: Dict[Task, List[TextGradient]],
    ) -> str:
        """Concatenate per-instance gradients with newlines (tg.sum analog).

        Per the paper (Appendix, Batch Optimization): "gradients propagating
        from multiple instances are concatenated together."  Task groups are
        separated by ``---`` dividers for readability in the multi-task case.
        """
        formatted: List[str] = []
        for task, grad_list in gradients.items():
            task_header: str = f"Feedback for task '{task.task_name}':"
            per_instance: List[str] = [f"  - {g.gradient_text}" for g in grad_list]
            formatted.append(task_header + "\n" + "\n".join(per_instance))
        return "\n\n---\n\n".join(formatted)

    @staticmethod
    def _build_context_block(
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        prompt_template: PromptTemplate,
        input_col_labels: Dict[str, str],
        task_name_filter: Optional[str] = None,
        feedback_text: Optional[str] = None,
    ) -> str:
        """Build a ``<CONTEXT>`` block with conversations and feedback.

        Matches the paper's TGD optimizer prompt structure (Appendix,
        textgrad_operations.tex lines 200-250): ``<CONTEXT>`` wraps both
        the forward-pass conversations AND the ``<FEEDBACK>`` block.
        The system prompt is shown once before the conversations (token-
        efficient adaptation; the paper repeats it per-conversation).

        When ``task_name_filter`` is set (separate_tasks mode), both
        ``<LM_SYSTEM_PROMPT>`` and ``<LM_OUTPUT>`` are filtered to show
        only the target task.  This prevents cross-task leakage: in
        separate_tasks mode, the optimizer LLM for fluency must not see
        consistency's output format, instruction, or prediction.

        When ``task_name_filter`` is ``None`` (combine_all_tasks mode),
        ``<LM_SYSTEM_PROMPT>`` shows the full prompt and ``<LM_OUTPUT>``
        shows the task LLM's ``raw_response`` (full JSON).

        Args:
            predictions: PredictionResult objects to include.
            ground_truths: DatasetSample objects to include.
            prompt_template: Current prompt template (for ``<LM_SYSTEM_PROMPT>``).
            input_col_labels: Column name -> display label mapping.
            task_name_filter: When provided, filters ``<LM_SYSTEM_PROMPT>``
                (via ``render_instructions(task_filter=...)``) and ``<LM_OUTPUT>``
                (shows only that task's prediction from ``task_outputs``) to the
                target task.  Used in ``separate_tasks`` mode.
                When ``None``, ``<LM_SYSTEM_PROMPT>`` shows all tasks and
                ``<LM_OUTPUT>`` shows ``raw_response``.
            feedback_text: Pre-formatted gradient text (``tg.sum`` concatenation).
                Embedded inside ``<FEEDBACK>`` tags within ``<CONTEXT>``.
                When ``None``, no ``<FEEDBACK>`` block is added.
        """
        if len(predictions) == 0 or len(ground_truths) == 0:
            return ""

        role_description: str = (
            f"the '{task_name_filter}' instruction"
            if task_name_filter is not None
            else "the task instructions"
        )

        lines: List[str] = ["<CONTEXT>", ""]

        lines.append(
            f"<LM_SYSTEM_PROMPT>\n{prompt_template.render_instructions(task_filter=task_name_filter)}\n</LM_SYSTEM_PROMPT>"
        )
        lines.append("")

        num_examples: int = min(len(predictions), len(ground_truths))
        for idx in range(num_examples):
            pred: PredictionResult = predictions[idx]
            gt: DatasetSample = ground_truths[idx]

            lines.append("<CONVERSATION>")
            lines.append("<LM_INPUT>")
            for col, val in gt.inputs.items():
                label: str = input_col_labels[col]
                text: str = str(val)
                lines.append(f"{label}: {text}")
            lines.append("</LM_INPUT>")
            lines.append("")

            lines.append("<LM_OUTPUT>")
            if task_name_filter is not None:
                if task_name_filter not in pred.task_outputs:
                    raise ValueError(
                        f"TextGradOptimizer._build_context_block: "
                        f"task_name_filter={task_name_filter!r} not found in "
                        f"pred.task_outputs. "
                        f"Available: {list(pred.task_outputs.keys())}."
                    )
                filtered_val: str = str(pred.task_outputs[task_name_filter])
                lines.append(f"{task_name_filter}: {filtered_val}")
            else:
                lines.append(pred.raw_response)
            lines.append("</LM_OUTPUT>")
            lines.append("</CONVERSATION>")
            lines.append("")

        if feedback_text is not None:
            lines.append(
                f"Here is the feedback we got for {role_description} in the conversations:"
            )
            lines.append("")
            lines.append("<FEEDBACK>")
            lines.append(feedback_text)
            lines.append("</FEEDBACK>")
            lines.append("")

        lines.append("</CONTEXT>")

        return "\n".join(lines)


class TextGradGradientComputer(GradientComputer):
    """TextGrad-specific Gradient Computer (Yuksekgonul et al., 2024).

    Implements the backward operator ``∇_LLM(variable, successor, ∂L/∂successor)``
    from the paper.  For each per-instance feedback, the gradient LLM sees:

    - ``<LM_SYSTEM_PROMPT>``: the current prompt template (the *variable*)
    - ``<LM_INPUT>``:  the input data for this instance
    - ``<LM_OUTPUT>``: the task LLM's prediction (the *successor*)
    - ``<OBJECTIVE_FUNCTION>``: the loss LLM's criticism (∂L/∂successor)
    - ``<VARIABLE>``: the specific instruction text to give feedback on

    Must be used with ``gradient_batch_size=1`` so each instance gets its own
    gradient computation, matching the paper's per-instance backward pass that
    are concatenated via ``tg.sum``.

    Multi-task strategy (``gradient_task_strategy``):
        ``"separate_tasks"`` (default): K gradient prompts per instance, each
            focused on one task.  ``<LM_OUTPUT>`` shows only that task's
            prediction; ``<VARIABLE>`` shows only that task's instruction.
        ``"combine_all_tasks"``: 1 gradient prompt per instance covering all
            tasks.  ``<LM_OUTPUT>`` shows the full model output (all tasks);
            ``<VARIABLE>`` shows all task instructions.  Produces a single
            combined gradient assigned to every task.
    """

    aliases = ["textgrad"]

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
        if predictions is None:
            raise ValueError(
                "TextGradGradientComputer requires 'predictions' to build "
                "backward-pass prompts with <LM_INPUT> and <LM_OUTPUT> context."
            )
        if ground_truths is None:
            raise ValueError(
                "TextGradGradientComputer requires 'ground_truths' to build "
                "backward-pass prompts with <LM_INPUT> context."
            )

        pred_by_sample: Dict[str, PredictionResult] = {}
        for p in predictions:
            pred_by_sample[p.sample_id] = p
        gt_by_sample: Dict[str, DatasetSample] = {}
        for g in ground_truths:
            gt_by_sample[g.sample_id] = g

        if gradient_task_strategy is None:
            raise ValueError(
                "TextGradGradientComputer.compute: gradient_task_strategy must be "
                "provided explicitly. Pass it from TextGrad._get_algorithm_context."
            )
        if gradient_task_strategy not in (SEPARATE_TASKS, COMBINE_ALL_TASKS):
            raise ValueError(
                f"TextGradGradientComputer: unknown gradient_task_strategy={gradient_task_strategy!r}. "
                f"Must be {SEPARATE_TASKS!r} or {COMBINE_ALL_TASKS!r}."
            )

        if gradient_task_strategy == COMBINE_ALL_TASKS:
            return self._compute_combined(
                feedbacks=feedbacks,
                prompt_template=prompt_template,
                tasks=tasks,
                llm_pool=llm_pool,
                gradient_batch_size=gradient_batch_size,
                pred_by_sample=pred_by_sample,
                gt_by_sample=gt_by_sample,
                input_col_labels=prompt_template.input_col_labels,
                verbosity=verbosity,
            )
        return self._compute_separate(
            feedbacks=feedbacks,
            prompt_template=prompt_template,
            tasks=tasks,
            llm_pool=llm_pool,
            gradient_batch_size=gradient_batch_size,
            pred_by_sample=pred_by_sample,
            gt_by_sample=gt_by_sample,
            input_col_labels=prompt_template.input_col_labels,
            verbosity=verbosity,
        )

    def _compute_separate(
        self,
        *,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        gradient_batch_size: int,
        pred_by_sample: Dict[str, PredictionResult],
        gt_by_sample: Dict[str, DatasetSample],
        input_col_labels: Dict[str, str],
        verbosity: int,
    ) -> Dict[Task, List[TextGradient]]:
        """Separate-tasks mode: K gradient prompts per instance."""
        result: Dict[Task, List[TextGradient]] = {}

        for task, feedback_list in feedbacks.items():
            if len(feedback_list) == 0:
                result[task] = []
                continue

            feedback_batches: List[List[Union[NumericFeedback, TextualFeedback]]] = (
                self._batch_feedbacks(
                    feedbacks=feedback_list,
                    batch_size=gradient_batch_size,
                )
            )

            prompts: List[str] = []
            for fb_batch in feedback_batches:
                grad_prompt: str = self._build_textgrad_gradient_prompt(
                    feedbacks=fb_batch,
                    task=task,
                    prompt_template=prompt_template,
                    pred_by_sample=pred_by_sample,
                    gt_by_sample=gt_by_sample,
                    input_col_labels=input_col_labels,
                )
                prompts.append(grad_prompt)

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
                            if len(fb.aggregated_from_samples) > 0:
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
                        f"Failed to compute gradient for {task.task_name}:\n"
                        f"{format_exception_msg(e)}"
                    ) from e

            result[task] = gradients

        return result

    def _compute_combined(
        self,
        *,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,  # LLMPool protocol; see types.py
        gradient_batch_size: int,
        pred_by_sample: Dict[str, PredictionResult],
        gt_by_sample: Dict[str, DatasetSample],
        input_col_labels: Dict[str, str],
        verbosity: int,
    ) -> Dict[Task, List[TextGradient]]:
        """Combine-all-tasks mode: 1 gradient prompt per sample for all tasks.

        Groups TextualFeedback objects by sample_id so that each combined
        gradient prompt receives every task's loss feedback for one instance.
        This ensures the ``<OBJECTIVE_FUNCTION>`` section contains all tasks'
        criticisms (not just one task's), matching the paper's design where
        the backward operator sees the full downstream loss signal.

        With ``loss_batch_size=1`` (enforced by TextGrad), each TextualFeedback
        has exactly one sample_id.  When loss is ``separate_tasks``, there are
        K feedbacks per sample (one per task).  When loss is ``combine_all_tasks``,
        there is 1 feedback per sample (already combined).  Both cases are
        handled: we group by sample_id and pass all feedbacks for that sample
        to the combined gradient prompt builder.
        """
        sample_to_feedbacks: Dict[str, List[TextualFeedback]] = {}
        for task, feedback_list in feedbacks.items():
            for fb in feedback_list:
                if not isinstance(fb, TextualFeedback):
                    continue
                for sid in fb.aggregated_from_samples:
                    if sid not in sample_to_feedbacks:
                        sample_to_feedbacks[sid] = []
                    sample_to_feedbacks[sid].append(fb)

        if len(sample_to_feedbacks) == 0:
            return {task: [] for task in tasks}

        sample_groups: List[List[TextualFeedback]] = list(sample_to_feedbacks.values())

        prompts: List[str] = []
        for fb_group in sample_groups:
            grad_prompt: str = self._build_textgrad_combined_gradient_prompt(
                feedbacks=fb_group,
                tasks=tasks,
                prompt_template=prompt_template,
                pred_by_sample=pred_by_sample,
                gt_by_sample=gt_by_sample,
                input_col_labels=input_col_labels,
            )
            prompts.append(grad_prompt)

        combined_gradients: List[TextGradient] = []
        if len(prompts) > 0:
            try:
                prompts = apply_prompt_suffix(prompts, llm_pool)
                responses: List[str] = llm_pool.call_llm_batch(
                    prompts=prompts, verbosity=verbosity
                ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)

                for fb_group, prompt, response in zip(
                    sample_groups, prompts, responses
                ):
                    feedback_ids: List[str] = []
                    for fb in fb_group:
                        feedback_ids.extend(fb.aggregated_from_samples)

                    gradient: TextGradient = TextGradient(
                        task_name="_combined",
                        gradient_text=response,
                        based_on_feedbacks=feedback_ids,
                        gradient_prompt=prompt,
                    )
                    combined_gradients.append(gradient)
            except (RuntimeError, TimeoutError, ValueError) as e:
                raise RuntimeError(
                    f"Failed to compute combined gradient:\n{format_exception_msg(e)}"
                ) from e

        result: Dict[Task, List[TextGradient]] = {}
        for task in tasks:
            result[task] = [
                TextGradient(
                    task_name=task.task_name,
                    gradient_text=g.gradient_text,
                    based_on_feedbacks=g.based_on_feedbacks,
                    gradient_prompt=g.gradient_prompt,
                )
                for g in combined_gradients
            ]

        return result

    @staticmethod
    def _build_textgrad_gradient_prompt(
        *,
        feedbacks: List[Union[NumericFeedback, TextualFeedback]],
        task: Task,
        prompt_template: PromptTemplate,
        pred_by_sample: Dict[str, PredictionResult],
        gt_by_sample: Dict[str, DatasetSample],
        input_col_labels: Dict[str, str],
    ) -> str:
        """Build a per-task backward-pass prompt (separate_tasks mode).

        Both ``<LM_SYSTEM_PROMPT>`` and ``<LM_OUTPUT>`` are filtered to the
        single task so the gradient LLM sees a consistent, single-task view
        of the forward pass.  ``<VARIABLE>`` shows only this task's instruction.
        """
        current_instruction: str = prompt_template.instruction[task.task_name]

        prompt: str = BACKWARD_SYSTEM_PROMPT + "\n\n"

        prompt += (
            "You will give feedback to a variable with the following role: "
            f"<ROLE> instruction for the '{task.task_name}' task in the system prompt "
            f"to a language model </ROLE>. "
            "Here is a conversation with a language model (LM):\n\n"
        )

        for fb in feedbacks:
            if not isinstance(fb, TextualFeedback):
                continue

            sample_ids: List[str] = fb.aggregated_from_samples
            for sid in sample_ids:
                if sid not in pred_by_sample:
                    raise ValueError(
                        f"TextGradGradientComputer: sample_id {sid!r} not found in "
                        f"pred_by_sample. Available: {list(pred_by_sample.keys())}."
                    )
                if sid not in gt_by_sample:
                    raise ValueError(
                        f"TextGradGradientComputer: sample_id {sid!r} not found in "
                        f"gt_by_sample. Available: {list(gt_by_sample.keys())}."
                    )
                pred_obj: PredictionResult = pred_by_sample[sid]
                gt_obj: DatasetSample = gt_by_sample[sid]

                prompt += f"<LM_SYSTEM_PROMPT>\n{prompt_template.render_instructions(task_filter=task.task_name)}\n</LM_SYSTEM_PROMPT>\n\n"

                prompt += "<LM_INPUT>\n"
                for col, val in gt_obj.inputs.items():
                    label: str = input_col_labels[col]
                    text: str = str(val)
                    prompt += f"{label}: {text}\n"
                prompt += "</LM_INPUT>\n\n"

                pred_val: str = str(pred_obj.task_outputs[task.task_name])
                prompt += f"<LM_OUTPUT>\n{task.task_name}: {pred_val}\n</LM_OUTPUT>\n\n"

            prompt += (
                "This conversation is part of a larger system. "
                "The <LM_OUTPUT> was later used as prediction from the language model.\n\n"
                "<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable "
                "to address the following feedback on the LM_OUTPUT: "
                f"{fb.feedback_text} </OBJECTIVE_FUNCTION>\n\n"
            )

        prompt += (
            f"We are interested in giving feedback to the instruction for the "
            f"'{task.task_name}' task in the system prompt to a language model "
            f"for this conversation. Specifically, give feedback to the following "
            f"span of text:\n\n"
            f"<VARIABLE> {current_instruction} </VARIABLE>\n\n"
            f"Given the above history, describe how the instruction for the "
            f"'{task.task_name}' task in the system prompt to a language model "
            f"could be improved to improve the <OBJECTIVE_FUNCTION>. "
            f"Write exactly 3 paragraphs:\n"
            f"Paragraph 1: What is working well in the current instruction and should be preserved.\n"
            f"Paragraph 2: What specific problems in the instruction led to incorrect predictions.\n"
            f"Paragraph 3: Concrete strategies to improve the instruction while preserving what works."
        )

        return prompt

    @staticmethod
    def _build_textgrad_combined_gradient_prompt(
        *,
        feedbacks: List[Union[NumericFeedback, TextualFeedback]],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        pred_by_sample: Dict[str, PredictionResult],
        gt_by_sample: Dict[str, DatasetSample],
        input_col_labels: Dict[str, str],
    ) -> str:
        """Build a combined all-tasks backward-pass prompt (combine_all_tasks mode).

        ``<LM_OUTPUT>`` shows the full model output (all tasks' predictions).
        ``<VARIABLE>`` shows all task instructions.
        ``<OBJECTIVE_FUNCTION>`` contains feedback from all tasks.
        """
        all_instructions: str = "\n".join(
            f"  {tn}: {instr}" for tn, instr in prompt_template.instruction.items()
        )

        task_names_str: str = ", ".join(t.task_name for t in tasks)

        prompt: str = BACKWARD_SYSTEM_PROMPT + "\n\n"

        prompt += (
            "You will give feedback to variables with the following role: "
            f"<ROLE> instructions for tasks ({task_names_str}) in the system prompt "
            f"to a language model </ROLE>. "
            "Here is a conversation with a language model (LM):\n\n"
        )

        seen_samples: Set[str] = set()
        for fb in feedbacks:
            if not isinstance(fb, TextualFeedback):
                continue

            for sid in fb.aggregated_from_samples:
                if sid in seen_samples:
                    continue
                seen_samples.add(sid)

                pred_obj: PredictionResult = pred_by_sample[sid]
                gt_obj: DatasetSample = gt_by_sample[sid]

                prompt += f"<LM_SYSTEM_PROMPT>\n{prompt_template.render_instructions()}\n</LM_SYSTEM_PROMPT>\n\n"

                prompt += "<LM_INPUT>\n"
                for col, val in gt_obj.inputs.items():
                    label: str = input_col_labels[col]
                    text: str = str(val)
                    prompt += f"{label}: {text}\n"
                prompt += "</LM_INPUT>\n\n"

                prompt += f"<LM_OUTPUT>\n{pred_obj.raw_response}\n</LM_OUTPUT>\n\n"

        prompt += (
            "This conversation is part of a larger system. "
            "The <LM_OUTPUT> was later used as predictions from the language model.\n\n"
        )

        seen_feedback_texts: Set[str] = set()
        feedback_parts: List[str] = []
        for fb in feedbacks:
            if not isinstance(fb, TextualFeedback):
                continue
            if fb.feedback_text in seen_feedback_texts:
                continue
            seen_feedback_texts.add(fb.feedback_text)
            feedback_parts.append(f"Feedback for {fb.task_name}:\n{fb.feedback_text}")

        if len(feedback_parts) > 0:
            concatenated_feedback: str = "\n\n".join(feedback_parts)
            prompt += (
                f"<OBJECTIVE_FUNCTION>Your goal is to "
                f"give feedback to the variable to address the following "
                f"feedback on the LM_OUTPUT:\n\n"
                f"{concatenated_feedback} "
                f"</OBJECTIVE_FUNCTION>\n\n"
            )

        prompt += (
            f"We are interested in giving feedback to the instructions for tasks "
            f"({task_names_str}) in the system prompt to a language model "
            f"for this conversation. Specifically, give feedback to the following:\n\n"
            f"<VARIABLE>\n{all_instructions}\n</VARIABLE>\n\n"
            f"Given the above history, describe how the instructions for tasks "
            f"({task_names_str}) in the system prompt to a language model "
            f"could be improved to improve the <OBJECTIVE_FUNCTION>. "
            f"Write exactly 3 paragraphs:\n"
            f"Paragraph 1: What is working well in the current instructions and should be preserved.\n"
            f"Paragraph 2: What specific problems in the instructions led to incorrect predictions.\n"
            f"Paragraph 3: Concrete strategies to improve the instructions while preserving what works."
        )

        return prompt


class TextGradLossComputer(LossComputer):
    """TextGrad-specific Loss Computer (Yuksekgonul et al., 2024).

    Produces per-instance textual feedback (the ``OBJECTIVE_FUNCTION`` in the
    paper).  Each loss prompt includes the input data, the task LLM's
    prediction, and the ground truth so the loss LLM can produce contextualized
    criticism rather than a bare "predicted X, expected Y" comparison.

    Must be used with ``loss_batch_size=1`` (enforced by the TextGrad
    algorithm class) so that each instance gets its own loss evaluation,
    matching the paper's ``tg.sum`` over per-instance losses.

    Multi-task strategy (``loss_task_strategy`` in algorithm context):
        ``"separate_tasks"`` (default): K independent loss calls per instance,
            one per task.  Each evaluator sees only that task's prediction and
            ground truth.  Returns ``Dict[Task, List[TextualFeedback]]``.
        ``"combine_all_tasks"``: 1 loss call per instance covering all K tasks.
            The evaluator sees all tasks' predictions and ground truths.
            Returns the combined feedback under a synthetic ``_all_tasks`` key
            AND copies it to each individual task key so downstream components
            that iterate per-task still work.
    """

    aliases = ["textgrad"]

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
        if llm_pool is None:
            raise ValueError(
                "TextGrad requires an LLM pool for textual feedback generation"
            )

        if loss_functions is None:
            raise ValueError(
                f"{self.__class__.__name__}.compute() requires 'loss_functions'"
            )

        if loss_task_strategy is None:
            raise ValueError(
                "TextGradLossComputer.compute: loss_task_strategy must be "
                "provided explicitly. Pass it from TextGrad._get_algorithm_context."
            )
        if loss_task_strategy not in (SEPARATE_TASKS, COMBINE_ALL_TASKS):
            raise ValueError(
                f"TextGradLossComputer: unknown loss_task_strategy={loss_task_strategy!r}. "
                f"Must be {SEPARATE_TASKS!r} or {COMBINE_ALL_TASKS!r}."
            )

        if loss_task_strategy == COMBINE_ALL_TASKS:
            return self._compute_combined(
                predictions=predictions,
                ground_truths=ground_truths,
                tasks=tasks,
                prompt_template=prompt_template,
                llm_pool=llm_pool,
                loss_batch_size=loss_batch_size,
                loss_functions=loss_functions,
                input_col_labels=prompt_template.input_col_labels,
                verbosity=verbosity,
            )
        return self._compute_separate(
            predictions=predictions,
            ground_truths=ground_truths,
            tasks=tasks,
            prompt_template=prompt_template,
            llm_pool=llm_pool,
            loss_batch_size=loss_batch_size,
            loss_functions=loss_functions,
            input_col_labels=prompt_template.input_col_labels,
            verbosity=verbosity,
        )

    def _compute_separate(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        llm_pool: Any,  # LLMPool protocol; see types.py
        loss_batch_size: int,
        loss_functions: Dict[str, Any],
        input_col_labels: Dict[str, str],
        verbosity: int,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Separate-tasks mode: K independent loss calls per instance."""
        result: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = {}

        for task in tasks:
            task_name: str = task.task_name
            if task_name not in loss_functions:
                if verbosity >= 2:
                    print(f"Task {task_name} not found in loss_functions")
                continue

            task_batches: List = LossComputer.batch_predictions(
                predictions=predictions,
                ground_truths=ground_truths,
                task=task,
                batch_size=loss_batch_size,
            )

            feedbacks: List[Union[NumericFeedback, TextualFeedback]] = []

            loss_fn_config: Dict[str, Any] = loss_functions[task_name]
            for pred_batch, gt_batch in task_batches:
                numeric: Optional[NumericFeedback] = self._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    feedbacks.append(numeric)

            prompts: List[str] = []
            for pred_batch, gt_batch in task_batches:
                feedback_prompt: str = self._build_textgrad_feedback_prompt(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    input_col_labels=input_col_labels,
                )
                prompts.append(feedback_prompt)

            if len(prompts) > 0:
                try:
                    prompts = apply_prompt_suffix(prompts, llm_pool)
                    responses: List[str] = llm_pool.call_llm_batch(
                        prompts=prompts, verbosity=verbosity
                    ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)

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
                        f"TextGradLossComputer: textual feedback LLM call "
                        f"failed for task {task.task_name}:\n{format_exception_msg(e)}"
                    ) from e

            result[task] = feedbacks

        return result

    def _compute_combined(
        self,
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        prompt_template: PromptTemplate,
        llm_pool: Any,  # LLMPool protocol; see types.py
        loss_batch_size: int,
        loss_functions: Dict[str, Any],
        input_col_labels: Dict[str, str],
        verbosity: int,
    ) -> Dict[Task, List[Union[NumericFeedback, TextualFeedback]]]:
        """Combine-all-tasks mode: 1 loss call per instance covering all tasks."""
        active_tasks: List[Task] = [t for t in tasks if t.task_name in loss_functions]
        if len(active_tasks) == 0:
            return {}

        task_batches: List = LossComputer.batch_predictions(
            predictions=predictions,
            ground_truths=ground_truths,
            task=active_tasks[0],
            batch_size=loss_batch_size,
        )

        prompts: List[str] = []
        for pred_batch, gt_batch in task_batches:
            feedback_prompt: str = self._build_textgrad_combined_feedback_prompt(
                predictions=pred_batch,
                ground_truths=gt_batch,
                tasks=active_tasks,
                input_col_labels=input_col_labels,
            )
            prompts.append(feedback_prompt)

        combined_feedbacks: List[TextualFeedback] = []
        if len(prompts) > 0:
            try:
                prompts = apply_prompt_suffix(prompts, llm_pool)
                responses: List[str] = llm_pool.call_llm_batch(
                    prompts=prompts, verbosity=verbosity
                ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)

                for (pred_batch, _gt_batch), prompt, response in zip(
                    task_batches, prompts, responses
                ):
                    sample_ids: List[str] = [p.sample_id for p in pred_batch]
                    textual: TextualFeedback = TextualFeedback(
                        task_name="_combined",
                        feedback_text=response,
                        aggregated_from_samples=sample_ids,
                        feedback_prompt=prompt,
                    )
                    combined_feedbacks.append(textual)
            except (RuntimeError, TimeoutError, ValueError) as e:
                raise RuntimeError(
                    f"TextGradLossComputer: combined textual feedback LLM call "
                    f"failed:\n{format_exception_msg(e)}"
                ) from e

        result: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = {}
        for task in active_tasks:
            task_numeric: List[Union[NumericFeedback, TextualFeedback]] = []
            loss_fn_config: Dict[str, Any] = loss_functions[task.task_name]
            for pred_batch, gt_batch in task_batches:
                numeric: Optional[NumericFeedback] = self._compute_numeric_loss(
                    predictions=pred_batch,
                    ground_truths=gt_batch,
                    task=task,
                    loss_fn_config=loss_fn_config,
                )
                if numeric is not None:
                    task_numeric.append(numeric)
            task_numeric.extend(
                TextualFeedback(
                    task_name=task.task_name,
                    feedback_text=fb.feedback_text,
                    aggregated_from_samples=fb.aggregated_from_samples,
                    feedback_prompt=fb.feedback_prompt,
                )
                for fb in combined_feedbacks
            )
            result[task] = task_numeric

        return result

    @staticmethod
    def _build_textgrad_feedback_prompt(
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        task: Task,
        input_col_labels: Dict[str, str],
    ) -> str:
        """Build a per-instance, per-task loss backward prompt (separate_tasks mode).

        Follows the paper's ``StringBasedFunction._backward_through_string_fn_base``
        structure: the loss backward receives the prediction, ground truth, and
        deterministic comparison result, then gives feedback to the prediction
        variable about how to improve.

        The loss backward does NOT see:
        - The original input data (summary, source text) — only pred + GT
        - The prompt template or task instruction — avoids leaking the
          optimization variable into the loss signal

        The output of this prompt (the loss LLM's response) flows into the
        gradient prompt's ``<OBJECTIVE_FUNCTION>`` as ``∂L/∂Answer``.
        """
        task_name: str = task.task_name

        prompt: str = BACKWARD_SYSTEM_PROMPT + "\n\n"

        prompt += (
            f"You will give feedback to a variable with the following role: "
            f"<ROLE> prediction from the language model for the task: {task_name} </ROLE>. "
            f"Here is an evaluation of the variable using a comparison function:\n\n"
        )

        for pred, gt in zip(predictions, ground_truths):
            pred_val: str = str(pred.task_outputs[task_name])
            gt_val: str = str(gt.ground_truths[task_name])
            is_correct: bool = pred_val == gt_val
            if is_correct:
                result_str: str = "CORRECT"
            else:
                try:
                    difference: float = abs(float(pred_val) - float(gt_val))
                    result_str = f"INCORRECT (difference: {difference:.2f})"
                except (ValueError, TypeError):
                    result_str = "INCORRECT"

            prompt += (
                f"Function purpose: Check if the model's prediction matches "
                f"the ground truth for {task_name}.\n\n"
                f"<INPUTS_TO_FUNCTION>\n"
                f"**Model prediction ({task_name})**: {pred_val}\n"
                f"**Ground truth ({task_name})**: {gt_val}\n"
                f"</INPUTS_TO_FUNCTION>\n\n"
                f"<OUTPUT_OF_FUNCTION> {result_str} </OUTPUT_OF_FUNCTION>\n\n"
            )

        prompt += (
            "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism "
            "to the variable given the above evaluation output. "
            "Our only goal is to improve the above metric, and nothing else. "
            "</OBJECTIVE_FUNCTION>\n\n"
        )

        pred_values_str: str = str(predictions[0].task_outputs[task_name])
        prompt += (
            f"We are interested in giving feedback to the prediction from the "
            f"language model for the task: {task_name} for this conversation. "
            f"Specifically, give feedback to the following span of text:\n\n"
            f"<VARIABLE> {pred_values_str} </VARIABLE>\n\n"
            f"Given the above evaluation, describe how the prediction from the "
            f"language model for the task: {task_name} could be improved to "
            f"improve the <OBJECTIVE_FUNCTION>. "
            f"If the prediction is correct, describe why it is correct. "
            f"If incorrect, describe specifically what went wrong and what the "
            f"correct prediction should look like. Be concise."
        )

        return prompt

    @staticmethod
    def _build_textgrad_combined_feedback_prompt(
        *,
        predictions: List[PredictionResult],
        ground_truths: List[DatasetSample],
        tasks: List[Task],
        input_col_labels: Dict[str, str],
    ) -> str:
        """Build a per-instance, all-tasks loss backward prompt (combine_all_tasks mode).

        Same ``StringBasedFunction`` backward structure as the single-task
        variant, but bundles all tasks' predictions and ground truths into
        one ``<INPUTS_TO_FUNCTION>`` block and one ``<OUTPUT_OF_FUNCTION>``
        block with per-task CORRECT/INCORRECT results.
        """
        task_names_str: str = ", ".join(t.task_name for t in tasks)

        prompt: str = BACKWARD_SYSTEM_PROMPT + "\n\n"

        prompt += (
            f"You will give feedback to a variable with the following role: "
            f"<ROLE> predictions from the language model for tasks: {task_names_str} </ROLE>. "
            f"Here is an evaluation of the variable using a comparison function:\n\n"
        )

        for pred, gt in zip(predictions, ground_truths):
            prompt += (
                f"Function purpose: Check if the model's predictions match "
                f"the ground truth for each task.\n\n"
            )

            prompt += "<INPUTS_TO_FUNCTION>\n"
            prompt += "**Model predictions**:\n"
            for task in tasks:
                pred_val: str = str(pred.task_outputs[task.task_name])
                prompt += f"  {task.task_name}: {pred_val}\n"
            prompt += "\n**Ground truths**:\n"
            for task in tasks:
                gt_val: str = str(gt.ground_truths[task.task_name])
                prompt += f"  {task.task_name}: {gt_val}\n"
            prompt += "</INPUTS_TO_FUNCTION>\n\n"

            prompt += "<OUTPUT_OF_FUNCTION>\n"
            num_correct: int = 0
            for task in tasks:
                p_val: str = str(pred.task_outputs[task.task_name])
                g_val: str = str(gt.ground_truths[task.task_name])
                is_correct: bool = p_val == g_val
                if is_correct:
                    prompt += f"  {task.task_name}: CORRECT\n"
                    num_correct += 1
                else:
                    try:
                        difference: float = abs(float(p_val) - float(g_val))
                        prompt += f"  {task.task_name}: INCORRECT (difference: {difference:.2f})\n"
                    except (ValueError, TypeError):
                        prompt += f"  {task.task_name}: INCORRECT\n"
            prompt += f"  Overall: {num_correct} of {len(tasks)} correct.\n"
            prompt += "</OUTPUT_OF_FUNCTION>\n\n"

        prompt += (
            "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism "
            "to the variable given the above evaluation output. "
            "Our only goal is to improve the above metric, and nothing else. "
            "</OBJECTIVE_FUNCTION>\n\n"
        )

        pred_values: List[str] = []
        for task in tasks:
            pred_values.append(
                f"  {task.task_name}: {predictions[0].task_outputs[task.task_name]}"
            )
        pred_values_str: str = "\n".join(pred_values)
        prompt += (
            f"We are interested in giving feedback to the predictions from the "
            f"language model for tasks: {task_names_str} for this conversation. "
            f"Specifically, give feedback to the following span of text:\n\n"
            f"<VARIABLE>\n{pred_values_str}\n</VARIABLE>\n\n"
            f"Given the above evaluation, describe how the predictions from the "
            f"language model for tasks: {task_names_str} could be improved to "
            f"improve the <OBJECTIVE_FUNCTION>. "
            f"For each task, if the prediction is correct, describe why it is correct. "
            f"If incorrect, describe specifically what went wrong and what the "
            f"correct prediction should look like. Be concise."
        )

        return prompt
