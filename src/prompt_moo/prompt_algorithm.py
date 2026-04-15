"""
Main Algorithm Implementation: Core prompt optimization loop.

This module implements the base algorithm class and specific algorithm implementations like OPRO.
"""

import os
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from concurry import ProgressBar
from morphic import Registry, Typed, validate
from morphic.typed import format_exception_msg
from pydantic import Field, PrivateAttr, conint

from .config import promptmoo_config
from .data_input import Dataset
from .data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from .gradient_computer import GradientComputer
from .loss_computer import LossComputer
from .observability import ObservabilityManager
from .prompt_optimizer import PromptOptimizer
from .prompt_template import PromptTemplate
from .task_predictor import TaskPredictor


@validate
def should_evaluate_at_step(
    *,
    step: conint(ge=0),
    total_steps: conint(ge=1),
    eval_every: conint(ge=1),
    eval_initial_prompt: bool = True,
    eval_first_step: bool = True,
    eval_last_step: bool = True,
) -> bool:
    """Determine whether to run evaluation at the given step.

    Step 0 is the baseline (initial prompt, before any optimization).
    Steps 1..total_steps are optimization steps.

    The ``eval_every`` parameter controls periodic evaluation: every
    ``eval_every`` steps starting from step ``eval_every``.  The three
    boolean flags control mandatory evaluation at the boundary steps.
    ``eval_every`` overrides the boolean flags: if a step is a multiple
    of ``eval_every``, it is always evaluated regardless of the flags.

    Args:
        step: Current step number (0 = baseline, 1..total_steps = optimization).
            Must be >= 0.  Enforced by ``@validate`` + ``conint(ge=0)``.
        total_steps: Total number of optimization steps (the ``steps`` param).
            Must be >= 1.
        eval_every: Evaluate every N optimization steps.  Must be >= 1.
        eval_initial_prompt: If True, always evaluate at step 0 (baseline).
        eval_first_step: If True, always evaluate at step 1 (first optimized prompt).
        eval_last_step: If True, always evaluate at step ``total_steps`` (final prompt).

    Returns:
        True if evaluation should run at this step.

    Raises:
        ValidationError: If step < 0, total_steps < 1, or eval_every < 1.
    """
    if step == 0:
        return eval_initial_prompt
    if step == 1 and eval_first_step:
        return True
    if step == total_steps and eval_last_step:
        return True
    if step >= 1 and step % eval_every == 0:
        return True
    return False


class PromptAlgorithm(Typed, Registry, ABC):
    """Base class for prompt optimization algorithms.

    This implements the core 4-step optimization loop:
    1. Predict: Generate predictions using task LLM
    2. Compute Losses: Calculate feedback from predictions
    3. Compute Gradients: Generate improvement suggestions
    4. Optimize: Update prompt based on gradients

    Step numbering convention:
        Step 0: Baseline evaluation of the initial prompt (no optimization).
        Steps 1..steps: Optimization steps.  Each step runs predict, loss,
        gradient, optimize, and (optionally) evaluation.
    """

    _allow_subclass_override = True

    # Core components (configured via dicts or Registry keys)
    task_predictor: Dict[str, Any] = Field(default_factory=lambda: {"name": "standard"})
    loss_computer: Dict[str, Any]
    gradient_computer: Dict[str, Any]
    prompt_optimizer: Dict[str, Any]

    # LLM workers (typed as Any because Pydantic's is_instance_of validation
    # rejects the LLMPool Protocol at construction time; duck-typing is
    # enforced at call sites via the LLMPool Protocol in types.py).
    task_llm: Any
    gradient_llm: Optional[Any] = None
    optimizer_llm: Any
    loss_llm: Optional[Any] = None

    # Per-role LLM temperatures.  These are algorithm hyperparameters: changing
    # them produces different experimental results (low temperature = deterministic
    # task predictions; high temperature = creative optimizer suggestions).
    # ``None`` means "defer to promptmoo_config.defaults.*_temperature" at LLM
    # creation time.  Subclasses override with paper-sourced values.
    task_llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    optimizer_llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    gradient_llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    loss_llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    # Training hyperparameters
    steps: int = 10
    batch_size: int = 12
    loss_batch_size: Optional[int] = None
    gradient_batch_size: Optional[int] = None
    eval_every: int = 5
    eval_initial_prompt: bool = True
    eval_first_step: bool = True
    eval_last_step: bool = True
    name: str
    verbosity: int = (
        1  # 0=silent, 1=default (progress bar), 2=detailed, 3=debug (with LLM I/O)
    )
    substep_delay: Optional[float] = None

    # Tasks
    tasks: List[Task]

    # Task demonstrations: (input, ground_truth) examples sampled from the
    # training set and shown to the optimizer LLM so it understands how the
    # prompt is applied.  GPO sets this to 3 following the paper.
    num_task_demonstrations: int = 0

    # Number of candidate prompts to generate per step.  When > 1, each
    # candidate is evaluated on the current batch and the one with the
    # highest direction-aware score is selected as the new prompt.  All
    # candidates (not just the winner) are passed to _update_state so
    # trajectory-based algorithms can add them all to the trajectory.
    # OPRO paper default: 8.  Other algorithms default to 1.
    num_candidates: int = 1

    # When True, the same training batch (sampled once at step 1) is reused
    # across all optimization steps.  This matches the OPRO paper (exp.tex L84):
    # "The same subset is used throughout optimization, so that the task
    # accuracies computed at intermediate optimization steps are approximations
    # of the training accuracy."  Without this, scores in the trajectory are
    # computed on different random subsets and are not comparable.
    # Algorithms that want per-step diversity (e.g., TextGrad, PE2) leave this
    # as False (the default).
    use_fixed_training_batch: bool = False

    # --- Mutable per-run state (set at the start of train()) ---
    _input_col_labels: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _task_demonstrations: Optional[List[DatasetSample]] = PrivateAttr(default=None)
    _task_output_specs: Optional[Dict[str, Any]] = PrivateAttr(
        default=None
    )  # Dict[str, TaskOutputSpec]
    _fixed_batch: Optional[Batch] = PrivateAttr(default=None)

    @validate
    def train(
        self,
        dataset: Dataset,
        initial_prompt: PromptTemplate,
        output_dir: Optional[str] = None,
        start_step: int = 1,
    ) -> Dict[str, Any]:
        """Main training loop. Subclasses customize via hooks, never by overriding.

        Step numbering:
            Step 0: Baseline evaluation of ``initial_prompt`` (no optimization).
            Steps 1..steps: Optimization steps.

        Args:
            dataset: Dataset object with train/test data
            initial_prompt: Initial prompt template
            output_dir: Optional output directory (auto-generated if None)
            start_step: Optimization step to start/resume from (default: 1).
                Step 0 (baseline eval) always runs when ``eval_initial_prompt``
                is True, regardless of ``start_step``.

        Returns:
            Dict with final_prompt, output_dir, and run_id
        """
        run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.loss_batch_size is None:
            raise ValueError(
                f"{self.__class__.__name__}: loss_batch_size is None. "
                f"Either pass it explicitly or use an algorithm subclass "
                f"(GPO, TextGrad) that sets it automatically via pre_initialize."
            )
        if self.gradient_batch_size is None:
            raise ValueError(
                f"{self.__class__.__name__}: gradient_batch_size is None. "
                f"Either pass it explicitly or use an algorithm subclass "
                f"(GPO, TextGrad) that sets it automatically via pre_initialize."
            )

        _substep_delay: Optional[float] = self.substep_delay
        if _substep_delay is None:
            _substep_delay = promptmoo_config.defaults.substep_delay
        if output_dir is None:
            output_dir = (
                f"outputs/{self.class_name}_{dataset.dataset_name}_{self.name}_{run_id}"
            )
        os.makedirs(output_dir, exist_ok=True)
        if self.verbosity >= 1:
            print(f"Output directory: {output_dir}")

        run_config: Dict[str, Any] = self._build_run_config(
            initial_prompt=initial_prompt, start_step=start_step
        )
        observer: ObservabilityManager = ObservabilityManager(
            output_dir=output_dir, verbosity=self.verbosity
        )
        observer.log_config(run_config)

        predictor: TaskPredictor = TaskPredictor.of(self.task_predictor["name"])
        loss_comp: LossComputer = LossComputer.of(self.loss_computer["name"])
        grad_comp: GradientComputer = GradientComputer.of(
            self.gradient_computer["name"]
        )
        optimizer: PromptOptimizer = PromptOptimizer.of(self.prompt_optimizer["name"])

        current_prompt: PromptTemplate = initial_prompt
        train_data: pd.DataFrame = dataset.train()
        self._input_col_labels = dataset.input_col_labels
        self._task_output_specs = {
            task.task_name: dataset.task_output_specs[task.task_name]
            for task in self.tasks
            if task.task_name in dataset.task_output_specs
        }

        self._before_train(dataset=dataset, initial_prompt=initial_prompt)

        # Initialize task demonstrations to empty for the baseline eval.
        # The per-step loop re-samples demonstrations at each step.
        self._task_demonstrations = []

        # --- Step 0: Baseline evaluation (initial prompt, no optimization) ---
        if should_evaluate_at_step(
            step=0,
            total_steps=self.steps,
            eval_every=self.eval_every,
            eval_initial_prompt=self.eval_initial_prompt,
            eval_first_step=self.eval_first_step,
            eval_last_step=self.eval_last_step,
        ):
            if self.verbosity >= 2:
                print("\n===== Step 0 (Baseline Evaluation) =====")
            eval_results_0: Dict[str, Any] = self.evaluate(
                dataset=dataset,
                prompt=current_prompt,
                step=0,
            )
            self._record_evaluation(observer=observer, step=0, results=eval_results_0)
            if self.verbosity >= 2:
                print(
                    f"  Baseline evaluation complete on "
                    f"{len(eval_results_0['prompt_predictions'])} predictions"
                )

        # --- Optimization loop: Steps 1..steps ---
        current_step: int = start_step
        try:
            for step in ProgressBar(
                list(range(start_step, self.steps + 1)),
                desc="Algorithm Progress",
                disable=self.verbosity == 0,
                style="notebook",
            ):
                current_step = step
                step_prefix: str = f"[Step {step}/{self.steps}]"
                if self.verbosity >= 2:
                    print(f"\n===== Step {step}/{self.steps} =====")
                observer.log_step_start(step)

                batch: Batch = self._sample_batch(
                    data=train_data, dataset=dataset, step=step
                )
                self._record_batch(observer=observer, batch=batch)

                self._task_demonstrations = self._sample_task_demonstrations(
                    data=train_data,
                    dataset=dataset,
                    batch=batch,
                    step=step,
                )

                # Step 1: Predict
                if self.verbosity >= 2:
                    print(
                        f"\n{step_prefix} 1/4: Predicting with {len(batch.samples)} samples..."
                    )
                predictions: List[PredictionResult] = predictor.predict(
                    batch,
                    current_prompt,
                    self.task_llm,
                    verbosity=self.verbosity,
                    **self._get_algorithm_context(step),
                )
                self._record_predictions(observer=observer, predictions=predictions)
                if self.verbosity >= 2:
                    print(f"  Generated {len(predictions)} predictions")

                self._after_predict(step=step, predictions=predictions, batch=batch)
                self._record_algorithm_context(observer=observer, step=step)

                if _substep_delay > 0:
                    time.sleep(_substep_delay)

                # Step 2: Compute losses/feedback
                if self.verbosity >= 2:
                    print(f"\n{step_prefix} 2/4: Computing losses...")
                loss_context: Dict[str, Any] = self._build_loss_context(step=step)
                feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]] = (
                    loss_comp.compute(
                        predictions=predictions,
                        ground_truths=batch.samples,
                        tasks=self.tasks,
                        prompt_template=current_prompt,
                        llm_pool=self.loss_llm,
                        loss_batch_size=self.loss_batch_size,
                        verbosity=self.verbosity,
                        **loss_context,
                    )
                )
                self._record_feedbacks(observer=observer, feedbacks=feedbacks)
                if self.verbosity >= 2:
                    print(f"  Computed feedbacks for {len(feedbacks)} tasks")

                if _substep_delay > 0:
                    time.sleep(_substep_delay)

                # Step 3: Compute gradients
                if self.verbosity >= 2:
                    print(f"\n{step_prefix} 3/4: Computing gradients...")
                gradient_context: Dict[str, Any] = self._build_gradient_context(
                    step=step
                )
                gradients: Dict[Task, List[TextGradient]] = grad_comp.compute(
                    feedbacks=feedbacks,
                    prompt_template=current_prompt,
                    tasks=self.tasks,
                    llm_pool=self.gradient_llm,
                    gradient_batch_size=self.gradient_batch_size,
                    verbosity=self.verbosity,
                    **gradient_context,
                )
                self._record_gradients(observer=observer, gradients=gradients)
                if self.verbosity >= 2:
                    print(f"  Computed gradients for {len(gradients)} tasks")

                self._before_optimize(
                    step=step,
                    feedbacks=feedbacks,
                    gradients=gradients,
                    current_prompt=current_prompt,
                )

                if _substep_delay > 0:
                    time.sleep(_substep_delay)

                # Step 4: Optimize prompt (with multi-candidate selection)
                if self.verbosity >= 2:
                    print(f"\n{step_prefix} 4/4: Optimizing prompt...")
                (
                    new_prompt,
                    best_meta_prompt,
                    best_raw_response,
                    candidate_prompts,
                    candidate_scores,
                ) = self._run_optimize(
                    optimizer=optimizer,
                    gradients=gradients,
                    current_prompt=current_prompt,
                    batch=batch,
                    step=step,
                )
                self._record_prompt_update(
                    observer=observer,
                    old_prompt=current_prompt,
                    new_prompt=new_prompt,
                    meta_prompt=best_meta_prompt,
                    optimizer_response=best_raw_response,
                    num_candidates_attempted=max(1, self.num_candidates),
                    num_candidates_succeeded=len(candidate_prompts)
                    if candidate_prompts is not None
                    else 1,
                    candidate_scores=candidate_scores,
                )

                accepted: bool = self._should_accept_prompt_update(
                    current_prompt=current_prompt,
                    new_prompt=new_prompt,
                    step=step,
                    observer=observer,
                )
                observer.record(
                    key="prompt_accepted",
                    value=accepted,
                )
                if not accepted:
                    new_prompt = current_prompt

                self._update_state(
                    step,
                    feedbacks,
                    gradients,
                    current_prompt,
                    new_prompt,
                    all_candidates=candidate_prompts,
                    all_candidate_scores=candidate_scores,
                )
                current_prompt = new_prompt

                self._record_algorithm_state(observer=observer)
                if self.verbosity >= 2:
                    print("=" * 80)
                    print(current_prompt.render_instructions())
                    print("=" * 80)

                if should_evaluate_at_step(
                    step=step,
                    total_steps=self.steps,
                    eval_every=self.eval_every,
                    eval_initial_prompt=self.eval_initial_prompt,
                    eval_first_step=self.eval_first_step,
                    eval_last_step=self.eval_last_step,
                ):
                    if self.verbosity >= 2:
                        print(f"\n{step_prefix} Evaluating...")
                    eval_results: Dict[str, Any] = self.evaluate(
                        dataset=dataset,
                        prompt=current_prompt,
                        step=step,
                    )
                    self._record_evaluation(
                        observer=observer, step=step, results=eval_results
                    )
                    if self.verbosity >= 2:
                        print(
                            f"  Evaluation complete on {len(eval_results['prompt_predictions'])} predictions"
                        )

                observer.log_step_end(step)

            observer.finalize()
            if self.verbosity >= 1:
                print(f"\nTraining complete! Results saved to: {output_dir}")
            try:
                run_logs: pd.DataFrame = ObservabilityManager.read_run_logs(output_dir)
            except (IOError, FileNotFoundError, OSError) as e:
                raise IOError(
                    f"Failed to read run logs from {output_dir!r}:\n"
                    f"{format_exception_msg(e)}"
                ) from e
            return {
                "run_id": run_id,
                "run_config": run_config,
                "output_dir": output_dir,
                "final_prompt": current_prompt,
                "run_logs": run_logs,
            }
        except Exception as e:
            observer.log_error(current_step, format_exception_msg(e))
            raise

    # ------------------------------------------------------------------
    # Abstract methods: subclasses MUST implement
    # ------------------------------------------------------------------
    @abstractmethod
    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """Get algorithm-specific context for pipeline components.

        Returns:
            Dict with algorithm-specific data (trajectory, loss configs, etc.)
        """
        pass

    @abstractmethod
    def _update_state(
        self,
        step: int,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        all_candidates: Optional[List[PromptTemplate]] = None,
        all_candidate_scores: Optional[List[Dict[Task, List[NumericFeedback]]]] = None,
    ) -> None:
        """Update algorithm-specific state after a step."""
        pass

    @abstractmethod
    def _serialize_algorithm_context(self, *, step: int) -> Dict[str, Any]:
        """Return a JSON-serializable dict of algorithm context for logging.

        The observer stores this as-is. All domain serialization happens here.
        """
        pass

    @abstractmethod
    def _serialize_algorithm_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of algorithm state for logging.

        The observer stores this as-is. All domain serialization happens here.
        """
        pass

    # ------------------------------------------------------------------
    # Hooks: subclasses MAY override for algorithm-specific behavior
    # ------------------------------------------------------------------
    def _before_train(
        self,
        *,
        dataset: Dataset,
        initial_prompt: PromptTemplate,
    ) -> None:
        """Hook called once at the start of ``train()``, before the step loop.

        Override to perform one-time setup that requires the dataset
        (e.g. TextGrad samples a fixed validation batch here).

        Args:
            dataset: The Dataset instance passed to ``train()``.
            initial_prompt: The initial prompt template.
        """
        pass

    def _should_accept_prompt_update(
        self,
        *,
        current_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        step: int,
        observer: ObservabilityManager,
    ) -> bool:
        """Hook called after Step 4 to decide whether to accept the new prompt.

        The default implementation always accepts (returns ``True``).
        TextGrad overrides this to implement validation gating: the new
        prompt is evaluated on a held-out validation batch and accepted
        only if it does not regress.

        Args:
            current_prompt: The prompt before this step's optimization.
            new_prompt: The candidate prompt produced by Step 4.
            step: Current optimization step number.
            observer: ObservabilityManager for recording the decision.

        Returns:
            True to accept ``new_prompt``, False to keep ``current_prompt``.
        """
        return True

    def _after_predict(
        self,
        *,
        step: int,
        predictions: List[PredictionResult],
        batch: Batch,
    ) -> None:
        """Hook called after Step 1 (predict).

        Override to store per-instance data needed by later steps
        (e.g. TextGrad/PE2 store predictions+ground_truths).
        """
        pass

    def _before_optimize(
        self,
        *,
        step: int,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
    ) -> None:
        """Hook called after Steps 2-3 (loss + gradient), before Step 4 (optimize).

        Trajectory-based algorithms (OPRO, GPO) override this to push the
        current prompt's (instruction, score) pair into the trajectory so
        the optimizer meta-prompt always contains the current prompt's
        evaluation.  Without this, the trajectory is empty on the first
        step --- the OPRO paper seeds the trajectory with the initial
        instruction before optimization begins.
        """
        pass

    def _build_loss_context(
        self,
        *,
        step: int,
    ) -> Dict[str, Any]:
        """Build context dict for the loss computer. Override to customize."""
        return self._get_algorithm_context(step)

    def _build_gradient_context(
        self,
        *,
        step: int,
    ) -> Dict[str, Any]:
        """Build context dict for the gradient computer. Override to inject
        predictions/ground_truths for TextGrad/PE2."""
        return self._get_algorithm_context(step)

    def _build_optimizer_context(
        self,
        *,
        step: int,
        batch: Batch,
    ) -> Dict[str, Any]:
        """Build context dict for the optimizer. Override to inject extra data."""
        return self._get_algorithm_context(step=step, batch=batch)

    # ------------------------------------------------------------------
    # Recording helpers: serialize domain objects → observer.record()
    # ------------------------------------------------------------------
    def _build_run_config(
        self,
        *,
        initial_prompt: PromptTemplate,
        start_step: int,
    ) -> Dict[str, Any]:
        llm_field_names: Set[str] = {
            "task_llm",
            "optimizer_llm",
            "gradient_llm",
            "loss_llm",
        }
        algo_config: Dict[str, Any] = self.model_dump(exclude=llm_field_names)

        def _make_json_safe(value: Any) -> Any:
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, dict):
                return {str(k): _make_json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_make_json_safe(v) for v in value]
            return repr(value)

        algo_config = _make_json_safe(algo_config)
        algo_config["algo_name"] = self.__class__.__name__
        algo_config["initial_prompt"] = initial_prompt.render_instructions()
        algo_config["start_step"] = start_step

        llm_models: Dict[str, Optional[str]] = {}
        for role in llm_field_names:
            worker: Any = getattr(self, role, None)
            if worker is not None:
                llm_models[role] = getattr(worker, "model_name", type(worker).__name__)
            else:
                llm_models[role] = None
        algo_config["llm_models"] = llm_models

        return algo_config

    @staticmethod
    def _record_batch(
        *,
        observer: ObservabilityManager,
        batch: Batch,
    ) -> None:
        observer.record(
            key="batch",
            value={
                "step": batch.step,
                "num_samples": len(batch.samples),
                "samples": [s.model_dump() for s in batch.samples],
            },
        )

    @staticmethod
    def _record_predictions(
        *,
        observer: ObservabilityManager,
        predictions: List[PredictionResult],
    ) -> None:
        observer.record(
            key="predictions",
            value={
                "num_predictions": len(predictions),
                "predictions": [p.model_dump() for p in predictions],
            },
        )

    def _record_algorithm_context(
        self, *, observer: ObservabilityManager, step: int
    ) -> None:
        observer.record(
            key="algorithm_context",
            value=self._serialize_algorithm_context(step=step),
        )

    @staticmethod
    def _record_feedbacks(
        *,
        observer: ObservabilityManager,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
    ) -> None:
        serialized: Dict[str, List[Dict[str, Any]]] = {}
        for task, feedback_list in feedbacks.items():
            serialized[task.task_name] = [fb.model_dump() for fb in feedback_list]
        observer.record(
            key="feedbacks",
            value={
                "num_tasks": len(feedbacks),
                "feedbacks": serialized,
            },
        )

    @staticmethod
    def _record_gradients(
        *,
        observer: ObservabilityManager,
        gradients: Dict[Task, List[TextGradient]],
    ) -> None:
        serialized: Dict[str, List[Dict[str, Any]]] = {}
        for task, gradient_list in gradients.items():
            serialized[task.task_name] = [g.model_dump() for g in gradient_list]
        observer.record(
            key="gradients",
            value={
                "num_tasks": len(gradients),
                "gradients": serialized,
            },
        )

    @staticmethod
    def _record_prompt_update(
        *,
        observer: ObservabilityManager,
        old_prompt: PromptTemplate,
        new_prompt: PromptTemplate,
        meta_prompt: Optional[str],
        optimizer_response: Optional[str],
        num_candidates_attempted: int,
        num_candidates_succeeded: int,
        candidate_scores: Optional[List[Dict[Task, List[NumericFeedback]]]] = None,
    ) -> None:
        observer.record(
            key="old_prompt_template",
            value=old_prompt.render_instructions(),
        )
        observer.record(key="meta_prompt", value=meta_prompt)
        observer.record(key="meta_prompt_response", value=optimizer_response)
        observer.record(
            key="new_prompt_template",
            value=new_prompt.render_instructions(),
        )

        serialized_candidate_scores = None
        if candidate_scores is not None:
            serialized_candidate_scores = [
                {
                    task.task_name: [fb.model_dump() for fb in fbs]
                    for task, fbs in cs.items()
                }
                for cs in candidate_scores
            ]

        observer.record(
            key="prompt_update",
            value={
                "old_instruction": old_prompt.instruction,
                "new_instruction": new_prompt.instruction,
                "num_candidates_attempted": num_candidates_attempted,
                "num_candidates_succeeded": num_candidates_succeeded,
                "candidate_scores": serialized_candidate_scores,
            },
        )

    def _record_algorithm_state(self, *, observer: ObservabilityManager) -> None:
        observer.record(
            key="algorithm_state",
            value=self._serialize_algorithm_state(),
        )

    def _record_evaluation(
        self,
        *,
        observer: ObservabilityManager,
        step: int,
        results: Dict[str, Any],
    ) -> None:
        """Flatten evaluation results into a DataFrame, compute all metrics, and write to parquet.

        Computes every registered Metric subclass on each task's (y_true, y_pred)
        arrays.  Metrics that require extra kwargs (e.g. ``num_classes`` for LCE)
        receive them from ``self._num_classes``.  Metrics that fail on a particular
        task (e.g. Spearman on constant predictions) are recorded as None.

        This lives on the algorithm (not the observer) because it knows the
        shape of PredictionResult and DatasetSample.
        """
        from .metrics import Metric

        predictions: List[PredictionResult] = results["prompt_predictions"]
        dataset_inputs: List[DatasetSample] = results["dataset_inputs"]

        pred_map: Dict[str, PredictionResult] = {p.sample_id: p for p in predictions}
        input_map: Dict[str, DatasetSample] = {s.sample_id: s for s in dataset_inputs}

        flattened: List[Dict[str, Any]] = []
        for sid, sample in input_map.items():
            pred_obj: Optional[PredictionResult] = pred_map.get(sid)
            if pred_obj is None:
                raise ValueError(f"No prediction found for sample {sid}")

            ground_truth_flat: Dict[str, Any] = {
                f"gt_{k}": v for k, v in sample.ground_truths.items()
            }
            pred_flat: Dict[str, Any] = {
                f"pred_{k}": v for k, v in pred_obj.task_outputs.items()
            }

            row: Dict[str, Any] = {
                "step": step,
                "sample_id": sid,
                "prompt": pred_obj.prompt,
                "raw_response": pred_obj.raw_response,
                "parser_error": pred_obj.parser_error,
                "inputs": sample.inputs,
            }
            row.update(ground_truth_flat)
            row.update(pred_flat)
            flattened.append(row)

        dataframe: pd.DataFrame = pd.DataFrame(flattened)
        num_parse_failures: int = int(dataframe["parser_error"].notna().sum())
        output_path: str = observer.write_parquet(
            relative_path=f"eval_step_{step}.parquet",
            dataframe=dataframe,
        )

        all_metric_classes: List[type] = Metric.subclasses()
        task_metrics: Dict[str, Dict[str, Optional[float]]] = {}

        for task in self.tasks:
            task_name: str = task.task_name
            gt_col: str = f"gt_{task_name}"
            pred_col: str = f"pred_{task_name}"
            if gt_col not in dataframe.columns or pred_col not in dataframe.columns:
                continue

            valid_mask = dataframe[gt_col].notna() & dataframe[pred_col].notna()
            y_true: List[Any] = dataframe.loc[valid_mask, gt_col].tolist()
            y_pred: List[Any] = dataframe.loc[valid_mask, pred_col].tolist()

            task_metric_results: Dict[str, Optional[float]] = {}
            metric_kwargs: Dict[str, Any] = {}
            if (
                self._task_output_specs is not None
                and task_name in self._task_output_specs
            ):
                nc: int = self._task_output_specs[task_name].num_classes
                if nc > 0:
                    metric_kwargs["num_classes"] = nc

            for metric_cls in all_metric_classes:
                try:
                    value: float = metric_cls.compute(
                        y_true=y_true, y_pred=y_pred, **metric_kwargs
                    )
                    task_metric_results[metric_cls.__name__] = value
                except (TypeError, ValueError, ZeroDivisionError):
                    task_metric_results[metric_cls.__name__] = None

            task_metrics[task_name] = task_metric_results

        observer.record(
            key="evaluation",
            value={
                "step": step,
                "num_samples": len(flattened),
                "num_parse_failures": num_parse_failures,
                "results_file": output_path,
                "task_metrics": task_metrics,
            },
        )

    # ------------------------------------------------------------------
    # Multi-candidate optimize helper
    # ------------------------------------------------------------------
    def _run_optimize(
        self,
        *,
        optimizer: PromptOptimizer,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        batch: Batch,
        step: int,
    ) -> Tuple[
        PromptTemplate,
        str,
        str,
        Optional[List[PromptTemplate]],
        Optional[List[Dict[Task, List[NumericFeedback]]]],
    ]:
        """Run Step 4 with optional multi-candidate selection.

        Returns:
            Tuple of (new_prompt, best_meta_prompt, best_raw_response,
            candidate_prompts_or_None, candidate_scores_or_None).
        """
        opt_context = self._build_optimizer_context(step=step, batch=batch)
        n_candidates = max(1, self.num_candidates)
        candidate_prompts: List[PromptTemplate] = []
        candidate_scores: List[Dict[Task, List[NumericFeedback]]] = []
        candidate_meta_prompts: List[str] = []
        candidate_raw_responses: List[str] = []

        for ci in range(n_candidates):
            try:
                result_i = optimizer.optimize(
                    gradients,
                    current_prompt,
                    self.tasks,
                    self.optimizer_llm,
                    verbosity=self.verbosity if ci == 0 else 0,
                    **opt_context,
                )
                candidate_prompts.append(result_i.new_prompt)
                candidate_meta_prompts.append(result_i.meta_prompt)
                candidate_raw_responses.append(result_i.raw_response)
            except (ValueError, RuntimeError) as exc:
                if self.verbosity >= 1:
                    warnings.warn(
                        f"Candidate {ci + 1}/{n_candidates} "
                        f"failed to generate a valid instruction: "
                        f"{format_exception_msg(exc)}",
                        stacklevel=2,
                    )
                continue

        if len(candidate_prompts) == 0:
            raise RuntimeError(
                f"All {n_candidates} candidate generations failed at step {step}."
            )

        if self.num_candidates > 1:
            if self.verbosity >= 2:
                print(
                    f"  Evaluating {len(candidate_prompts)} candidate(s) on "
                    f"{len(batch.samples)} batch samples..."
                )
            for cand in candidate_prompts:
                scores_i = self._evaluate_candidate_prompt(
                    candidate=cand,
                    batch=batch,
                    step=step,
                )
                candidate_scores.append(scores_i)
            ranking = [self._candidate_ranking_score(cs) for cs in candidate_scores]
            best_idx = int(max(range(len(ranking)), key=lambda i: ranking[i]))
            if self.verbosity >= 2:
                ranking_strs = [f"{r:.4f}" for r in ranking]
                print(
                    f"  Candidate scores: [{', '.join(ranking_strs)}] "
                    f"-> best = #{best_idx + 1}"
                )
        else:
            best_idx = 0

        if self.verbosity >= 2:
            print(f"  Selected candidate #{best_idx + 1} of {len(candidate_prompts)}")

        return (
            candidate_prompts[best_idx],
            candidate_meta_prompts[best_idx],
            candidate_raw_responses[best_idx],
            candidate_prompts if self.num_candidates > 1 else None,
            candidate_scores if self.num_candidates > 1 else None,
        )

    def _require_input_col_labels(self) -> Dict[str, str]:
        """Return input_col_labels, raising if train() has not been called."""
        if self._input_col_labels is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._input_col_labels is None. "
                f"This field is set at the start of train(). "
                f"If calling _get_algorithm_context outside of train(), "
                f"set _input_col_labels = dataset.input_col_labels first."
            )
        return self._input_col_labels

    def _require_task_demonstrations(self) -> List[DatasetSample]:
        """Return task_demonstrations, raising if train() has not been called."""
        if self._task_demonstrations is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._task_demonstrations is None. "
                f"This field is set each step in train(). "
                f"If calling _get_algorithm_context outside of train(), "
                f"set _task_demonstrations = [] first."
            )
        return self._task_demonstrations

    def _build_loss_fn_config(
        self,
        *,
        task_name: str,
        use_textual: bool,
    ) -> Dict[str, Any]:
        """Build per-task loss function config dict with metric-specific kwargs.

        Injects ``num_classes`` from the task's ``TaskOutputSpec`` when
        available and > 0 (discrete tasks).  Continuous tasks
        (``num_classes == 0``) do not receive ``num_classes``.

        Args:
            task_name: Name of the task.
            use_textual: Whether to enable textual feedback for this task.

        Returns:
            Config dict suitable for ``_compute_numeric_loss(loss_fn_config=...)``.
        """
        config: Dict[str, Any] = {
            "metric": self.task_losses[task_name],
            "use_textual": use_textual,
        }
        if self._task_output_specs is not None and task_name in self._task_output_specs:
            nc: int = self._task_output_specs[task_name].num_classes
            if nc > 0:
                config["num_classes"] = nc
        return config

    def _evaluate_candidate_prompt(
        self,
        *,
        candidate: PromptTemplate,
        batch: Batch,
        step: int,
    ) -> Dict[Task, List[NumericFeedback]]:
        """Run predict + numeric loss for a candidate prompt on the batch.

        This is the common "score a candidate" operation used by multi-candidate
        selection (OPRO generates 8 candidates per step, evaluates each, keeps
        the best).  Only numeric metrics are computed — no LLM-based textual
        feedback — because the purpose is fast, cheap candidate ranking.

        Args:
            candidate: The prompt template to evaluate.
            batch: The training batch to evaluate on.
            step: Current optimization step (for algorithm context).

        Returns:
            Dict mapping Task -> list of NumericFeedback for that candidate.
        """
        predictor: TaskPredictor = TaskPredictor.of(self.task_predictor["name"])
        loss_comp: LossComputer = LossComputer.of(self.loss_computer["name"])
        algorithm_context: Dict[str, Any] = self._get_algorithm_context(step)

        predictions: List[PredictionResult] = predictor.predict(
            batch,
            candidate,
            self.task_llm,
            verbosity=0,
            **algorithm_context,
        )

        force_numeric_context: Dict[str, Any] = dict(algorithm_context)
        if "loss_functions" not in force_numeric_context:
            raise ValueError(
                f"algorithm_context must contain 'loss_functions', "
                f"but only found keys: {list(force_numeric_context.keys())}."
            )
        loss_fns: Dict[str, Dict[str, Any]] = force_numeric_context["loss_functions"]
        numeric_only_loss_fns: Dict[str, Dict[str, Any]] = {
            k: {**v, "use_textual": False} for k, v in loss_fns.items()
        }
        force_numeric_context["loss_functions"] = numeric_only_loss_fns

        feedbacks = loss_comp.compute(
            predictions=predictions,
            ground_truths=batch.samples,
            tasks=self.tasks,
            prompt_template=candidate,
            llm_pool=None,
            loss_batch_size=self.loss_batch_size,
            verbosity=0,
            **force_numeric_context,
        )

        numeric_scores: Dict[Task, List[NumericFeedback]] = {}
        for task, fb_list in feedbacks.items():
            nfbs: List[NumericFeedback] = [
                fb for fb in fb_list if isinstance(fb, NumericFeedback)
            ]
            if len(nfbs) > 0:
                numeric_scores[task] = nfbs
        return numeric_scores

    @staticmethod
    def _candidate_ranking_score(
        numeric_scores: Dict[Task, List[NumericFeedback]],
    ) -> float:
        """Compute a single scalar ranking score for candidate selection.

        Uses NumericFeedback.normalized_score (higher-is-better regardless of
        direction) so that "maximize accuracy" and "minimize loss" are handled
        uniformly.  The ranking score is the mean of per-task means.

        Args:
            numeric_scores: Per-task numeric feedbacks for one candidate.

        Returns:
            Scalar ranking score (higher is better).
        """
        if len(numeric_scores) == 0:
            return float("-inf")

        task_means: List[float] = []
        for _task, feedbacks in numeric_scores.items():
            if len(feedbacks) > 0:
                task_mean: float = sum(fb.normalized_score for fb in feedbacks) / len(
                    feedbacks
                )
                task_means.append(task_mean)

        if len(task_means) == 0:
            return float("-inf")
        return sum(task_means) / len(task_means)

    def _sample_batch(
        self,
        *,
        data: pd.DataFrame,
        dataset: Dataset,
        step: int,
        full: bool = False,
    ) -> Batch:
        """Sample batch from dataset.

        When ``use_fixed_training_batch`` is True, the first call (step 1)
        samples a batch and caches it.  All subsequent calls return the
        same cached batch (with ``step`` updated).  This matches the OPRO
        paper (exp.tex L84): "The same subset is used throughout
        optimization."  ``full=True`` calls (evaluation) always sample
        fresh because evaluation uses the full test set.

        When ``use_fixed_training_batch`` is False (default), each step
        samples a different batch using ``step`` as the random seed.

        Args:
            data: Full dataset DataFrame
            dataset: Dataset object (for column info)
            step: Step number (used as random seed when not fixed)
            full: If True, use all data (for evaluation).

        Returns:
            Batch object with samples
        """
        if not full and self.use_fixed_training_batch:
            if self._fixed_batch is not None:
                cached_samples: List[DatasetSample] = [
                    DatasetSample(
                        sample_id=f"step{step}_sample{i}",
                        inputs=s.inputs,
                        ground_truths=s.ground_truths,
                    )
                    for i, s in enumerate(self._fixed_batch.samples)
                ]
                return Batch(step=step, samples=cached_samples)

        shuffled: pd.DataFrame = data.sample(
            frac=1,
            random_state=0 if self.use_fixed_training_batch else abs(int(step)),
        ).reset_index(drop=True)
        batch_data: pd.DataFrame = shuffled if full else shuffled.head(self.batch_size)

        samples: List[DatasetSample] = []
        for idx, row in batch_data.iterrows():
            inputs: Dict[str, Any] = {}
            ground_truths: Dict[str, Any] = {}

            # Extract input columns
            for col in dataset.input_cols:
                if col in row:
                    inputs[col] = row[col]

            # Extract ground truth columns
            for col in dataset.gt_cols:
                if col in row:
                    ground_truths[col] = row[col]

            samples.append(
                DatasetSample(
                    sample_id=f"step{step}_sample{idx}",
                    inputs=inputs,
                    ground_truths=ground_truths,
                )
            )

        batch: Batch = Batch(step=step, samples=samples)
        if not full and self.use_fixed_training_batch and self._fixed_batch is None:
            self._fixed_batch = batch
        return batch

    def _sample_task_demonstrations(
        self,
        *,
        data: pd.DataFrame,
        dataset: Dataset,
        batch: Batch,
        step: int,
    ) -> List[DatasetSample]:
        """Sample task demonstrations from the training set, excluding the current batch.

        These are (input, ground_truth) pairs shown to the optimizer LLM so it
        understands how the prompt is applied to real data.  The GPO paper uses
        3 such examples (Appendix A: "we randomly sample 3 examples from the
        dataset and fill them into the meta-prompt of the prompt optimizer").

        Returns an empty list when ``num_task_demonstrations == 0``.

        .. todo:: Hard-negative sampling (OPRO paper, application.tex L95):
            "choose the ones the previous instructions fall short of."
            Accept predictions from the current step and preferentially
            select exemplars where the scorer LLM got the wrong answer,
            giving the optimizer indirect signal about failure modes.
        """
        if self.num_task_demonstrations <= 0:
            return []

        batch_sample_ids: Set[str] = {s.sample_id for s in batch.samples}

        shuffled: pd.DataFrame = data.sample(
            frac=1,
            random_state=abs(int(step)),
        ).reset_index(drop=True)

        demos: List[DatasetSample] = []
        for idx, row in shuffled.iterrows():
            if len(demos) >= self.num_task_demonstrations:
                break
            sample_id: str = f"step{step}_demo{idx}"
            if sample_id in batch_sample_ids:
                continue

            inputs: Dict[str, Any] = {
                col: row[col] for col in dataset.input_cols if col in row
            }
            ground_truths: Dict[str, Any] = {
                col: row[col] for col in dataset.gt_cols if col in row
            }
            demos.append(
                DatasetSample(
                    sample_id=sample_id,
                    inputs=inputs,
                    ground_truths=ground_truths,
                )
            )

        return demos

    @validate
    def evaluate(
        self,
        dataset: Dataset,
        prompt: PromptTemplate,
        step: int,
    ) -> Dict[str, Any]:
        """Evaluate prompt on test set.

        Args:
            dataset: Dataset object
            prompt: Prompt template to evaluate
            step: Current step number

        Returns:
            Dict with prompt_predictions, dataset_inputs
        """
        # Default implementation: return empty dict
        # Subclasses can override for actual evaluation
        test_data = dataset.test()
        test_batch = self._sample_batch(
            data=test_data,
            dataset=dataset,
            step=step,
            full=True,
        )

        predictor: TaskPredictor = TaskPredictor.of(self.task_predictor["name"])
        if self.verbosity >= 2:
            print(
                f"Evaluating {len(test_batch.samples)} samples using {self.task_predictor['name']} predictor"
            )

        context_kwargs = self._get_algorithm_context(step=step, batch=test_batch)
        context_kwargs.pop("batch", None)

        predictions = predictor.predict(
            test_batch,
            prompt,
            self.task_llm,
            verbosity=self.verbosity,
            **context_kwargs,
        )

        if self.verbosity >= 2:
            print(f"Generated {len(predictions)} predictions.")

        return {
            "prompt_predictions": predictions,
            "dataset_inputs": test_batch.samples,
        }
