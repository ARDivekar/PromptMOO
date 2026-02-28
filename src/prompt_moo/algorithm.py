"""
Main Algorithm Implementation: Core prompt optimization loop.

This module implements the base algorithm class and specific algorithm implementations like OPRO.
"""

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# from tqdm import tqdm as ProgressBar
from concurry import ProgressBar
from morphic import Registry, Typed
from morphic.typed import format_exception_msg

from .data_input import Dataset
from .data_structures import (
    Batch,
    DatasetSample,
    NumericFeedback,
    Task,
)
from .gradient_computer import GradientComputer
from .loss_computer import LossComputer
from .observability import ObservabilityManager
from .prompt_optimizer import PromptOptimizer
from .prompt_template_utils import PromptTemplate
from .prompt_trajectory import PromptTrajectory, TrajectoryElement, OPROTrajectoryElement, GPOTrajectoryElement
from .task_predictor import TaskPredictor


class PromptAlgorithm(Typed, Registry, ABC):
    """Base class for prompt optimization algorithms.

    This implements the core 4-step optimization loop:
    1. Predict: Generate predictions using task LLM
    2. Compute Losses: Calculate feedback from predictions
    3. Compute Gradients: Generate improvement suggestions
    4. Optimize: Update prompt based on gradients
    """

    _allow_subclass_override = True

    # Core components (configured via dicts or Registry keys)
    task_predictor: Dict[str, Any] = {"name": "standard"}
    loss_computer: Dict[str, Any]
    gradient_computer: Dict[str, Any]
    prompt_optimizer: Dict[str, Any]

    # LLM workers (Ray workers with AsyncIO pools)
    task_llm: Any
    gradient_llm: Optional[Any] = None  # For gradient computation
    optimizer_llm: Any  # For prompt optimization
    loss_llm: Optional[Any] = None  # For textual feedback

    # Training hyperparameters
    steps: int
    batch_size: int
    loss_batch_size: int
    gradient_batch_size: int
    eval_every: int
    name: str
    verbosity: int = (
        1  # 0=silent, 1=default (progress bar), 2=detailed, 3=debug (with LLM I/O)
    )
    substep_delay: float = 1.5  # Delay in seconds between substeps (for rate limiting)

    # Tasks
    tasks: List[Task]

    def train(
        self,
        dataset: Dataset,
        initial_prompt: PromptTemplate,
        output_dir: Optional[str] = None,
        start_step: int = 0 ## Resume from this step
    ) -> Dict[str, Any]:
        """Main training loop.

        Args:
            dataset: Dataset object with train/test data
            initial_prompt: Initial prompt template
            output_dir: Optional output directory (auto-generated if None)
            start_step: Step to start/resume training from (default: 0)

        Returns:
            Dict with final_prompt, output_dir, and run_id
        """
        # Setup observability
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = (
                f"outputs/{self.class_name}_{dataset.dataset_name}_{self.name}_{run_id}"
            )
        os.makedirs(output_dir, exist_ok=True)
        if self.verbosity >= 1:
            print(f"Output directory: {output_dir}")

        run_config = {
            "algorithm": self.__class__.__name__,
            "task_predictor": self.task_predictor,
            "loss_computer": self.loss_computer,
            "gradient_computer": self.gradient_computer,
            "prompt_optimizer": self.prompt_optimizer,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "loss_batch_size": self.loss_batch_size,
            "gradient_batch_size": self.gradient_batch_size,
            "eval_every": self.eval_every,
            "tasks": [t.model_dump() for t in self.tasks],
            "initial_prompt": initial_prompt.to_str(),
            "start_step" : start_step
        }
        observer = ObservabilityManager(output_dir)
        observer.log_config(run_config)

        # Get component instances
        predictor = TaskPredictor.of(self.task_predictor["name"])
        loss_comp = LossComputer.of(self.loss_computer["name"])
        grad_comp = GradientComputer.of(self.gradient_computer["name"])
        optimizer = PromptOptimizer.of(self.prompt_optimizer["name"])

        current_prompt = initial_prompt
        train_data = dataset.train()

        # Control progress bar based on verbosity
        disable_progress_bar = self.verbosity == 0
        current_step = start_step
        try: 
            for step in ProgressBar(
                list(range(start_step, self.steps)), desc="Training", disable=disable_progress_bar
            ):
                current_step = step
                if self.verbosity >= 2:
                    print(f"\n===== Step {step + 1}/{self.steps} =====")
                observer.log_step_start(step)

                # Sample batch
                batch = self._sample_batch(data=train_data, dataset=dataset, step=step)
                observer.log_batch(batch)

                step_prefix = f"[Step {step + 1}/{self.steps}]"

                # Step 1: Predict
                if self.verbosity >= 2:
                    print(
                        f"\n{step_prefix} 1/4: Predicting with {len(batch.samples)} samples..."
                    )
                predictions = predictor.predict(
                    batch,
                    current_prompt,
                    self.task_llm,
                    verbosity=self.verbosity,
                    **self._get_algorithm_context(step),
                )
                observer.log_predictions(predictions)
                if self.verbosity >= 2:
                    print(f"  Generated {len(predictions)} predictions")

                # Log algorithm context used for this step
                algorithm_context = self._get_algorithm_context(step)
                observer.log_algorithm_context(algorithm_context)

                # Delay between substeps for rate limiting
                if self.substep_delay > 0:
                    time.sleep(self.substep_delay)

                # Step 2: Compute losses/feedback
                if self.verbosity >= 2:
                    print(f"\n{step_prefix} 2/4: Computing losses...")
                feedbacks = loss_comp.compute(
                    predictions=predictions,
                    ground_truths=batch.samples,
                    tasks=self.tasks,
                    llm_pool=self.loss_llm,
                    loss_batch_size=self.loss_batch_size,
                    verbosity=self.verbosity,
                    **algorithm_context,
                )
                observer.log_feedbacks(feedbacks)
                if self.verbosity >= 2:
                    print(f"  Computed feedbacks for {len(feedbacks)} tasks")

                # Delay between substeps for rate limiting
                if self.substep_delay > 0:
                    time.sleep(self.substep_delay)

                # Step 3: Compute gradients
                if self.verbosity >= 2:
                    print(f"\n{step_prefix} 3/4: Computing gradients...")
                gradients = grad_comp.compute(
                    feedbacks=feedbacks,
                    prompt_template=current_prompt,
                    tasks=self.tasks,
                    llm_pool=self.gradient_llm,
                    gradient_batch_size=self.gradient_batch_size,
                    verbosity=self.verbosity,
                    **self._get_algorithm_context(step),
                )
                observer.log_gradients(gradients)
                if self.verbosity >= 2:
                    print(f"  Computed gradients for {len(gradients)} tasks")

                # Delay between substeps for rate limiting
                if self.substep_delay > 0:
                    time.sleep(self.substep_delay)

                # Step 4: Optimize prompt
                if self.verbosity >= 2:
                    print(f"\n{step_prefix} 4/4: Optimizing prompt...")
                candidates_context = self._get_algorithm_context(step=step, batch=batch)
                optimizer_result = optimizer.optimize(
                    gradients,
                    current_prompt,
                    self.tasks,
                    self.optimizer_llm,
                    verbosity=self.verbosity,
                    **candidates_context,
                )
                new_prompt = optimizer_result.new_prompt
                observer.log_prompt_update(
                    current_prompt,
                    new_prompt,
                    meta_prompt=optimizer_result.meta_prompt,
                    optimizer_response=optimizer_result.raw_response,
                )
                if self.verbosity >= 2:
                    print("  Generated new prompt")

                # Update algorithm state
                self._update_state(step, feedbacks, gradients, new_prompt)
                current_prompt = new_prompt

                # Log updated algorithm state
                observer.log_algorithm_state(self._get_algorithm_state())
                if self.verbosity >= 2:
                    print("=" * 80)
                    print(current_prompt.to_str())
                    print("=" * 80)

                # Evaluate if needed
                if step % self.eval_every == 0 or step == self.steps - 1:
                    if self.verbosity >= 2:
                        print(f"\n{step_prefix} Evaluating...")
                    eval_results: Dict[str, Any] = self.evaluate(
                        dataset=dataset,
                        prompt=current_prompt,
                        step=step,
                    )
                    observer.log_evaluation(step, eval_results)
                    if self.verbosity >= 2:
                        print(
                            f"  Evaluation complete on {len(eval_results['prompt_predictions'])} predictions"
                        )

                observer.log_step_end(step)

            # Finalize
            observer.finalize()
            if self.verbosity >= 1:
                print(f"\nTraining complete! Results saved to: {output_dir}")
            run_logs_path = os.path.join(output_dir, "run_logs.parquet")
            try:
                run_logs = pd.read_parquet(run_logs_path, engine="pyarrow")
            except Exception as e:
                raise IOError(
                    f"Failed to read run logs parquet at {run_logs_path!r}:\n"
                    f"{format_exception_msg(e)}"
                ) from e
            # run_logs.to_parquet(
            #     os.path.join(output_dir, "run_logs.parquet"), engine="fastparquet"
            # )

            return {
                "run_id": run_id,
                "run_config": run_config,
                "output_dir": output_dir,
                "final_prompt": current_prompt,
                "run_logs": run_logs,
            }
        except Exception as e:
            observer.log_error(current_step, str(e))
            raise e

    @abstractmethod
    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """Get algorithm-specific context for components.

        Args:
            step: Current step number

        Returns:
            Dict with algorithm-specific data (e.g., trajectory, loss configs)
        """
        pass

    @abstractmethod
    def _update_state(
        self, step: int, feedbacks: Dict, gradients: Dict, new_prompt: PromptTemplate
    ):
        """Update algorithm-specific state.

        Args:
            step: Current step number
            feedbacks: Feedbacks from loss computer
            gradients: Gradients from gradient computer
            new_prompt: New prompt template
        """
        pass

    @abstractmethod
    def _get_algorithm_state(self) -> Dict[str, Any]:
        """Get current algorithm state for logging.

        Returns:
            Dict with algorithm-specific state (trajectory, previous_instructions, etc.)
        """
        pass

    def _sample_batch(
        self, *, data: pd.DataFrame, dataset: Dataset, step: int, full: bool = False
    ) -> Batch:
        """Sample batch from dataset.

        Args:
            data: Full dataset DataFrame
            dataset: Dataset object (for column info)
            step: Step number (used as random seed)

        Returns:
            Batch object with samples
        """
        # Shuffle with step as seed for reproducibility
        shuffled = data.sample(frac=1, random_state=step).reset_index(drop=True)
        batch_data = shuffled if full else shuffled.head(self.batch_size)

        samples = []
        for idx, row in batch_data.iterrows():
            inputs = {}
            ground_truths = {}

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

        return Batch(step=step, samples=samples)

    def evaluate(
        self,
        dataset: Dataset,
        prompt: PromptTemplate,
        step: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate prompt on test set.

        Args:
            dataset: Dataset object
            prompt: Prompt template to evaluate

        Returns:
            Dict with evaluation metrics
        """
        # Default implementation: return empty dict
        # Subclasses can override for actual evaluation
        test_data = dataset.test()
        test_batch = self._sample_batch(
            data=test_data, dataset=dataset, step=step, full=True
        )

        predictor = TaskPredictor.of(self.task_predictor["name"])
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
            "task_prompt": prompt.to_str(),
            "prompt_predictions": predictions,
            "dataset_inputs": test_batch.samples,
        }


class OPRO(PromptAlgorithm):
    """OPRO algorithm implementation.

    OPRO uses a top-k trajectory of (prompt, score) pairs to guide optimization.
    """

    # Component configuration (will be dicts with "name" key)
    loss_computer: Dict[str, Any] = {"name": "opro"}
    gradient_computer: Dict[str, Any] = {"name": "opro"}
    prompt_optimizer: Dict[str, Any] = {"name": "opro"}

    # OPRO-specific parameters
    k: int = 3  # Top-k tracking
    task_losses: Dict[str, str] = {}  # Map of task_name -> metric_name

    # Trajectory tracking (initialized lazily)
    _trajectory: Optional[PromptTrajectory] = None

    @property
    def trajectory(self) -> PromptTrajectory:
        """Get or create trajectory."""
        if self._trajectory is None:
            self._trajectory = PromptTrajectory(k=self.k)
        return self._trajectory

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """Get OPRO-specific context.

        Args:
            step: Current step number

        Returns:
            Dict with trajectory and loss_functions config
        """
        # Build loss functions config
        loss_functions = {}
        for task in self.tasks:
            if task.task_name in self.task_losses:
                loss_functions[task.task_name] = {
                    "metric": self.task_losses[task.task_name],
                    "use_textual": False,  # OPRO doesn't use textual feedback
                }
            else:
                raise ValueError(f"Unsupported task: {task.task_name}")
        return {"trajectory": self.trajectory, "loss_functions": loss_functions}

    def _update_state(
        self, step: int, feedbacks: Dict, gradients: Dict, new_prompt: PromptTemplate
    ):
        """Update OPRO trajectory.

        Args:
            step: Current step number
            feedbacks: Feedbacks from loss computer
            gradients: Gradients from gradient computer
            new_prompt: New prompt template
        """
        # Extract scores from feedbacks - AVERAGE scores per metric across batches
        scores = {}
        score_counts = {}  # Track count for averaging
        for task, feedback_list in feedbacks.items():
            if task.task_name not in scores:
                scores[task.task_name] = {}
                score_counts[task.task_name] = {}
            for fb in feedback_list:
                if isinstance(fb, NumericFeedback):
                    metric = fb.metric_name
                    if metric not in scores[task.task_name]:
                        scores[task.task_name][metric] = 0.0
                        score_counts[task.task_name][metric] = 0
                    scores[task.task_name][metric] += fb.value
                    score_counts[task.task_name][metric] += 1
        
        # Compute averages
        for task_name in scores:
            for metric in scores[task_name]:
                count = score_counts[task_name][metric]
                if count > 0:
                    scores[task_name][metric] /= count

        # Extract grads text
        grads = {}
        # for task, grad_list in gradients.items():
        #     grads[task.task_name] = " ".join(g.gradient_text for g in grad_list)
        for task_name, task_scores in scores.items():
            metric_strs = [f"{metric}: {value:.4f}" for metric, value in task_scores.items()]
            grads[task_name] = ", ".join(metric_strs)

        # Extract instructions
        instructions = new_prompt.instruction

        # Build loss_fns dict for TrajectoryElement (simplified - just metadata)
        loss_fns = {}
        for task in self.tasks:
            if task.task_name in self.task_losses:
                metric = self.task_losses[task.task_name]
                loss_type = "max" if metric.lower() in ["accuracy", "f1"] else "min"
                loss_fns[task.task_name] = {
                    "metric": metric,
                    "type": loss_type,
                }
            else:
                raise ValueError(f"Unsupported task: {task.task_name}")

        # Create trajectory element
        element = OPROTrajectoryElement(
            loss_fns=loss_fns, scores=scores, grads=grads, instructions=instructions
        )
        self.trajectory.push(element)
        
        # Debug output when verbosity >= 2
        if self.verbosity >= 2:
            top_k_str = self.trajectory.get_top_k_str()
            print(f"\\n[OPRO Debug] Trajectory top-k string length: {len(top_k_str)} chars")
            print(f"[OPRO Debug] Trajectory element count: {len(self.trajectory)}/{self.k}")
            print(f"[OPRO Debug] Top-k string preview:")
            print(top_k_str)
            # print(f"[OPRO Debug] Top-k string preview (last 500 chars):")
            # print(top_k_str[-500:])

    def _get_algorithm_state(self) -> Dict[str, Any]:
        """Get OPRO algorithm state.

        Returns:
            Dict with trajectory state
        """
        return {
            "trajectory": self.trajectory,
            "k": self.k,
        }


class GPO(PromptAlgorithm):
    """
    GPO algorithm implementation.

    GPO performs iterative prompt optimization using:
    - Retrieval-based trajectory for update direction
    - Meta-prompt generation for candidate prompts
    - Candidate evaluation + best selection
    """

    aliases = ["gpo"]

    loss_computer: Dict[str, Any] = {"name": "gpo"}
    gradient_computer: Dict[str, Any] = {"name": "gpo"}
    prompt_optimizer: Dict[str, Any] = {"name": "gpo"}

    k: int = 5  # Number of past prompts to  retrieve for update direction
    warmup_steps: int = 3
    initial_step_size: int = (
        25  ## How much % of the initial instruction would be updated initially
    )
    final_step_size: int = 20  ## How much % of the instruction will be updated finally.
    use_warmup: bool = True
    task_losses: Dict[str, str] = {}  # Map of task_name -> Metric Name

    # Trajectory
    _trajectory: Optional[PromptTrajectory] = None

    @property
    def trajectory(self) -> PromptTrajectory:
        """Get or create trajectory."""
        if self._trajectory is None:
            self._trajectory = PromptTrajectory(k=self.k)
        return self._trajectory

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """
        Get GPO-specific context for loss, gradient, and optimizer components.

        Includes:
        - Trajectory: full past prompt-performance history
        - Loss functions configuration per task
        """
        loss_functions = {}
        for task in self.tasks:
            if task.task_name in self.task_losses:
                loss_functions[task.task_name] = {
                    "metric": self.task_losses[task.task_name],
                    "use_textual": True,  # GPO may leverage textual signals
                }
            else:
                raise ValueError(f"Unsupported task: {task.task_name}")

        context: Dict[str, Any] = {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.steps,
            "use_warmup": self.use_warmup,
            "initial_step_size": self.initial_step_size,
            "final_step_size": self.final_step_size,
            "trajectory": self.trajectory,
            "loss_functions": loss_functions,
            "top_k_retrieve": self.k,
        }

        if batch is not None:
            context["batch"] = batch

        return context

    def _update_state(
        self, step: int, feedbacks: Dict, gradients: Dict, new_prompt: PromptTemplate
    ):
        """
        Update GPO Trajectory after each step.

        Collects:
        - scores from feeddbacks (loss/metric)
        - Gradients as textual signals for meta-prompt
        - new prompt instructions
        - loss functions metadata for retrieval
        """

        ## Extract scores per task from feedback - AVERAGE scores per metric across batches
        scores = {}
        score_counts = {}  # Track count for averaging
        for task, feedback_list in feedbacks.items():
            if task.task_name not in scores:
                scores[task.task_name] = {}
                score_counts[task.task_name] = {}
            for fb in feedback_list:
                if isinstance(fb, NumericFeedback):
                    metric = fb.metric_name
                    if metric not in scores[task.task_name]:
                        scores[task.task_name][metric] = 0.0
                        score_counts[task.task_name][metric] = 0
                    scores[task.task_name][metric] += fb.value
                    score_counts[task.task_name][metric] += 1
        
        # Compute averages
        for task_name in scores:
            for metric in scores[task_name]:
                count = score_counts[task_name][metric]
                if count > 0:
                    scores[task_name][metric] /= count

        ## Extract gradients as textual descriptions
        # grads = {}
        # for task, grad_list in gradients.items():
        #     grads[task.task_name] = " ".join(g.gradient_text for g in grad_list)
        ## GPO uses textual gradients + numerical summary
        ## Store as list format: [textual_grad_1, ..., textual_grad_N, numerical_summary]
        grads = {}
        for task, grad_list in gradients.items():
            # Each grad in grad_list is a Textual gradient from LLM
            textual_grads = [g.gradient_text for g in grad_list]
            
            # Add numerical summary ass the last element
            task_scores = scores.get(task.task_name, {})
            numerical_summary = ", ".join(f"{metric}: {score:.4f}" for metric, score in task_scores.items())

            # Combine textual gradients and numerical summary
            all_grads = textual_grads + [f"[Score] {numerical_summary}"]
            grads[task.task_name] = " | ".join(all_grads)

        ## Extract the prompt instruction content
        instructions = new_prompt.instruction

        ## Build loss_fns dictionary (metric + max/min objective)
        loss_fns = {}
        for task in self.tasks:
            if task.task_name in self.task_losses:
                metric = self.task_losses[task.task_name]
                loss_type = "max" if metric.lower() in ["accuracy", "f1"] else "min"
                loss_fns[task.task_name] = {
                    "metric": metric,
                    "type": loss_type,
                }
            else:
                raise ValueError(f"Unsupported task: {task.task_name}")

        ## Create Trajectory Element for retrieval in next steps
        # Extract textual gradients and numerical summary for GPOTrajectoryElement
        text_grads_list = []
        numerical_summary = None
        for task, grad_list in gradients.items():
            text_grads_list.extend([g.gradient_text for g in grad_list])
            task_scores = scores.get(task.task_name, {})
            numerical_summary = ", ".join(f"{m}: {v:.4f}" for m, v in task_scores.items())
        
        element = GPOTrajectoryElement(
            loss_fns=loss_fns, scores=scores, grads=grads, instructions=instructions,
            text_grads=text_grads_list, numerical_grads=numerical_summary
        )
        self.trajectory.push(element)
        
        # Debug output when verbosity >= 2
        if self.verbosity >= 2:
            top_k_str = self.trajectory.get_top_k_str()
            print(f"\\n[GPO Debug] Trajectory top-k string length: {len(top_k_str)} chars")
            print(f"[GPO Debug] Trajectory element count: {len(self.trajectory)}/{self.k}")
            print(f"[GPO Debug] Top-k string preview (first 500 chars):")
            print(top_k_str[:500])
            print(f"[GPO Debug] Top-k string preview (last 500 chars):")
            print(top_k_str[-500:])

    def _get_algorithm_state(self) -> Dict[str, Any]:
        """Get GPO algorithm state.

        Returns:
            Dict with trajectory and GPO-specific parameters
        """
        return {
            "trajectory": self.trajectory,
            "k": self.k,
            "warmup_steps": self.warmup_steps,
            "use_warmup": self.use_warmup,
            "initial_step_size": self.initial_step_size,
            "final_step_size": self.final_step_size,
        }


class TextGrad(PromptAlgorithm):
    aliases = ["text-g", "textgrad"]

    loss_computer: Dict[str, Any] = {"name": "textgrad"}
    gradient_computer: Dict[str, Any] = {"name": "textgrad"}
    prompt_optimizer: Dict[str, Any] = {"name": "textgrad"}

    # TextGrad-specific state tracking
    _previous_instructions: Optional[Dict[str, str]] = None

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """
        Get TextGrad-specific context for loss, gradient, and optimizer components.

        Includes:
        - Step number
        - Task list
        - Batch data (optional)
        - Loss functions configuration (textual feedback enabled by default)
        - Previous instructions for context
        """
        # TextGrad always uses textual feedback for all tasks
        loss_functions = {}
        for task in self.tasks:
            loss_functions[task.task_name] = {
                "use_textual": True,
                # TextGrad uses textual feedback by default
            }

        context: Dict[str, Any] = {
            "algorithm": "textgrad",
            "step": step,
            # "tasks": [t.task_name for t in self.tasks],
            "loss_functions": loss_functions,
        }

        if self._previous_instructions is not None:
            context["previous_instructions"] = self._previous_instructions

        if batch is not None:
            context["batch"] = batch

        return context

    def _update_state(
        self, step: int, feedbacks: Dict, gradients: Dict, new_prompt: PromptTemplate
    ):
        """
        Update TextGrad state after each optimization step.

        Args:
            step: Current step number
            feedbacks: Feedbacks from loss computer
            gradients: Gradients from gradient computer
            new_prompt: New prompt template
        """
        # Extract and store the new prompt instructions for next iteration
        instructions = new_prompt.instruction
        self._previous_instructions = instructions

    def _get_algorithm_state(self) -> Dict[str, Any]:
        """Get TextGrad algorithm state.

        Returns:
            Dict with previous instructions
        """
        return {
            "previous_instructions": self._previous_instructions,
        }
