"""GPO algorithm, optimizer, gradient computer, and loss computer."""

from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union

import numpy as np
from morphic import validate
from morphic.imports import optional_dependency
from morphic.typed import format_exception_msg
from pydantic import Field, PrivateAttr

from ..config import promptmoo_config
from ..data_structures import (
    Batch,
    CombinedFeedback,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from ..gradient_computer import GradientComputer
from ..llm_utils import apply_prompt_suffix
from ..metrics import Metric
from ..prompt_algorithm import PromptAlgorithm
from ..prompt_optimizer import PromptOptimizer
from ..prompt_template import PromptTemplate
from ..prompt_trajectory import (
    PromptTrajectory,
    TrajectoryElement,
    _instructions_to_text,
)
from ..types import extract_instructions_json


class GPOTrajectoryElement(TrajectoryElement):
    """GPO trajectory element: ``Prompt: {text} / Score: {score}``.

    Matches the GPO paper's source code (``k_update.py``) format:
    ``Prompt: {text}\\nScore: {score}`` with ``\\n\\n`` between entries.
    Multi-task uses ``Prompt:\\n- task: text\\nScores: task=N, ...``.

    No textual gradients are stored — the paper found that
    reflection in the trajectory hurts performance.
    """

    aliases = ["gpo"]

    def __str__(self) -> str:
        lines: List[str] = []
        if isinstance(self.instructions, str):
            lines.append(f"Prompt: {self.instructions}")
            for task_name, feedbacks in self.numeric_scores.items():
                for fb in feedbacks:
                    lines.append(f"Score: {fb.display_score}")
        else:
            lines.append("Prompt:")
            for task_name, instruction_text in self.instructions.items():
                lines.append(f"- {task_name}: {instruction_text}")
            score_parts: List[str] = []
            for task_name, feedbacks in self.numeric_scores.items():
                for fb in feedbacks:
                    score_parts.append(f"{task_name}={fb.display_score}")
            if len(score_parts) > 0:
                lines.append(f"Scores: {', '.join(score_parts)}")
        return "\n".join(lines)


class GPO(PromptAlgorithm):
    """GPO algorithm implementation (Tang et al., AAAI 2025).

    Faithful to the paper: by default GPO uses only deterministic numeric
    scoring (no Loss LLM, no Gradient LLM).  The optimizer LLM sees the
    trajectory of (prompt, score) pairs, the current prompt template, and
    a cosine-decay edit distance constraint.

    The trajectory stores **all** past (prompt, score) pairs (unbounded).
    At each step the ``k`` most relevant entries are retrieved using the
    ``trajectory_strategy`` and shown to the optimizer in the meta-prompt.

    Trajectory strategies (GPO paper Section 3.1 "Analogical Momentum"):
        ``"relevance"``: Retrieve the k most semantically similar past prompts
            using sentence-transformer embeddings + cosine similarity.  This is
            the paper's winning configuration (+15% over recency).
        ``"importance"``: Retrieve the k highest-scoring past prompts.  This is
            the OPRO-style approach.

    Set ``use_textual_feedback=True`` to enable the non-paper hybrid mode
    that adds LLM-generated textual feedback (Loss LLM) and LLM-generated
    gradients (Gradient LLM).  This is useful for the gradient conflict
    analysis in the PromptMOO paper but is NOT the GPO paper's design.
    """

    aliases = ["gpo"]

    loss_computer: Dict[str, Any] = Field(
        default_factory=lambda: {"name": "task-level"}
    )
    gradient_computer: Dict[str, Any] = Field(default_factory=lambda: {"name": "gpo"})
    prompt_optimizer: Dict[str, Any] = Field(default_factory=lambda: {"name": "gpo"})

    k: int = (
        7  # Paper: "best performance when setting the length of the trajectory to 7"
    )

    warmup_steps: int = 0
    initial_step_size: int = 25
    final_step_size: int = (
        5  # Paper: "reduce to approximately 20% of its maximum value"
    )
    task_losses: Dict[str, str] = Field(default_factory=dict)

    # Paper: "we randomly sample 3 examples from the dataset and fill them
    # into the meta-prompt of the prompt optimizer" (Appendix A)
    num_task_demonstrations: int = 3

    # Paper: "8 candidate prompts generated per step" (Appendix A, line 38)
    num_candidates: int = 8

    # Paper-faithful default: no LLM-generated textual feedback or gradients.
    # Set True to enable the hybrid mode (Loss LLM + Gradient LLM).
    use_textual_feedback: bool = False

    # Paper (Appendix A): "set its temperature to 0 to make the output as
    # deterministic as possible" for task LLM; "set its temperature to 1.0
    # to encourage the generation to be more diverse" for optimizer LLM.
    # Paper (Appendix B, line 102): "performance shows an upward trend as
    # temperature increases, reaching its peak at 1.0."
    task_llm_temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    optimizer_llm_temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    gradient_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    loss_llm_temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

    # Trajectory retrieval strategy (GPO paper Section 3.1).
    # "relevance" = paper-faithful: semantic similarity via sentence-transformer.
    # "importance" = fallback: top-k by score (same as OPRO).
    trajectory_strategy: Literal["relevance", "importance"] = "relevance"

    # Sentence-transformer model for embedding prompt instructions when
    # trajectory_strategy="relevance".  Only loaded when first needed.
    embedding_model_name: str = "mixedbread-ai/mxbai-embed-xsmall-v1"

    # Trajectory order for presentation in the meta-prompt.
    # Paper (Appendix F): trajectory shown ascending (worst first, best last)
    # so the optimizer sees the best examples most recently in context.
    # For "relevance" strategy, elements are sorted by ascending similarity
    # regardless of this setting.
    trajectory_order: Literal["worst_to_best", "best_to_worst"] = "worst_to_best"

    # Trajectory — stores ALL past (prompt, score) pairs (k=None → unbounded).
    _trajectory: Optional[PromptTrajectory] = PrivateAttr(default=None)
    # Lazy-loaded sentence-transformer model (only for trajectory_strategy="relevance").
    _embedding_model: Optional[Any] = PrivateAttr(default=None)

    @property
    def trajectory(self) -> PromptTrajectory:
        if self._trajectory is None:
            metrics = list(dict.fromkeys(self.task_losses.values()))
            self._trajectory = PromptTrajectory(
                k=None,
                order=self.trajectory_order,
                metric_priority=metrics,
            )
        return self._trajectory

    @classmethod
    def pre_initialize(cls, data: dict) -> None:
        """Auto-default batch sizes and force unused LLMs to None.

        The GPO paper evaluates the entire training batch as one unit,
        producing a single aggregate score per task per step.  When the
        user omits these fields (``None``), they are filled from
        ``batch_size`` so the user only needs to specify one batch size.

        The GPO paper uses only a Task LLM and an Optimizer LLM.  There
        is no Loss LLM or Gradient LLM (the paper found that LLM-generated
        reflection hurts performance).  When ``use_textual_feedback`` is
        False (the default), ``loss_llm`` and ``gradient_llm`` are forced
        to None regardless of what the caller passed.
        """
        batch_size = data["batch_size"]
        if batch_size is not None:
            if data.get("loss_batch_size") is None:
                data["loss_batch_size"] = batch_size
            if data.get("gradient_batch_size") is None:
                data["gradient_batch_size"] = batch_size

        if data["use_textual_feedback"] is False:
            data["loss_llm"] = None
            data["gradient_llm"] = None

    def post_initialize(self) -> None:
        """Validate that loss and gradient batch sizes equal the training batch size.

        If the user explicitly passed different values, this catches the mistake.
        """
        if self.loss_batch_size != self.batch_size:
            raise ValueError(
                f"GPO requires loss_batch_size == batch_size "
                f"(got loss_batch_size={self.loss_batch_size}, batch_size={self.batch_size}). "
                f"The GPO paper evaluates the full training batch as one unit, "
                f"producing exactly one aggregate score per task per step."
            )
        if self.gradient_batch_size != self.batch_size:
            raise ValueError(
                f"GPO requires gradient_batch_size == batch_size "
                f"(got gradient_batch_size={self.gradient_batch_size}, batch_size={self.batch_size}). "
                f"The GPO paper evaluates the full training batch as one unit."
            )

    def _build_loss_fn_config(
        self,
        *,
        task_name: str,
        use_textual: bool,
    ) -> Dict[str, Any]:
        """GPO override: inject ``decimals=0`` for integer-rounded scores.

        The GPO paper's meta-prompt (Appendix F) states "The score ranges
        from 0 to 100" — integer scores, matching OPRO's 100-bucket format.
        """
        config = super()._build_loss_fn_config(
            task_name=task_name,
            use_textual=use_textual,
        )
        config["decimals"] = 0
        return config

    def _get_embedding_model(self) -> Any:
        """Lazy-load the sentence-transformer model for similarity retrieval.

        Only called when ``trajectory_strategy="relevance"``.  The model is
        loaded once and reused across all steps.

        Returns:
            A SentenceTransformer model instance.
        """
        if self._embedding_model is None:
            with optional_dependency("sentence_transformers", error="raise"):
                from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _embed_instructions(
        self,
        *,
        instructions: Union[str, Dict[str, str]],
    ) -> "np.ndarray":
        """Embed prompt instructions into an L2-normalized vector.

        Args:
            instructions: Single instruction string or dict mapping
                task_name -> instruction text.

        Returns:
            1-D numpy array (L2-normalized) suitable for cosine similarity
            via dot product.
        """
        text = _instructions_to_text(instructions)
        model = self._get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding

    def _get_algorithm_context(
        self, step: int, batch: Optional[Batch] = None
    ) -> Dict[str, Any]:
        """Get GPO-specific context for loss, gradient, and optimizer components.

        When ``use_textual_feedback`` is False (default, paper-faithful):
        loss uses numeric-only scoring, gradient is a deterministic score
        summary (no LLM call), and the optimizer meta-prompt omits
        the "gradients" section.
        """
        loss_functions = {}
        for task in self.tasks:
            if task.task_name in self.task_losses:
                loss_functions[task.task_name] = self._build_loss_fn_config(
                    task_name=task.task_name,
                    use_textual=self.use_textual_feedback,
                )
            else:
                raise ValueError(f"Unsupported task: {task.task_name}")

        context: Dict[str, Any] = {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.steps,
            "initial_step_size": self.initial_step_size,
            "final_step_size": self.final_step_size,
            "trajectory": self.trajectory,
            "loss_functions": loss_functions,
            "top_k_retrieve": self.k,
            "use_textual_feedback": self.use_textual_feedback,
            "trajectory_strategy": self.trajectory_strategy,
        }

        if self.trajectory_strategy == "relevance":
            context["embed_fn"] = self._embed_instructions
        elif self.trajectory_strategy == "importance":
            pass
        else:
            raise ValueError(
                f"Unknown trajectory_strategy={self.trajectory_strategy!r}. "
                f"Must be 'relevance' or 'importance'."
            )

        task_demonstrations = self._require_task_demonstrations()
        if len(task_demonstrations) > 0:
            context["task_demonstrations"] = task_demonstrations

        if batch is not None:
            context["batch"] = batch

        return context

    def _before_optimize(
        self,
        *,
        step: int,
        feedbacks: Dict,
        gradients: Dict,
        current_prompt: PromptTemplate,
    ) -> None:
        """Push the current prompt's (instruction, score) to the trajectory.

        Same rationale as OPRO: the trajectory must contain the current
        prompt's evaluation before the optimizer builds its meta-prompt.

        When ``trajectory_strategy="relevance"``, the element is embedded
        at push time so ``get_most_similar()`` can retrieve it later.
        """
        use_embeddings = self.trajectory_strategy == "relevance"

        numeric_scores: Dict[str, List[NumericFeedback]] = {}
        for task, feedback_list in feedbacks.items():
            task_fbs = [fb for fb in feedback_list if isinstance(fb, NumericFeedback)]
            if len(task_fbs) > 0:
                numeric_scores[task.task_name] = task_fbs

        element = GPOTrajectoryElement(
            instructions=current_prompt.instruction,
            numeric_scores=numeric_scores,
        )
        embedding = (
            self._embed_instructions(instructions=current_prompt.instruction)
            if use_embeddings
            else None
        )
        self.trajectory.push(element, embedding=embedding)

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

        When ``trajectory_strategy="relevance"``, each element is embedded
        at push time so ``get_most_similar()`` can retrieve by cosine
        similarity later.

        When multi-candidate generation is active (num_candidates > 1),
        ALL candidates and their evaluation scores are pushed, matching
        the GPO paper where all candidates per step enter the history.
        """
        use_embeddings = self.trajectory_strategy == "relevance"

        if all_candidates is not None and all_candidate_scores is not None:
            for cand, cand_scores in zip(all_candidates, all_candidate_scores):
                cand_ns: Dict[str, List[NumericFeedback]] = {
                    task.task_name: fbs for task, fbs in cand_scores.items()
                }
                cand_element = GPOTrajectoryElement(
                    instructions=cand.instruction,
                    numeric_scores=cand_ns,
                )
                cand_embedding = (
                    self._embed_instructions(instructions=cand.instruction)
                    if use_embeddings
                    else None
                )
                self.trajectory.push(cand_element, embedding=cand_embedding)

        if self.verbosity >= 2:
            print(
                f"\n[GPO Debug] Trajectory ({len(self.trajectory)} total, retrieving top {self.k}):"
            )
            print(self.trajectory.get_top_k_str(limit=self.k))

    def _serialize_algorithm_context(self, *, step: int) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "trajectory": self.trajectory.to_serializable_list(limit=self.k),
            "trajectory_k": self.trajectory.k,
            "loss_functions": {
                task.task_name: self._build_loss_fn_config(
                    task_name=task.task_name,
                    use_textual=self.use_textual_feedback,
                )
                for task in self.tasks
                if task.task_name in self.task_losses
            },
            "use_textual_feedback": self.use_textual_feedback,
            "trajectory_strategy": self.trajectory_strategy,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.steps,
            "initial_step_size": self.initial_step_size,
            "final_step_size": self.final_step_size,
            "top_k_retrieve": self.k,
        }
        demos = self._require_task_demonstrations()
        if len(demos) > 0:
            ctx["task_demonstrations"] = [d.model_dump() for d in demos]
        return ctx

    def _serialize_algorithm_state(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory.to_serializable_list(),
            "trajectory_k": self.trajectory.k,
            "trajectory_size": len(self.trajectory),
            "k": self.k,
            "warmup_steps": self.warmup_steps,
            "initial_step_size": self.initial_step_size,
            "final_step_size": self.final_step_size,
        }


class GPOOptimizer(PromptOptimizer):
    """
    GPO-Specific Prompt Optimizer:
    1. Build meta-prompt using trajectory (top-k)
    2. Generate candidate prompts using LLM
    3. Evaluate each candidate
    4. Select best candidate and push to trajectory.
    """

    aliases: ClassVar[List[str]] = ["gpo"]

    def _calculate_step_size(
        self,
        *,
        step: int,
        warmup_steps: int,
        total_steps: int,
        initial_step_size: int,
        final_step_size: int,
    ) -> int:
        """Calculate the edit distance constraint (number of words) for this step.

        Implements the GPO paper's cosine-decay schedule:
        - During warmup (step < warmup_steps): linear ramp from 0 to
          initial_step_size.  warmup_steps=0 means no warmup.
        - After warmup: cosine decay from initial_step_size down to
          final_step_size (paper: "reduce to approximately 20% of its
          maximum value").

        Returns an absolute word count (int), not a percentage.
        """
        if warmup_steps > 0 and step < warmup_steps:
            progress: float = (step + 1) / warmup_steps
            current_step_size: float = initial_step_size * progress
        else:
            decay_start: int = warmup_steps
            decay_length: int = max(1, total_steps - warmup_steps)
            progress: float = min(1.0, (step - decay_start) / decay_length)
            cosine_decay: float = 0.5 * (1 + np.cos(np.pi * progress))
            current_step_size: float = (
                final_step_size + (initial_step_size - final_step_size) * cosine_decay
            )

        return max(1, round(current_step_size))

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
        use_textual_feedback: bool,
        optimizer_task_strategy: Optional[str] = None,
        warmup_steps: int,
        total_steps: int,
        initial_step_size: int,
        final_step_size: int,
        top_k_retrieve: int,
        trajectory_strategy: str,
        embed_fn: Optional[Callable] = None,
        loss_functions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Build GPO meta-prompt following the paper's source code structure.

        Adapted from ``RUCAIBox/GPO/src/optimization/utilize_gradient/
        generate_without_gradient.py`` (the paper's winning configuration:
        generation-based refinement without reflection gradients).

        Structure (matches the paper's ``gen_util_gradient_text``):
        1. Opening instruction: "write instructions to replace <Instruction_*>"
        2. Trajectory: previous (instruction, score) pairs
        3. "The current instructions are:" (matches paper's "The current prompt is:")
        4. Template with ``<Instruction_*>`` placeholders showing WHERE instructions go
        5. Task exemplars with expected output (compact — no skeleton repetition)
        6. [Hybrid mode only] Gradient/improvement suggestions
        7. Generation instruction + edit distance constraint
        8. JSON output format (multi-task adaptation of paper's ``<START>/<END>``)
        """
        if batch is None:
            raise ValueError("[GPOOptimizer] Missing required 'batch'")
        if trajectory is None:
            raise ValueError("[GPOOptimizer] Missing required 'trajectory'")

        step: int = batch.step
        step_0idx: int = step - 1

        max_word_changes: int = self._calculate_step_size(
            step=step_0idx,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            initial_step_size=initial_step_size,
            final_step_size=final_step_size,
        )

        # --- Retrieve trajectory elements ---
        if trajectory_strategy == "relevance":
            if embed_fn is None:
                raise ValueError(
                    "[GPOOptimizer] trajectory_strategy='relevance' requires "
                    "'embed_fn' in kwargs (set by GPO._get_algorithm_context)."
                )
            query_embedding: "np.ndarray" = embed_fn(
                instructions=current_prompt.instruction
            )
            top_k_elements: List[TrajectoryElement] = trajectory.get_most_similar(
                query_embedding=query_embedding,
                limit=top_k_retrieve,
            )
        elif trajectory_strategy == "importance":
            top_k_elements: List[TrajectoryElement] = trajectory.get_topk(
                limit=top_k_retrieve
            )
        else:
            raise ValueError(
                f"[GPOOptimizer] Unknown trajectory_strategy={trajectory_strategy!r}. "
                f"Must be 'relevance' or 'importance'."
            )

        top_k_text: str = "\n\n".join([str(e) for e in top_k_elements])

        # --- Dynamic score range from Metric ---
        score_range_desc: str = self._build_score_range_description(
            tasks=tasks,
            loss_functions=loss_functions,
        )

        # --- Placeholder names for each task ---
        task_names: List[str] = [t.task_name for t in tasks]
        placeholders: List[str] = [f"<Instruction_{tn}>" for tn in task_names]
        placeholders_str: str = ", ".join(placeholders)

        # --- Current instructions section (matches paper's "The current prompt is:") ---
        current_instr_lines: List[str] = []
        for task in tasks:
            instr_text: str = current_prompt.instruction[task.task_name]
            current_instr_lines.append(f"- {task.task_name}: {instr_text}")
        current_instructions_text: str = "\n".join(current_instr_lines)

        # --- Template with <Instruction_*> placeholders (paper-faithful: shows
        # WHERE the instruction goes, but uses placeholders instead of actual text) ---
        placeholder_instr_lines: List[str] = [
            f"- {tn}: <Instruction_{tn}>" for tn in task_names
        ]
        placeholder_instr_str: str = "\n".join(placeholder_instr_lines)
        sample_placeholder_lines: List[str] = [
            f"{label}: ({label.lower()} will be inserted here)"
            for label in current_prompt.input_col_labels.values()
        ]
        sample_placeholder_str: str = "\n".join(sample_placeholder_lines)
        template_section: str = (
            "The task LLM receives a prompt with this structure, where each "
            f"{placeholders_str} is replaced with your instruction and the "
            "sample data is filled in:\n"
            "---\n"
            f"{current_prompt.skeleton.strip()}\n\n"
            f"## Instructions:\n{placeholder_instr_str}\n\n"
            f"## Sample:\n{sample_placeholder_str}\n\n"
            "## Response:\n"
            "---\n"
        )

        # --- Task exemplars ---
        task_demos: List[DatasetSample] = (
            task_demonstrations if task_demonstrations is not None else []
        )
        demos_section: str = self._format_task_demonstrations(
            demonstrations=task_demos,
            tasks=tasks,
            input_col_labels=current_prompt.input_col_labels,
        )

        # --- Gradient section (hybrid mode only) ---
        has_gradients: bool = use_textual_feedback and any(
            len(grad_list) > 0 for grad_list in gradients.values()
        )
        gradient_section: str = ""
        if has_gradients:
            gradient_text: str = self._format_gradients_for_meta_prompt(gradients)
            gradient_section = f"""
Below are the **Gradients / Improvement suggestions** from the last step:
{gradient_text}

Your primary goal is to incorporate the improvement suggestions above. You may make the instruction shorter, longer, or restructure it entirely.
Preserve aspects that are working well, but remove or rewrite aspects that are causing errors.
"""

        # --- JSON output format ---
        example_instruction: str = ",\n".join(
            [
                f'        "{task.task_name}": "improved instruction for {task.task_name}"'
                for task in tasks
            ]
        )

        # --- Generation instruction ---
        generation_instruction: str = (
            "Carefully analyze the previous instructions and their scores, "
            f"and write new improved instructions to replace each {placeholders_str} "
            "in the ## Instructions section above."
        )

        meta_prompt: str = f"""Your task is to write instructions to replace {placeholders_str}.

Below are the previous instructions with their scores. {score_range_desc}

{top_k_text}

The current instructions are:
{current_instructions_text}

{template_section}
{demos_section}{gradient_section}{generation_instruction}
You are allowed to change up to {max_word_changes} words in each current instruction.

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
        """Parse GPO meta-prompt response.

        Args:
            response: Raw LLM response
            tasks: List of tasks
            **kwargs: Unused

        Returns:
            Dict mapping task names to new instructions

        Raises:
            ValueError: If parsing fails
        """
        task_names: List[str] = [t.task_name for t in tasks]
        result: Dict[str, str] = extract_instructions_json(
            response, task_names=task_names
        )
        for task in tasks:
            if task.task_name not in result:
                raise ValueError(
                    f"GPO optimizer LLM did not produce instruction for task "
                    f"{task.task_name!r}. Parsed keys: {list(result.keys())}. "
                    f"Expected keys: {task_names}."
                )
        return result

    def _format_gradients_for_meta_prompt(
        self, gradients: Dict[Task, List[TextGradient]]
    ) -> str:
        """Format gradients into readable text for meta-prompt."""
        formatted: List[str] = []
        for task, grad_list in gradients.items():
            gtext: str = " ".join(g.gradient_text for g in grad_list)
            formatted.append(f"Task {task.task_name}: {gtext}")
        return "\n".join(formatted)

    @staticmethod
    def _build_score_range_description(
        *,
        tasks: List[Task],
        loss_functions: Dict[str, Dict[str, Any]],
    ) -> str:
        """Build a dynamic score range description from the Metric registry.

        Looks up the metric name for each task from ``loss_functions``
        (which maps task_name -> config dict with a ``"metric"`` key),
        resolves the ``Metric`` subclass, and reads ``display_score_range``
        and ``display_direction``.

        When all tasks share the same range and direction, produces a
        single sentence.  Otherwise, lists per-task ranges.

        Args:
            tasks: List of tasks.
            loss_functions: Per-task loss config dicts from
                ``_get_algorithm_context``.

        Returns:
            Human-readable sentence for the meta-prompt.
        """
        ranges = set()
        directions = set()
        for task in tasks:
            if task.task_name not in loss_functions:
                raise ValueError(
                    f"_build_score_range_description: task {task.task_name!r} not found "
                    f"in loss_functions. Available: {list(loss_functions.keys())}"
                )
            task_config: Dict[str, Any] = loss_functions[task.task_name]
            if "metric" not in task_config:
                raise ValueError(
                    f"_build_score_range_description: loss_functions[{task.task_name!r}] "
                    f"must contain 'metric' key, but only found keys: {list(task_config.keys())}"
                )
            metric_name: str = task_config["metric"]
            try:
                metric_cls = Metric.get_subclass(metric_name)
            except (KeyError, ValueError):
                raise ValueError(
                    f"_build_score_range_description: unknown metric {metric_name!r} "
                    f"for task {task.task_name!r}. Register it as a Metric subclass "
                    f"in metrics.py."
                )
            low, high = metric_cls.display_score_range
            direction = metric_cls.display_direction
            ranges.add((low, high))
            directions.add(direction)

        def _format_bound(val: float) -> str:
            if val == float("inf"):
                return "∞"
            if val == float(int(val)):
                return str(int(val))
            return str(val)

        if len(ranges) == 1 and len(directions) == 1:
            low, high = ranges.pop()
            direction = directions.pop()
            return (
                f"The score ranges from {_format_bound(low)} to "
                f"{_format_bound(high)}, and {direction} scores indicate "
                f"better quality."
            )
        parts: List[str] = []
        for task in tasks:
            task_config_2: Dict[str, Any] = loss_functions[task.task_name]
            metric_name = task_config_2["metric"]
            try:
                metric_cls = Metric.get_subclass(metric_name)
            except (KeyError, ValueError):
                raise ValueError(
                    f"_build_score_range_description: unknown metric {metric_name!r} "
                    f"for task {task.task_name!r}. Register it as a Metric subclass "
                    f"in metrics.py."
                )
            low, high = metric_cls.display_score_range
            direction = metric_cls.display_direction
            parts.append(
                f"{task.task_name}: {_format_bound(low)}-{_format_bound(high)} "
                f"({direction} is better)"
            )
        return "Score ranges: " + "; ".join(parts) + "."

    @staticmethod
    def _format_task_demonstrations(
        *,
        demonstrations: List[DatasetSample],
        tasks: List[Task],
        input_col_labels: Dict[str, str],
    ) -> str:
        """Format task exemplars following the GPO paper (Appendix F).

        The phrasing defines the success criterion: output matches
        expected output.  The prompt template is shown separately
        (in ``create_meta_prompt``) so it's always visible even when
        there are no exemplars.

        Returns an empty string when no demonstrations are available.
        """
        if len(demonstrations) == 0:
            return ""

        lines: List[str] = [
            "The following exemplars show how the task LLM uses your "
            "instructions. You replace each <Instruction_*> in the prompt "
            "template above with your new instruction, then the task LLM "
            "reads the input and gives an output. The output is correct if "
            "it matches the expected output.",
            "",
        ]
        for i, demo in enumerate(demonstrations, 1):
            lines.append(f"Exemplar {i}:")
            lines.append("  Input:")
            for col, val in demo.inputs.items():
                if col not in input_col_labels:
                    raise ValueError(
                        f"GPO._format_task_demonstrations: column {col!r} has no label in "
                        f"input_col_labels. Available: {list(input_col_labels.keys())}."
                    )
                label: str = input_col_labels[col]
                lines.append(f"    {label}: {val}")
            gt_parts: List[str] = []
            for task in tasks:
                if task.task_name not in demo.ground_truths:
                    raise ValueError(
                        f"GPO._format_task_demonstrations: demo {demo.sample_id!r} "
                        f"has no ground truth for task {task.task_name!r}. "
                        f"Available ground truth keys: {list(demo.ground_truths.keys())}. "
                        f"Expected keys for all tasks: {[t.task_name for t in tasks]}."
                    )
                gt_val = demo.ground_truths[task.task_name]
                gt_parts.append(f"{task.task_name}: {gt_val}")
            lines.append(f"  Expected output: {', '.join(gt_parts)}")
            lines.append("")

        return "\n".join(lines) + "\n"


class GPOGradientComputer(GradientComputer):
    """GPO Gradient Computer (Tang et al., AAAI 2025).

    Paper-faithful mode (default): no LLM call.  Produces a deterministic
    score summary from numeric feedbacks, identical to OPROGradientComputer.
    The GPO paper found that LLM-generated "reflection" gradients hurt
    performance compared to trajectory-only optimization.

    Hybrid mode (when ``use_textual_feedback=True``): combines numeric and
    textual feedbacks and uses the Gradient LLM to synthesize per-task
    improvement suggestions.  This enables the gradient conflict analysis
    from the PromptMOO paper.
    """

    aliases = ["gpo"]

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
        use_textual_feedback: bool,
        predictions: Optional[List[PredictionResult]] = None,
        ground_truths: Optional[List[DatasetSample]] = None,
        input_col_labels: Optional[Dict[str, str]] = None,
        gradient_task_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[Task, List[TextGradient]]:
        has_textual_feedbacks = any(
            isinstance(fb, TextualFeedback) for fbs in feedbacks.values() for fb in fbs
        )

        if not use_textual_feedback or not has_textual_feedbacks:
            return self._compute_numeric_only(feedbacks=feedbacks)

        return self._compute_with_llm(
            feedbacks=feedbacks,
            prompt_template=prompt_template,
            tasks=tasks,
            llm_pool=llm_pool,
            gradient_batch_size=gradient_batch_size,
            verbosity=verbosity,
        )

    def _compute_numeric_only(
        self,
        *,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
    ) -> Dict[Task, List[TextGradient]]:
        """Deterministic score summary — no LLM call.  Matches the GPO paper.

        The GPO paper has no gradient step at all; the optimizer LLM infers
        improvement direction entirely from the (prompt, score) trajectory.
        This method exists to satisfy the 4-step pipeline interface: it
        produces a lightweight, non-LLM TextGradient that is logged for
        observability but never included in the optimizer meta-prompt (the
        GPOOptimizer checks ``has_llm_gradients`` and omits the gradient
        section when all gradients have ``gradient_prompt is None``).

        Score formatting uses ``NumericFeedback.display_score`` which
        delegates to the ``Metric`` instance's ``decimals`` setting.  GPO
        injects ``decimals=0`` via ``_build_loss_fn_config``, producing
        integer-rounded 0-100 scores consistent with the trajectory.
        """
        result: Dict[Task, List[TextGradient]] = {}
        for task, feedback_list in feedbacks.items():
            scores: List[str] = []
            for fb in feedback_list:
                if isinstance(fb, NumericFeedback):
                    scores.append(f"{fb.metric_name}: {fb.display_score}")

            if len(scores) > 0:
                gradient: TextGradient = TextGradient(
                    task_name=task.task_name,
                    gradient_text=f"Scores: {', '.join(scores)}",
                    based_on_feedbacks=[],
                    gradient_prompt=None,
                )
                result[task] = [gradient]
            else:
                result[task] = []
        return result

    def _compute_with_llm(
        self,
        *,
        feedbacks: Dict[Task, List[Union[NumericFeedback, TextualFeedback]]],
        prompt_template: PromptTemplate,
        tasks: List[Task],
        llm_pool: Any,
        gradient_batch_size: int,
        verbosity: int,
    ) -> Dict[Task, List[TextGradient]]:
        """Hybrid mode: use LLM to generate textual gradients from combined feedback."""
        result: Dict[Task, List[TextGradient]] = {}

        for task, feedback_list in feedbacks.items():
            if len(feedback_list) == 0:
                result[task] = []
                continue

            combined_feedbacks: List[CombinedFeedback] = (
                self._combine_feedbacks_by_samples(feedbacks=feedback_list, task=task)
            )
            feedback_batches: List[List[CombinedFeedback]] = (
                self._batch_combined_feedbacks(
                    feedbacks=combined_feedbacks, batch_size=gradient_batch_size
                )
            )

            prompts: List[str] = []
            for fb_batch in feedback_batches:
                grad_prompt: str = self._build_gradient_prompt(
                    feedbacks=fb_batch,
                    task=task,
                    prompt_template=prompt_template,
                    tasks=tasks,
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
                        for combined_fb in fb_batch:
                            if len(combined_fb.aggregated_from_samples) > 0:
                                feedback_ids.extend(combined_fb.aggregated_from_samples)

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
            sample_key: frozenset = frozenset(fb.aggregated_from_samples)

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
        combined: List[CombinedFeedback] = []
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

        batches: List[List[CombinedFeedback]] = []
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

        Groups numeric and textual feedback by sample batch so the gradient LLM
        can see which metrics and critiques correspond to which examples.

        Args:
            feedbacks: Batch of CombinedFeedback objects (numeric + textual)
            task: Task object
            prompt_template: Current prompt template
            tasks: List of all tasks

        Returns:
            A string prompt for gradient generation
        """
        feedback_sections: List[str] = []
        for i, combined_fb in enumerate(feedbacks, 1):
            section_lines: List[str] = [
                f"Batch {i} (samples: {', '.join(combined_fb.aggregated_from_samples)}):"
            ]
            for nf in combined_fb.numeric_feedbacks:
                section_lines.append(
                    f"  - {nf.metric_name}: {nf.value:.4f} ({nf.optimization_direction})"
                )
            for tf in combined_fb.textual_feedbacks:
                section_lines.append(f"  - Textual: {tf.feedback_text}")
            if (
                len(combined_fb.numeric_feedbacks) == 0
                and len(combined_fb.textual_feedbacks) == 0
            ):
                section_lines.append("  (no feedback)")
            feedback_sections.append("\n".join(section_lines))

        prompt: str = f"""You are optimizing the task: {task.task_name}

Current prompt template:
{prompt_template.render_instructions()}

Feedback on performance (grouped by sample batch):
{chr(10).join(feedback_sections) if len(feedback_sections) > 0 else "None"}

Based on this feedback, suggest how to update or improve the instructions for this task.
Do not give abstract advice (e.g. "add more detail" or "consider providing examples").
Instead, write the specific sentences or phrases that should be added to, removed from, or changed in the current instruction to fix the identified issues.
Return concrete, ready-to-use instruction text in plain text.
"""
        return prompt
