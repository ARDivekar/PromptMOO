"""
Prompt Optimizer: Transforms gradients into updated prompt.

This is Step 4 of the optimization pipeline.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from morphic import Registry, Typed, validate

from .config import promptmoo_config
from .data_structures import (
    Batch,
    DatasetSample,
    OptimizerResult,
    PredictionResult,
    Task,
    TextGradient,
)
from .llm_utils import apply_prompt_suffix
from .prompt_template import PromptTemplate
from .prompt_trajectory import PromptTrajectory
from .types import (
    extract_instructions_json,
)

# Export validator for use when creating LLM pools
__all__ = [
    "PromptOptimizer",
    "LLMBasedOptimizer",
    "OPROOptimizer",
    "GPOOptimizer",
    "TextGradOptimizer",
    "validate_optimizer_response",
]


def validate_optimizer_response(result: str, **context) -> bool:
    """Validator for optimizer responses - ensures valid JSON with instructions.

    Args:
        result: Raw LLM response text
        **context: Additional context (unused)

    Returns:
        True if response contains valid JSON with instructions, False otherwise
    """
    try:
        start: int = result.find("{")
        end: int = result.rfind("}") + 1
        if start == -1 or end == 0:
            return False
        json_str: str = result[start:end]
        parsed: Any = json.loads(json_str)
        # Must have instructions key or be a dict of task names to instructions
        if isinstance(parsed, dict):
            if "instructions" in parsed or "instruction" in parsed:
                return True
            # Or it's a dict mapping task names to instructions
            if len(parsed) > 0 and all(isinstance(v, str) for v in parsed.values()):
                return True
        return False
    except (json.JSONDecodeError, ValueError):
        return False


class PromptOptimizer(Typed, Registry, ABC):
    """Transforms gradients into updated prompt.

    This is a transformer component that generates new prompts from gradients.

    Subclasses must implement:
    - create_meta_prompt(): Build the meta-prompt for the LLM
    - parse_meta_prompt_response(): Parse the LLM response into instructions
    """

    _allow_subclass_override: ClassVar[bool] = True

    @abstractmethod
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
        """Create meta-prompt for prompt optimization.

        Args:
            gradients: Dict of gradients from gradient computer
            current_prompt: Current prompt template
            tasks: List of tasks
            trajectory: Optimization trajectory (PromptTrajectory) for OPRO/GPO
            batch: Current batch info for GPO step-size scheduling
            task_demonstrations: Example (input, ground_truth) pairs for the meta-prompt
            input_col_labels: Mapping from column names to display labels
            predictions: Forward-pass predictions for TextGrad context blocks
            ground_truths: Ground truth samples for TextGrad context blocks
            optimizer_task_strategy: "combine_all_tasks" or "separate_tasks" (TextGrad)
            **kwargs: Algorithm-specific context (e.g. GPO use_textual_feedback
                and cosine-decay params, PE2 conversation state)

        Returns:
            Meta-prompt string for the LLM
        """
        pass

    @abstractmethod
    def parse_meta_prompt_response(
        self, *, response: str, tasks: List[Task], **kwargs: Any
    ) -> Dict[str, str]:
        """Parse LLM response into task instructions.

        Args:
            response: Raw LLM response text
            tasks: List of tasks
            **kwargs: Algorithm-specific context

        Returns:
            Dict mapping task names to new instructions

        Raises:
            ValueError: If parsing fails
        """
        pass

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
        """Generate new prompt from gradients.

        This default implementation:
        1. Creates meta-prompt using create_meta_prompt()
        2. Calls LLM
        3. Parses response using parse_meta_prompt_response()
        4. Creates new PromptTemplate

        Args:
            gradients: Dict of gradients from gradient computer
            current_prompt: Current prompt template
            tasks: List of tasks
            llm_pool: LLM pool for optimization
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)
            trajectory: Optimization trajectory (PromptTrajectory) for OPRO/GPO
            batch: Current batch info for GPO step-size scheduling
            task_demonstrations: Example (input, ground_truth) pairs for the meta-prompt
            input_col_labels: Mapping from column names to display labels
            predictions: Forward-pass predictions for TextGrad context blocks
            ground_truths: Ground truth samples for TextGrad context blocks
            optimizer_task_strategy: "combine_all_tasks" or "separate_tasks" (TextGrad)
            **kwargs: Algorithm-specific context (e.g. GPO use_textual_feedback)

        Returns:
            OptimizerResult containing new PromptTemplate, meta_prompt, and raw_response
        """
        # Build meta-prompt
        meta_prompt: str = self.create_meta_prompt(
            gradients=gradients,
            current_prompt=current_prompt,
            tasks=tasks,
            trajectory=trajectory,
            batch=batch,
            task_demonstrations=task_demonstrations,
            input_col_labels=input_col_labels,
            predictions=predictions,
            ground_truths=ground_truths,
            optimizer_task_strategy=optimizer_task_strategy,
            **kwargs,
        )

        # Call optimizer LLM with structural validator for retry
        task_names: List[str] = [t.task_name for t in tasks]

        def _optimizer_validator(response_text: str) -> Dict[str, str]:
            """Parse and validate the response, returning the instructions dict.

            SlowBurnLLM's ``validator`` uses the return value as the result
            when it is not ``False``.  Returning the parsed dict directly
            avoids a redundant second parse and ensures the result type is
            ``Dict[str, str]`` rather than a raw string or bool.

            Returns:
                Parsed instructions dict (triggers success / stops retry).

            Raises:
                ValueError: If parsing fails or required tasks are missing
                    (triggers retry via SlowBurnLLM's retry_on=[ValueError]).
            """
            parsed: Dict[str, str] = self.parse_meta_prompt_response(
                response=response_text, tasks=tasks, **kwargs
            )
            if not isinstance(parsed, dict) or len(parsed) == 0:
                raise ValueError("Parsed instructions dict is empty")
            if len(set(task_names) & set(parsed.keys())) == 0:
                raise ValueError(
                    f"No expected task names found in parsed instructions. "
                    f"Expected: {task_names}, got: {list(parsed.keys())}"
                )
            return parsed

        prompts_to_send: List[str] = apply_prompt_suffix([meta_prompt], llm_pool)
        responses: List[Any] = llm_pool.call_llm_batch(
            prompts=prompts_to_send,
            verbosity=verbosity,
            validator=_optimizer_validator,
        ).result(timeout=promptmoo_config.defaults.batch_invocation_timeout)
        if len(responses) == 0:
            raise ValueError(f"{self.__class__.__name__}: No responses from LLM")
        new_instructions: Dict[str, str] = responses[0]

        if len(new_instructions) == 0:
            raise ValueError(
                f"{self.__class__.__name__}: Parsed instructions dict is empty"
            )

        # Update tasks with new instructions
        updated_tasks: List[Task] = []
        for task in tasks:
            new_instr: Optional[str] = new_instructions.get(task.task_name)
            if new_instr is None:
                raise ValueError(
                    f"{self.__class__.__name__}: optimizer LLM did not produce "
                    f"instruction for task '{task.task_name}'. "
                    f"Received instructions for: {list(new_instructions.keys())}"
                )
            updated_tasks.append(
                Task(
                    task_name=task.task_name,
                    task_description=task.task_description,
                    task_instruction=new_instr,
                    gt_col=task.gt_col,
                )
            )

        # Create new prompt template
        new_prompt: PromptTemplate = PromptTemplate(
            skeleton=current_prompt.skeleton,
            instruction=new_instructions,
            tasks=updated_tasks,
            input_col_labels=current_prompt.input_col_labels,
        )

        return OptimizerResult(
            new_prompt=new_prompt,
            meta_prompt=meta_prompt,
            raw_response=str(new_instructions),
        )


class LLMBasedOptimizer(PromptOptimizer):
    """Use LLM to generate improved prompt."""

    aliases: ClassVar[List[str]] = ["llm-based", "meta-prompt"]

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
        """Build meta-prompt for optimization.

        Args:
            gradients: Gradients to incorporate
            current_prompt: Current prompt template
            tasks: List of tasks
            **kwargs: Unused

        Returns:
            Meta-prompt string
        """
        prompt: str = """You are a meta-optimizer that improves prompts based on feedback.

Current prompt template:
"""
        prompt += current_prompt.render_instructions()
        prompt += "\n\nImprovement suggestions:\n"

        for task, grad_list in gradients.items():
            prompt += f"\nFor task '{task.task_name}':\n"
            for grad in grad_list:
                prompt += f"- {grad.gradient_text}\n"

        task_names: List[str] = [t.task_name for t in tasks]
        prompt += f"""
Based on these suggestions, generate improved instructions for each task.

Return ONLY a valid JSON object in this format:
{{
  "instructions": {{
    "{task_names[0] if len(task_names) > 0 else "task1"}": "improved instruction",
    ...
  }}
}}

Use these exact task names: {", ".join(task_names)}
"""
        return prompt

    @validate
    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Parse instructions from LLM response."""
        task_names: List[str] = [t.task_name for t in tasks]
        return extract_instructions_json(response, task_names=task_names)
