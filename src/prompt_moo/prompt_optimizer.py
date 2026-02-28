"""
Prompt Optimizer: Transforms gradients into updated prompt.

This is Step 4 of the optimization pipeline.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from morphic import Registry, Typed
from morphic.typed import format_exception_msg

from .data_structures import Batch, OptimizerResult, Task, TextGradient
from .llm_workers import LLM, BATCH_INVOCATION_TIMEOUT
from .prompt_template_utils import PromptTemplate
from .prompt_trajectory import PromptTrajectory

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
        start = result.find("{")
        end = result.rfind("}") + 1
        if start == -1 or end == 0:
            return False
        json_str = result[start:end]
        parsed = json.loads(json_str)
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
        **kwargs: Dict[str, Any],
    ) -> str:
        """Create meta-prompt for prompt optimization.

        Args:
            gradients: Dict of gradients from gradient computer
            current_prompt: Current prompt template
            tasks: List of tasks
            **kwargs: Algorithm-specific context

        Returns:
            Meta-prompt string for the LLM
        """
        pass

    @abstractmethod
    def parse_meta_prompt_response(
        self, *, response: str, tasks: List[Task], **kwargs: Dict[str, Any]
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

    def optimize(
        self,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        llm_pool: LLM,
        verbosity: int = 1,
        **kwargs: Dict[str, Any],
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
            **kwargs: Algorithm-specific context

        Returns:
            OptimizerResult containing new PromptTemplate, meta_prompt, and raw_response
        """
        # Build meta-prompt
        meta_prompt = self.create_meta_prompt(
            gradients=gradients,
            current_prompt=current_prompt,
            tasks=tasks,
            **kwargs,
        )

        # Call optimizer LLM
        responses = llm_pool.call_llm_batch(
            prompts=[meta_prompt], verbosity=verbosity
        ).result(timeout=BATCH_INVOCATION_TIMEOUT)
        if len(responses) == 0:
            raise ValueError(f"{self.__class__.__name__}: No responses from LLM")
        response = responses[0]

        # Parse response into new instructions with retry logic
        max_retries = 5
        last_exception = None
        new_instructions = None

        for attempt in range(max_retries):
            try:
                new_instructions = self.parse_meta_prompt_response(
                    response=response, tasks=tasks, **kwargs
                )
                break  # Success, exit retry loop
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(0.01)  # Wait before retrying

        # If all retries failed, raise the last exception
        if new_instructions is None:
            raise ValueError(
                f"{self.__class__.__name__}: Failed to parse response after {max_retries} attempts. "
                f"Last error: {last_exception}"
            )

        if len(new_instructions) == 0:
            raise ValueError(
                f"{self.__class__.__name__}: Failed to parse any instructions from response: {response}"
            )

        # Update tasks with new instructions
        updated_tasks = []
        for task in tasks:
            updated_tasks.append(
                Task(
                    task_name=task.task_name,
                    task_description=task.task_description,
                    task_instruction=new_instructions.get(
                        task.task_name, task.task_instruction
                    ),
                    gt_col=task.gt_col,  # Preserve ground truth column mapping
                )
            )

        # Create new prompt template
        new_prompt = PromptTemplate.of(
            "multi",
            skeleton=current_prompt.skeleton,
            instruction=new_instructions,
            tasks=updated_tasks,
        )

        # Return result with LLM interaction details for observability
        return OptimizerResult(
            new_prompt=new_prompt,
            meta_prompt=meta_prompt,
            raw_response=response,
        )


class LLMBasedOptimizer(PromptOptimizer):
    """Use LLM to generate improved prompt."""

    aliases: ClassVar[List[str]] = ["llm-based", "meta-prompt"]

    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
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
        prompt = """You are a meta-optimizer that improves prompts based on feedback.

Current prompt instructions:
"""
        prompt += current_prompt.to_str()
        prompt += "\n\nImprovement suggestions:\n"

        for task, grad_list in gradients.items():
            prompt += f"\nFor task '{task.task_name}':\n"
            for grad in grad_list:
                prompt += f"- {grad.gradient_text}\n"

        task_names = [t.task_name for t in tasks]
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

    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Parse instructions from LLM response.

        Args:
            response: Raw LLM response
            tasks: List of tasks
            **kwargs: Unused

        Returns:
            Dict mapping task names to new instructions

        Raises:
            ValueError: If parsing fails
        """
        json_str = response.strip().replace("{{", "{").replace("}}", "}")
        start = json_str.find("{")
        end = json_str.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response:\n{response}")

        json_str = (
            json_str[start:end]
            .replace("\n", " ")
            .strip()
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
        )
        try:
            parsed: Dict[str, Any] = json.loads(json_str)

            # Handle different response formats
            if "instructions" in parsed and isinstance(parsed["instructions"], dict):
                return parsed["instructions"]
            elif isinstance(parsed, dict):
                # Assume top-level keys are task names
                return parsed
            else:
                raise ValueError(f"Invalid response format: {response}")
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON: {format_exception_msg(e)}. Response:\n{response}\nExtracted string:\n{json_str}"
            )


class OPROOptimizer(PromptOptimizer):
    """OPRO-specific: Use top-k trajectory."""

    aliases: ClassVar[List[str]] = ["opro"]

    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Create meta-prompt using OPRO's top-k trajectory approach.

        Args:
            gradients: Gradients (used for scores in OPRO)
            current_prompt: Current prompt template
            tasks: List of tasks
            **kwargs: Must contain 'trajectory' for top-k tracking

        Returns:
            Meta-prompt string
        """
        # Get trajectory from kwargs
        trajectory: Optional[PromptTrajectory] = kwargs.get("trajectory")

        # Build meta-prompt with top-k
        top_k_str: str = ""
        if trajectory is not None and len(trajectory) > 0:
            top_k_str = trajectory.get_top_k_str()

        task_names: List[str] = [t.task_name for t in tasks]

        meta_prompt = f"""Generate improved instructions in JSON format, based on the previous performances:

{top_k_str}

Task names: {", ".join(task_names)}

Return ONLY a valid JSON object in this format:
{{
  "instructions": {{
    "{task_names[0]}": "improved instruction",
    ...
  }}
}}

DO NOT include any explanations or text outside the JSON.
"""
        return meta_prompt

    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Parse JSON response from LLM.

        Args:
            response: Raw LLM response
            tasks: List of tasks
            **kwargs: Unused

        Returns:
            Dict of instructions

        Raises:
            ValueError: If parsing fails
        """
        json_str = response.strip().replace("{{", "{").replace("}}", "}")
        start = json_str.find("{")
        end = json_str.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response:\n{response}")

        json_str = (
            json_str[start:end]
            .replace("\n", " ")
            .strip()
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
        )
        try:
            parsed: Dict[str, Any] = json.loads(json_str)
            if "instructions" in parsed and isinstance(parsed["instructions"], dict):
                return parsed["instructions"]
            elif isinstance(parsed, dict):
                return parsed
            else:
                raise ValueError(f"Invalid response format: {response}")
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON: {format_exception_msg(e)}. Response:\n{response}"
            )


class GPOOptimizer(PromptOptimizer):
    """
    GPO-Specific Prompt Optimizer:
    1. Build meta-prompt using trajectory (top-k)
    2. Generate candidate prompts using LLM
    3. Evaluate each candidate
    4. Select best candidate and push to trajectory.
    """

    aliases: ClassVar[List[str]] = ["gpo"]

    def _calculate_step_size(self, step: int, **kwargs: Dict[str, Any]) -> float:
        use_warmup = kwargs.get("use_warmup", True)
        warmup_steps = kwargs.get("warmup_steps", 3)
        total_steps = kwargs.get("total_steps", 10)
        initial_step_size = kwargs.get("initial_step_size", 25)
        final_step_size = kwargs.get("final_step_size", 25)

        if use_warmup and step < warmup_steps:
            progress = step / max(1, warmup_steps)
            current_step_size = initial_step_size * progress

        else:
            progress = min(
                1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            current_step_size = (
                final_step_size + (initial_step_size - final_step_size) * cosine_decay
            )

        return round(current_step_size, 2)

    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Build GPO meta-prompt with trajectory and step-size scheduling.

        Args:
            gradients: Gradients from gradient computer
            current_prompt: Current prompt template
            tasks: List of tasks
            **kwargs: Must contain 'batch' and 'trajectory'

        Returns:
            Meta-prompt string

        Raises:
            ValueError: If 'batch' is missing from kwargs
        """
        batch: Optional[Batch] = kwargs.get("batch")
        if batch is None:
            raise ValueError("[GPOOptimizer] Missing required 'batch' in kwargs")

        trajectory: Optional[PromptTrajectory] = kwargs.get("trajectory")
        step: int = batch.step

        ## Cosine decay update percentage
        update_percentage = self._calculate_step_size(step=step, **kwargs)

        ## Build Meta prompt
        top_k_elements = trajectory.get_topk() if trajectory else []
        top_k_text = trajectory.get_top_k_str() if trajectory else "N/A"
        gradient_text = self._format_gradients_for_meta_prompt(gradients)

        # Example instruction placeholders
        example_instruction = ",\n".join(
            [
                f'        "{task.task_name}": "Improved instruction for {task.task_name} based on all the things provided to you in this prompt."'
                for task in tasks
            ]
        )

        meta_prompt = f"""
You are an expert prompt engineer. You are optimizing a task prompt for multiple tasks.

Below is the **Top-{len(top_k_elements)} Trajectory** (past best prompts and their scores):
{top_k_text}

Below are the **Gradients / Improvement suggestions** from the last step:
{gradient_text}

Update only about **{update_percentage}%** of the words from the current instruction for each task, . 
Preserve as much useful context as possible from the existing prompt.

Generate exactly ONE improved candidate prompt **as a single JSON dictionary only**. 
Do not include any text outside the JSON.

The JSON structure should be:
{{
  "instructions": {{
{example_instruction}
  }}
}}
            """.strip()

        return meta_prompt

    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
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
        json_str = response.strip().replace("{{", "{").replace("}}", "}")
        start = json_str.find("{")
        end = json_str.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(
                f"[GPOOptimizer] No JSON object found in LLM output:\n{response}"
            )

        json_str = (
            json_str[start:end]
            .replace("\n", " ")
            .strip()
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
        )

        try:
            response_json: Dict[str, Any] = json.loads(json_str)
            candidate: Dict[str, str] = response_json["instructions"]
        except Exception as e:
            raise ValueError(
                f"[GPOOptimizer] Failed to parse JSON: {format_exception_msg(e)}. Response:\n{response}\nExtracted string:\n{json_str}"
            )

        if not isinstance(candidate, dict):
            raise ValueError(
                f"[GPOOptimizer] Failed to parse candidate instructions from response: {response}"
            )

        # Return instructions for each task
        return {
            task.task_name: candidate.get(task.task_name, task.task_instruction)
            for task in tasks
        }

    def _format_gradients_for_meta_prompt(
        self, gradients: Dict[Task, List[TextGradient]]
    ) -> str:
        """Format gradients into readable text for meta-prompt.

        Args:
            gradients: Dict mapping tasks to their gradients

        Returns:
            Formatted string of gradients
        """
        formatted: List[str] = []
        for task, grad_list in gradients.items():
            gtext = " ".join(g.gradient_text for g in grad_list)
            formatted.append(f"Task {task.task_name}: {gtext}")
        return "\n".join(formatted)


class TextGradOptimizer(PromptOptimizer):
    """
    TextGrad-Specific Prompt Optimizer:
    1. Takes textual gradients from gradient computer
    2. Generates updated instruction based on gradients
    3. Returns new prompt with improved instructions

    Unlike GPO, TextGrad focuses on direct instruction updates based on
    natural language criticism without trajectory or step size scheduling.
    """

    aliases: ClassVar[List[str]] = ["textgrad"]

    def create_meta_prompt(
        self,
        *,
        gradients: Dict[Task, List[TextGradient]],
        current_prompt: PromptTemplate,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Create TextGrad meta-prompt based on current instructions and gradients.

        Args:
            gradients: Textual gradients for each task
            current_prompt: Current prompt template
            tasks: List of tasks
            **kwargs: Unused

        Returns:
            Meta-prompt string
        """
        # Format gradients for meta-prompt
        gradient_text = self._format_gradients_for_meta_prompt(gradients)

        # Get current instructions
        current_instructions = current_prompt.instruction
        current_instructions_text = "\n".join(
            [
                f'  "{task.task_name}": "{current_instructions.get(task.task_name, task.task_instruction)}"'
                for task in tasks
            ]
        )

        # Example instruction placeholders
        example_instruction = ",\n".join(
            [
                f'        "{task.task_name}": "Improved instruction for {task.task_name} based on the feedback provided."'
                for task in tasks
            ]
        )

        meta_prompt = f"""
You are an expert prompt engineer. You are optimizing task instructions based on improvement suggestions.

**Current Instructions:**
{{
{current_instructions_text}
}}

**Improvement Suggestions:**
{gradient_text}

Based on the suggestions above, generate improved instructions for each task.
Update the instructions to address the issues and incorporate the recommendations.

Generate exactly ONE improved instruction set **as a single JSON dictionary only**.
Do not include any text outside the JSON.

The JSON structure should be:
{{
  "instructions": {{
{example_instruction}
  }}
}}
        """.strip()

        return meta_prompt

    def parse_meta_prompt_response(
        self,
        *,
        response: str,
        tasks: List[Task],
        **kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Parse TextGrad meta-prompt response.

        Args:
            response: Raw LLM response
            tasks: List of tasks
            **kwargs: Unused

        Returns:
            Dict mapping task names to new instructions

        Raises:
            ValueError: If parsing fails
        """
        json_str = response.strip().replace("{{", "{").replace("}}", "}")
        start = json_str.find("{")
        end = json_str.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(
                f"[TextGradOptimizer] No JSON object found in LLM output:\n{response}"
            )

        json_str = (
            json_str[start:end]
            .replace("\n", " ")
            .strip()
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
            .removeprefix('"')
            .removesuffix('"')
        )
        try:
            response_json: Dict[str, Any] = json.loads(json_str)
            candidate: Dict[str, str] = response_json["instructions"]
        except Exception as e:
            raise ValueError(
                f"[TextGradOptimizer] Failed to parse JSON: {format_exception_msg(e)}.\nResponse:\n{response}\nExtracted string:\n{json_str}"
            )

        if len(candidate) == 0 or not isinstance(candidate, dict):
            raise ValueError(
                f"[TextGradOptimizer] Failed to parse candidate instructions from response:\n{response}"
            )

        # Return instructions for each task
        return {
            task.task_name: candidate.get(task.task_name, task.task_instruction)
            for task in tasks
        }

    def _format_gradients_for_meta_prompt(
        self,
        gradients: Dict[Task, List[TextGradient]],
    ) -> str:
        """Format gradients into readable text for meta-prompt.

        Args:
            gradients: Dict mapping tasks to their gradients

        Returns:
            Formatted string of gradients
        """
        formatted: List[str] = []
        for task, grad_list in gradients.items():
            gtext = " ".join(g.gradient_text for g in grad_list)
            formatted.append(f"Task '{task.task_name}':\n  {gtext}")
        return "\n\n".join(formatted)
