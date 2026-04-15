"""
Prompt Template: the single data structure representing a multi-task prompt.

A ``PromptTemplate`` knows how to render itself for every audience in the
pipeline (task LLM, optimizer LLM, gradient/loss LLM, observability logs)
so that callers never need to manually assemble prompt sections.
"""

from typing import Dict, List, Optional

from morphic import Typed, validate
from pydantic import Field

from .data_structures import DatasetSample, Task


class PromptTemplate(Typed):
    """Multi-task prompt template.

    Stores the frozen skeleton (evaluation directive + output format) alongside
    mutable per-task instructions and the column-label mapping needed to render
    sample data.

    Rendering methods:

    - ``render_instructions()`` — skeleton + per-task instructions (no sample
      data).  Used by gradient/loss prompts, observability logging, debug
      printing, and anywhere the instruction text is needed without sample data.
    - ``render_task_prompt(sample=...)`` — full prompt sent to the task LLM:
      instructions + formatted sample data + response marker.
    - ``render_for_optimizer()`` — instructions + placeholder sample section.
      Shown to the optimizer LLM so it sees the full structure without real data.
    - ``render_sample(sample=...)`` — just the formatted input fields for one
      sample, reusable by any component that needs to display sample data.

    Attributes:
        skeleton: Frozen evaluation directive + output constraints + JSON format.
        tasks: Ordered list of Task objects in this template.
        instruction: Mutable per-task instructions, updated by the optimizer.
        input_col_labels: Maps raw column names to human-readable display labels
            for the ``## Sample:`` section.
    """

    skeleton: str
    tasks: List[Task]
    instruction: Dict[str, str]
    input_col_labels: Dict[str, str] = Field(default_factory=dict)
    task_output_formats: Dict[str, str] = Field(default_factory=dict)

    def render_instructions(self, *, task_filter: Optional[str] = None) -> str:
        """Render skeleton + output format + per-task instructions (no sample data).

        Used for: observability logging, prompt-to-prompt diffs, gradient/loss
        prompts where the caller adds its own context, debug printing.

        When ``task_output_formats`` is populated, the ``## Output format``
        section is generated dynamically from this dict (filtered by
        ``task_filter``).  When empty, the skeleton is passed through as-is
        (backward compat for callers that bake the format into the skeleton).

        Args:
            task_filter: When provided, render only the instruction AND output
                format for this task name.  Used by the TextGrad gradient
                prompt in ``separate_tasks`` mode so the gradient LLM sees a
                consistent single-task view of the forward pass (no cross-task
                leakage).  When ``None`` (default), all tasks are rendered.

        Returns:
            Formatted string: skeleton + ``## Output format:`` (if applicable)
            + ``## Instructions:`` + per-task lines.
        """
        if task_filter is not None:
            if task_filter not in self.instruction:
                raise ValueError(
                    f"PromptTemplate.render_instructions: task_filter={task_filter!r} "
                    f"not found in instruction dict. "
                    f"Available: {list(self.instruction.keys())}."
                )
            instr_str: str = f"- {task_filter}: {self.instruction[task_filter]}"
        else:
            instr_lines: List[str] = []
            for task in self.tasks:
                instr_text: str = self.instruction[task.task_name]
                instr_lines.append(f"- {task.task_name}: {instr_text}")
            instr_str = "\n".join(instr_lines)

        parts: List[str] = [self.skeleton.strip()]

        if len(self.task_output_formats) > 0:
            if task_filter is not None:
                if task_filter not in self.task_output_formats:
                    raise ValueError(
                        f"PromptTemplate.render_instructions: task_filter={task_filter!r} "
                        f"not found in task_output_formats. "
                        f"Available: {list(self.task_output_formats.keys())}."
                    )
                json_lines: List[str] = [
                    f'  "{task_filter}": {self.task_output_formats[task_filter]}'
                ]
            else:
                json_lines = [
                    f'  "{t.task_name}": {self.task_output_formats[t.task_name]}'
                    for t in self.tasks
                    if t.task_name in self.task_output_formats
                ]
            json_format: str = "{\n" + ",\n".join(json_lines) + "\n}"
            parts.append(f"\n## Output format (follow this EXACTLY):\n{json_format}")

        parts.append(f"\n## Instructions:\n{instr_str}\n")

        return "\n".join(parts)

    @validate
    def render_sample(self, *, sample: DatasetSample) -> str:
        """Format a single sample's input fields using ``input_col_labels``.

        Produces lines of ``{label}: {value}`` for each input column, with
        raw column names mapped to display labels when available.

        Args:
            sample: The dataset sample whose inputs to render.

        Returns:
            Multi-line string with one ``{label}: {value}`` per input column.
        """
        lines: List[str] = []
        for col, val in sample.inputs.items():
            label: str = self.input_col_labels[col]
            lines.append(f"{label}: {val}")
        return "\n".join(lines)

    @validate
    def render_task_prompt(self, *, sample: DatasetSample) -> str:
        """Full prompt sent to the task LLM for one sample.

        Concatenates: instructions + ``## Sample:`` with real data +
        ``## Response:`` marker.  This is the complete, final string; no
        caller assembly is needed.

        Args:
            sample: The dataset sample to include in the prompt.

        Returns:
            Complete prompt string ready for the task LLM.
        """
        sample_text: str = self.render_sample(sample=sample)
        return (
            f"{self.render_instructions()}\n\n"
            f"## Sample:\n"
            f"{sample_text}\n"
            f"\n## Response:\n"
        )

    def render_for_optimizer(self) -> str:
        """Prompt structure shown to the optimizer LLM in the meta-prompt.

        Identical to ``render_task_prompt`` but with a placeholder instead of
        real sample data.  Shows the optimizer the full structure so it
        understands how its instructions will be used, without exposing actual
        data points (which would cause overfitting).

        Returns:
            Template string with a ``## Sample:`` placeholder.
        """
        return (
            f"{self.render_instructions()}\n"
            f"## Sample:\n"
            f"(input data will be inserted here)\n"
            f"\n## Response:\n"
        )
