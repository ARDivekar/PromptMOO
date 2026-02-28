"""
Prompt Template Utilities: Template classes for single and multi-objective prompts.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Union

from morphic import Registry, Typed
from morphic.autoenum import AutoEnum

from .tasks_utils import Task


class PromptTemplate(Typed, Registry, ABC):
    """Base class for prompt templates.

    Supports both single-objective and multi-objective prompt formats.
    """

    _allow_subclass_override: ClassVar[bool] = True

    skeleton: str

    @abstractmethod
    def build(self) -> Dict[str, Any]:
        """Build the template as a dictionary.

        Returns:
            Dictionary representation of the template
        """
        pass

    @abstractmethod
    def to_str(self) -> str:
        """Convert the template to a string.

        Returns:
            String representation of the complete prompt
        """
        pass


class UniObjectivePromptTemplate(PromptTemplate):
    """Single-objective prompt template for one task."""

    aliases: ClassVar[List[str]] = ["uni", "single"]

    instruction: str
    task: str

    def build(self) -> Dict[str, str]:
        """Build the template as a dictionary.

        Returns:
            Dictionary with skeleton and instruction
        """
        return {"skeleton": self.skeleton, "instruction": self.instruction}

    def to_str(self) -> str:
        """Convert the template to a string.

        Returns:
            Formatted prompt string
        """
        return (
            f"{self.skeleton}\n\n"
            f"## Task:\n {self.task}\n\n"
            f"## Instruction:\n {self.instruction}\n"
        )


class MultiObjectivePromptTemplate(PromptTemplate):
    """Multi-objective prompt template for multiple tasks."""

    aliases: ClassVar[List[str]] = ["multi", "multiple"]

    tasks: List[Task]
    instruction: Dict[str, str]

    def build(self) -> Dict[str, Union[str, List[Dict[str, str]], Dict[str, str]]]:
        """Build the template as a dictionary.

        Returns:
            Dictionary with skeleton, tasks, and instruction
        """
        return {
            "skeleton": self.skeleton,
            "tasks": [t.to_dict() for t in self.tasks],
            "instruction": self.instruction,
        }

    def to_str(self) -> str:
        """Convert the template to a string.

        Returns:
            Formatted prompt string with all tasks and instructions
        """
        tasks_str = "\n".join([f"- {str(t)}" for t in self.tasks])
        instr_str = "\n".join(f"{k} : {v}" for k, v in self.instruction.items())
        return (
            f"{self.skeleton.strip()}\n\n"
            f"## Tasks:\n {tasks_str}\n\n"
            f"## Instruction:\n {instr_str}\n"
        )


TemplateTypes = AutoEnum.create("TemplateTypes", ["UNI", "MULTI"])
