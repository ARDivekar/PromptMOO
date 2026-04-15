"""
Task Output Specification: describes the valid output format for a single task.

Each ``TaskOutputSpec`` subclass knows:
- What values are valid (integers 1-5, labels "Cat"/"Dog", floats -1 to 1)
- How to render them in a JSON example (``format_str``)
- How to describe them in natural language (``description_str``)
- How many classes there are (``num_classes``, for metric computation)
- What noun to use for the output (``output_noun``: "score", "label", "value")

Usage::

    from prompt_moo.task_output_spec import TaskOutputSpec

    spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
    spec.format_str       # "1|2|3|4|5"
    spec.description_str  # "single integer between 1 and 5"
    spec.num_classes       # 5
    spec.output_noun       # "score"

    spec = TaskOutputSpec.categorical(labels=["harmful", "unharmful"])
    spec.format_str       # '"harmful"|"unharmful"'
    spec.description_str  # 'one of "harmful" or "unharmful"'
    spec.num_classes       # 2
    spec.output_noun       # "label"
"""

from abc import ABC
from typing import Callable, ClassVar, Dict, List, Optional

from morphic import Registry, Typed


class TaskOutputSpec(Typed, Registry, ABC):
    """Base class for task output format specifications.

    Subclasses define the valid values and how to present them to the LLM.

    Attributes:
        format_fn: Optional callable ``(TaskOutputSpec) -> str`` that
            overrides ``description_str`` when provided.  The callable
            receives the spec instance as its sole argument.
        output_noun_override: Optional override for ``output_noun``.
            When ``None``, the subclass default is used.
    """

    _allow_subclass_override: ClassVar[bool] = True

    format_fn: Optional[Callable] = None
    output_noun_override: Optional[str] = None

    @property
    def format_str(self) -> str:
        """Pipe-separated valid values for the JSON example line.

        Example: ``"1|2|3|4|5"`` or ``'"harmful"|"unharmful"'``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement format_str"
        )

    @property
    def description_str(self) -> str:
        """Natural-language description for the instruction line.

        Example: ``"single integer between 1 and 5"`` or
        ``'one of "harmful" or "unharmful"'``.

        When ``_format_fn`` is set, it is called with ``self`` and its
        return value is used instead of the subclass default.
        """
        if self.format_fn is not None:
            return self.format_fn(self)
        return self._default_description_str()

    def _default_description_str(self) -> str:
        """Subclass-specific default description. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _default_description_str"
        )

    @property
    def num_classes(self) -> int:
        """Number of discrete classes. 0 for continuous ranges."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement num_classes"
        )

    @property
    def output_noun(self) -> str:
        """Noun describing the output: "score", "label", or "value".

        Used in prompt text like "Output {output_noun} for the following tasks".
        """
        if self.output_noun_override is not None:
            return self.output_noun_override
        return self._default_output_noun()

    def _default_output_noun(self) -> str:
        """Subclass-specific default noun. Override in subclasses."""
        return "value"

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------
    @classmethod
    def ordinal_int(
        cls,
        *,
        min_val: int,
        max_val: int,
        format_fn: Optional[Callable] = None,
        output_noun: Optional[str] = None,
    ) -> "OrdinalIntOutputSpec":
        """Ordinal integer scale (e.g. 1-5 rating).

        Args:
            min_val: Minimum valid integer (inclusive).
            max_val: Maximum valid integer (inclusive).
            format_fn: Optional ``(spec) -> str`` to override description_str.
            output_noun: Optional override for output_noun (default: "score").
        """
        return OrdinalIntOutputSpec(
            min_val=min_val,
            max_val=max_val,
            format_fn=format_fn,
            output_noun_override=output_noun,
        )

    @classmethod
    def ordinal_str(
        cls,
        *,
        ordered_labels: List[str],
        format_fn: Optional[Callable] = None,
        output_noun: Optional[str] = None,
    ) -> "OrdinalStrOutputSpec":
        """Ordinal string scale (e.g. "Low", "Medium", "High").

        Args:
            ordered_labels: Labels in ascending order.
            format_fn: Optional ``(spec) -> str`` to override description_str.
            output_noun: Optional override for output_noun (default: "rating").
        """
        return OrdinalStrOutputSpec(
            ordered_labels=ordered_labels,
            format_fn=format_fn,
            output_noun_override=output_noun,
        )

    @classmethod
    def categorical(
        cls,
        *,
        labels: List[str],
        format_fn: Optional[Callable] = None,
        output_noun: Optional[str] = None,
    ) -> "CategoricalOutputSpec":
        """Unordered categorical labels (e.g. "Cat", "Dog", "Rabbit").

        Args:
            labels: Valid label strings (order in list = order in prompt).
            format_fn: Optional ``(spec) -> str`` to override description_str.
            output_noun: Optional override for output_noun (default: "label").
        """
        return CategoricalOutputSpec(
            labels=labels,
            format_fn=format_fn,
            output_noun_override=output_noun,
        )

    @classmethod
    def binary(
        cls,
        *,
        true_label: str,
        false_label: str,
        format_fn: Optional[Callable] = None,
        output_noun: Optional[str] = None,
    ) -> "BinaryOutputSpec":
        """Binary classification (e.g. "Y"/"N", "harmful"/"unharmful").

        Args:
            true_label: The positive/true label string.
            false_label: The negative/false label string.
            format_fn: Optional ``(spec) -> str`` to override description_str.
            output_noun: Optional override for output_noun (default: "label").
        """
        return BinaryOutputSpec(
            true_label=true_label,
            false_label=false_label,
            format_fn=format_fn,
            output_noun_override=output_noun,
        )

    @classmethod
    def float_range(
        cls,
        *,
        min_val: float,
        max_val: float,
        format_fn: Optional[Callable] = None,
        output_noun: Optional[str] = None,
    ) -> "FloatRangeOutputSpec":
        """Continuous float range (e.g. -1.0 to 1.0).

        Args:
            min_val: Minimum valid value (inclusive).
            max_val: Maximum valid value (inclusive).
            format_fn: Optional ``(spec) -> str`` to override description_str.
            output_noun: Optional override for output_noun (default: "value").
        """
        return FloatRangeOutputSpec(
            min_val=min_val,
            max_val=max_val,
            format_fn=format_fn,
            output_noun_override=output_noun,
        )


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


class OrdinalIntOutputSpec(TaskOutputSpec):
    """Ordinal integer scale: 1, 2, 3, 4, 5."""

    aliases: ClassVar[List[str]] = ["ordinal_int", "ordinal-int", "int"]

    min_val: int
    max_val: int

    @property
    def format_str(self) -> str:
        return "|".join(str(v) for v in range(self.min_val, self.max_val + 1))

    def _default_description_str(self) -> str:
        return f"single integer between {self.min_val} and {self.max_val}"

    @property
    def num_classes(self) -> int:
        return self.max_val - self.min_val + 1

    def _default_output_noun(self) -> str:
        return "score"


class OrdinalStrOutputSpec(TaskOutputSpec):
    """Ordinal string scale: "Low", "Medium", "High"."""

    aliases: ClassVar[List[str]] = ["ordinal_str", "ordinal-str"]

    ordered_labels: List[str]

    @property
    def format_str(self) -> str:
        return "|".join(f'"{label}"' for label in self.ordered_labels)

    def _default_description_str(self) -> str:
        quoted = [f'"{label}"' for label in self.ordered_labels]
        if len(quoted) <= 2:
            return "one of " + " or ".join(quoted)
        return "one of " + ", ".join(quoted[:-1]) + ", or " + quoted[-1]

    @property
    def num_classes(self) -> int:
        return len(self.ordered_labels)

    def _default_output_noun(self) -> str:
        return "rating"


class CategoricalOutputSpec(TaskOutputSpec):
    """Unordered categorical labels: "Cat", "Dog", "Rabbit"."""

    aliases: ClassVar[List[str]] = ["categorical", "multiclass"]

    labels: List[str]

    @property
    def format_str(self) -> str:
        return "|".join(f'"{label}"' for label in self.labels)

    def _default_description_str(self) -> str:
        quoted = [f'"{label}"' for label in self.labels]
        if len(quoted) <= 2:
            return "one of " + " or ".join(quoted)
        return "one of " + ", ".join(quoted[:-1]) + ", or " + quoted[-1]

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def _default_output_noun(self) -> str:
        return "label"


class BinaryOutputSpec(TaskOutputSpec):
    """Binary classification: "harmful"/"unharmful", "Y"/"N"."""

    aliases: ClassVar[List[str]] = ["binary"]

    true_label: str
    false_label: str

    @property
    def format_str(self) -> str:
        return f'"{self.true_label}"|"{self.false_label}"'

    def _default_description_str(self) -> str:
        return f'"{self.true_label}" or "{self.false_label}"'

    @property
    def num_classes(self) -> int:
        return 2

    def _default_output_noun(self) -> str:
        return "label"


class FloatRangeOutputSpec(TaskOutputSpec):
    """Continuous float range: -1.0 to 1.0."""

    aliases: ClassVar[List[str]] = ["float_range", "float-range", "float", "continuous"]

    min_val: float
    max_val: float

    @property
    def format_str(self) -> str:
        return f"<{self.min_val} to {self.max_val}>"

    def _default_description_str(self) -> str:
        return f"decimal value between {self.min_val} and {self.max_val}"

    @property
    def num_classes(self) -> int:
        return 0

    def _default_output_noun(self) -> str:
        return "value"


# ---------------------------------------------------------------------------
# Utility: derive output_noun for a collection of specs
# ---------------------------------------------------------------------------
def collective_output_noun(specs: Dict[str, TaskOutputSpec]) -> str:
    """Determine the collective noun for a set of task output specs.

    Returns "scores" if all tasks are ordinal (int or str), "labels" if all
    are categorical/binary, or "values" for mixed or continuous types.

    Args:
        specs: Mapping of task_name -> TaskOutputSpec.

    Returns:
        Plural noun string: "scores", "labels", or "values".
    """
    nouns = {spec.output_noun for spec in specs.values()}
    if len(nouns) == 1:
        noun = nouns.pop()
        return noun + "s"
    return "values"
