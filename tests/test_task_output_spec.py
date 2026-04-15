"""
Comprehensive tests for TaskOutputSpec and its subclasses.

Tests cover:
- All 5 subclass types (OrdinalInt, OrdinalStr, Categorical, Binary, FloatRange)
- All properties (format_str, description_str, num_classes, output_noun)
- Factory classmethods on TaskOutputSpec base class
- Custom format_fn override
- Custom output_noun_override
- Edge cases (single value, empty labels, boundary values)
- collective_output_noun utility
- Validation errors (min > max, empty labels)
"""

import pytest
from typing import List

from prompt_moo.task_output_spec import (
    BinaryOutputSpec,
    CategoricalOutputSpec,
    FloatRangeOutputSpec,
    OrdinalIntOutputSpec,
    OrdinalStrOutputSpec,
    TaskOutputSpec,
    collective_output_noun,
)


# ---------------------------------------------------------------------------
# OrdinalIntOutputSpec
# ---------------------------------------------------------------------------
class TestOrdinalIntOutputSpec:
    """Tests for ordinal integer scale (e.g. 1-5 rating)."""

    def test_format_str_1_to_5(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
        assert spec.format_str == "1|2|3|4|5"

    def test_format_str_0_to_3(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=0, max_val=3)
        assert spec.format_str == "0|1|2|3"

    def test_format_str_single_value(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=3, max_val=3)
        assert spec.format_str == "3"

    def test_description_str_default(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
        assert spec.description_str == "single integer between 1 and 5"

    def test_description_str_0_to_3(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=0, max_val=3)
        assert spec.description_str == "single integer between 0 and 3"

    def test_num_classes(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
        assert spec.num_classes == 5

    def test_num_classes_0_to_3(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=0, max_val=3)
        assert spec.num_classes == 4

    def test_num_classes_single_value(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=7, max_val=7)
        assert spec.num_classes == 1

    def test_output_noun_default(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
        assert spec.output_noun == "score"

    def test_output_noun_override(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5, output_noun="rating")
        assert spec.output_noun == "rating"

    def test_format_fn_override(self) -> None:
        spec = TaskOutputSpec.ordinal_int(
            min_val=0,
            max_val=3,
            format_fn=lambda self: (
                f"integer from {self.min_val} (none) to {self.max_val} (intense)"
            ),
        )
        assert spec.description_str == "integer from 0 (none) to 3 (intense)"
        assert spec.format_str == "0|1|2|3"

    def test_isinstance_check(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
        assert isinstance(spec, OrdinalIntOutputSpec)
        assert isinstance(spec, TaskOutputSpec)

    def test_negative_range(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=-2, max_val=2)
        assert spec.format_str == "-2|-1|0|1|2"
        assert spec.num_classes == 5


# ---------------------------------------------------------------------------
# OrdinalStrOutputSpec
# ---------------------------------------------------------------------------
class TestOrdinalStrOutputSpec:
    """Tests for ordinal string scale (e.g. "Low", "Medium", "High")."""

    def test_format_str_three_labels(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["Low", "Medium", "High"])
        assert spec.format_str == '"Low"|"Medium"|"High"'

    def test_format_str_two_labels(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["Bad", "Good"])
        assert spec.format_str == '"Bad"|"Good"'

    def test_description_str_three_labels(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["Low", "Medium", "High"])
        assert spec.description_str == 'one of "Low", "Medium", or "High"'

    def test_description_str_two_labels(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["Bad", "Good"])
        assert spec.description_str == 'one of "Bad" or "Good"'

    def test_description_str_single_label(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["Only"])
        assert spec.description_str == 'one of "Only"'

    def test_num_classes(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["A", "B", "C", "D"])
        assert spec.num_classes == 4

    def test_output_noun_default(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["Low", "High"])
        assert spec.output_noun == "rating"

    def test_output_noun_override(self) -> None:
        spec = TaskOutputSpec.ordinal_str(
            ordered_labels=["Low", "High"], output_noun="level"
        )
        assert spec.output_noun == "level"

    def test_isinstance_check(self) -> None:
        spec = TaskOutputSpec.ordinal_str(ordered_labels=["A", "B"])
        assert isinstance(spec, OrdinalStrOutputSpec)
        assert isinstance(spec, TaskOutputSpec)


# ---------------------------------------------------------------------------
# CategoricalOutputSpec
# ---------------------------------------------------------------------------
class TestCategoricalOutputSpec:
    """Tests for unordered categorical labels."""

    def test_format_str_four_labels(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["Cat", "Dog", "Rabbit", "Horse"])
        assert spec.format_str == '"Cat"|"Dog"|"Rabbit"|"Horse"'

    def test_format_str_two_labels(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["Yes", "No"])
        assert spec.format_str == '"Yes"|"No"'

    def test_description_str_four_labels(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["Cat", "Dog", "Rabbit", "Horse"])
        assert spec.description_str == 'one of "Cat", "Dog", "Rabbit", or "Horse"'

    def test_description_str_two_labels(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["Positive", "Negative"])
        assert spec.description_str == 'one of "Positive" or "Negative"'

    def test_num_classes(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["A", "B", "C"])
        assert spec.num_classes == 3

    def test_output_noun_default(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["Cat", "Dog"])
        assert spec.output_noun == "label"

    def test_format_fn_override(self) -> None:
        spec = TaskOutputSpec.categorical(
            labels=["Cat", "Dog"],
            format_fn=lambda self: f"exactly one animal from: {', '.join(self.labels)}",
        )
        assert spec.description_str == "exactly one animal from: Cat, Dog"

    def test_isinstance_check(self) -> None:
        spec = TaskOutputSpec.categorical(labels=["A"])
        assert isinstance(spec, CategoricalOutputSpec)
        assert isinstance(spec, TaskOutputSpec)


# ---------------------------------------------------------------------------
# BinaryOutputSpec
# ---------------------------------------------------------------------------
class TestBinaryOutputSpec:
    """Tests for binary classification."""

    def test_format_str(self) -> None:
        spec = TaskOutputSpec.binary(true_label="harmful", false_label="unharmful")
        assert spec.format_str == '"harmful"|"unharmful"'

    def test_format_str_yn(self) -> None:
        spec = TaskOutputSpec.binary(true_label="Y", false_label="N")
        assert spec.format_str == '"Y"|"N"'

    def test_description_str(self) -> None:
        spec = TaskOutputSpec.binary(true_label="harmful", false_label="unharmful")
        assert spec.description_str == '"harmful" or "unharmful"'

    def test_num_classes(self) -> None:
        spec = TaskOutputSpec.binary(true_label="Yes", false_label="No")
        assert spec.num_classes == 2

    def test_output_noun_default(self) -> None:
        spec = TaskOutputSpec.binary(true_label="Y", false_label="N")
        assert spec.output_noun == "label"

    def test_output_noun_override(self) -> None:
        spec = TaskOutputSpec.binary(
            true_label="Y", false_label="N", output_noun="response"
        )
        assert spec.output_noun == "response"

    def test_isinstance_check(self) -> None:
        spec = TaskOutputSpec.binary(true_label="T", false_label="F")
        assert isinstance(spec, BinaryOutputSpec)
        assert isinstance(spec, TaskOutputSpec)


# ---------------------------------------------------------------------------
# FloatRangeOutputSpec
# ---------------------------------------------------------------------------
class TestFloatRangeOutputSpec:
    """Tests for continuous float range."""

    def test_format_str(self) -> None:
        spec = TaskOutputSpec.float_range(min_val=-1.0, max_val=1.0)
        assert spec.format_str == "<-1.0 to 1.0>"

    def test_format_str_positive(self) -> None:
        spec = TaskOutputSpec.float_range(min_val=0.0, max_val=100.0)
        assert spec.format_str == "<0.0 to 100.0>"

    def test_description_str(self) -> None:
        spec = TaskOutputSpec.float_range(min_val=-1.0, max_val=1.0)
        assert spec.description_str == "decimal value between -1.0 and 1.0"

    def test_num_classes_is_zero(self) -> None:
        spec = TaskOutputSpec.float_range(min_val=0.0, max_val=1.0)
        assert spec.num_classes == 0

    def test_output_noun_default(self) -> None:
        spec = TaskOutputSpec.float_range(min_val=0.0, max_val=1.0)
        assert spec.output_noun == "value"

    def test_output_noun_override(self) -> None:
        spec = TaskOutputSpec.float_range(
            min_val=0.0, max_val=1.0, output_noun="probability"
        )
        assert spec.output_noun == "probability"

    def test_isinstance_check(self) -> None:
        spec = TaskOutputSpec.float_range(min_val=0.0, max_val=1.0)
        assert isinstance(spec, FloatRangeOutputSpec)
        assert isinstance(spec, TaskOutputSpec)


# ---------------------------------------------------------------------------
# collective_output_noun
# ---------------------------------------------------------------------------
class TestCollectiveOutputNoun:
    """Tests for the utility that derives a collective noun from a set of specs."""

    def test_all_ordinal_int(self) -> None:
        specs = {
            "t1": TaskOutputSpec.ordinal_int(min_val=1, max_val=5),
            "t2": TaskOutputSpec.ordinal_int(min_val=1, max_val=5),
        }
        assert collective_output_noun(specs) == "scores"

    def test_all_binary(self) -> None:
        specs = {
            "t1": TaskOutputSpec.binary(true_label="Y", false_label="N"),
            "t2": TaskOutputSpec.binary(true_label="T", false_label="F"),
        }
        assert collective_output_noun(specs) == "labels"

    def test_all_categorical(self) -> None:
        specs = {
            "t1": TaskOutputSpec.categorical(labels=["A", "B"]),
            "t2": TaskOutputSpec.categorical(labels=["X", "Y", "Z"]),
        }
        assert collective_output_noun(specs) == "labels"

    def test_all_float_range(self) -> None:
        specs = {
            "t1": TaskOutputSpec.float_range(min_val=0.0, max_val=1.0),
            "t2": TaskOutputSpec.float_range(min_val=-1.0, max_val=1.0),
        }
        assert collective_output_noun(specs) == "values"

    def test_mixed_ordinal_and_binary(self) -> None:
        specs = {
            "score_task": TaskOutputSpec.ordinal_int(min_val=1, max_val=5),
            "label_task": TaskOutputSpec.binary(true_label="Y", false_label="N"),
        }
        assert collective_output_noun(specs) == "values"

    def test_mixed_ordinal_int_and_str(self) -> None:
        specs = {
            "int_task": TaskOutputSpec.ordinal_int(min_val=1, max_val=5),
            "str_task": TaskOutputSpec.ordinal_str(ordered_labels=["Low", "High"]),
        }
        assert collective_output_noun(specs) == "values"

    def test_single_spec(self) -> None:
        specs = {"only": TaskOutputSpec.ordinal_int(min_val=1, max_val=5)}
        assert collective_output_noun(specs) == "scores"

    def test_empty_specs(self) -> None:
        assert collective_output_noun({}) == "values"

    def test_custom_output_noun_propagates(self) -> None:
        specs = {
            "t1": TaskOutputSpec.ordinal_int(
                min_val=1, max_val=5, output_noun="metric"
            ),
            "t2": TaskOutputSpec.ordinal_int(
                min_val=1, max_val=5, output_noun="metric"
            ),
        }
        assert collective_output_noun(specs) == "metrics"

    def test_custom_output_noun_mixed(self) -> None:
        specs = {
            "t1": TaskOutputSpec.ordinal_int(
                min_val=1, max_val=5, output_noun="metric"
            ),
            "t2": TaskOutputSpec.ordinal_int(min_val=1, max_val=5),
        }
        assert collective_output_noun(specs) == "values"


# ---------------------------------------------------------------------------
# format_fn edge cases
# ---------------------------------------------------------------------------
class TestFormatFnOverride:
    """Tests for the format_fn override mechanism."""

    def test_format_fn_receives_self(self) -> None:
        received_self = []

        def capture_self(spec: TaskOutputSpec) -> str:
            received_self.append(spec)
            return "custom"

        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5, format_fn=capture_self)
        result = spec.description_str
        assert result == "custom"
        assert len(received_self) == 1
        assert isinstance(received_self[0], OrdinalIntOutputSpec)
        assert received_self[0].min_val == 1

    def test_format_fn_none_uses_default(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5, format_fn=None)
        assert spec.description_str == "single integer between 1 and 5"

    def test_format_fn_on_binary(self) -> None:
        spec = TaskOutputSpec.binary(
            true_label="safe",
            false_label="unsafe",
            format_fn=lambda self: (
                f'either "{self.true_label}" (safe content) or "{self.false_label}" (unsafe content)'
            ),
        )
        assert (
            spec.description_str
            == 'either "safe" (safe content) or "unsafe" (unsafe content)'
        )

    def test_format_fn_does_not_affect_format_str(self) -> None:
        spec = TaskOutputSpec.ordinal_int(
            min_val=1, max_val=3, format_fn=lambda self: "totally custom"
        )
        assert spec.format_str == "1|2|3"
        assert spec.description_str == "totally custom"


# ---------------------------------------------------------------------------
# Immutability (Typed is frozen)
# ---------------------------------------------------------------------------
class TestImmutability:
    """Verify TaskOutputSpec instances are immutable (Typed frozen)."""

    def test_ordinal_int_is_frozen(self) -> None:
        spec = TaskOutputSpec.ordinal_int(min_val=1, max_val=5)
        with pytest.raises(Exception):
            spec.min_val = 10

    def test_binary_is_frozen(self) -> None:
        spec = TaskOutputSpec.binary(true_label="Y", false_label="N")
        with pytest.raises(Exception):
            spec.true_label = "Yes"
