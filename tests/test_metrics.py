"""
Comprehensive tests for prompt_moo.metrics.

Tests cover:
- Registry lookup (Metric.of) for every alias
- ClassVar properties (optimization_direction, display_score_range)
- Instance construction and serialization (model_dump)
- .display_score formatting for every metric type
- normalized_score direction-awareness
- compute() correctness, edge cases, and error handling
- Integration with NumericFeedback delegation properties
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_moo.metrics import Accuracy, F1, LCE, Metric, Precision, Recall
from prompt_moo.data_structures import NumericFeedback


# =====================================================================
# Registry Lookup
# =====================================================================


class TestMetricRegistryLookup:
    """Verify Metric.of() resolves every alias to the correct subclass."""

    @pytest.mark.parametrize(
        "alias,expected_cls",
        [
            ("accuracy", Accuracy),
            ("acc", Accuracy),
            ("f1", F1),
            ("precision", Precision),
            ("recall", Recall),
        ],
    )
    def test_of_resolves_alias(self, alias: str, expected_cls: type) -> None:
        instance = Metric.of(alias)
        assert isinstance(instance, expected_cls)

    @pytest.mark.parametrize(
        "alias,expected_cls",
        [
            ("lce", LCE),
            ("ce", LCE),
        ],
    )
    def test_of_resolves_lce_alias(self, alias: str, expected_cls: type) -> None:
        instance = Metric.of(alias, num_classes=5)
        assert isinstance(instance, expected_cls)
        assert instance.num_classes == 5

    def test_of_unknown_metric_raises(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            Metric.of("nonexistent_metric_xyz")

    def test_lce_of_without_num_classes_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            Metric.of("lce")

    @pytest.mark.parametrize("alias", ["accuracy", "acc", "f1", "precision", "recall"])
    def test_of_returns_instance_with_default_value(self, alias: str) -> None:
        instance = Metric.of(alias)
        assert instance.value == 0.0

    @pytest.mark.parametrize(
        "alias,expected_cls",
        [
            ("accuracy", Accuracy),
            ("acc", Accuracy),
            ("f1", F1),
            ("precision", Precision),
            ("recall", Recall),
            ("lce", LCE),
            ("ce", LCE),
        ],
    )
    def test_get_subclass_resolves_alias(self, alias: str, expected_cls: type) -> None:
        cls = Metric.get_subclass(alias)
        assert cls is expected_cls


# =====================================================================
# ClassVar Properties
# =====================================================================


class TestMetricClassVars:
    """Verify ClassVar and field defaults on each subclass."""

    @pytest.mark.parametrize(
        "metric_cls,expected_name,expected_direction",
        [
            (Accuracy, "accuracy", "maximize"),
            (F1, "f1", "maximize"),
            (Precision, "precision", "maximize"),
            (Recall, "recall", "maximize"),
            (LCE, "lce", "minimize"),
        ],
    )
    def test_class_level_properties(
        self,
        metric_cls: type,
        expected_name: str,
        expected_direction: str,
    ) -> None:
        extra_kwargs = {"num_classes": 5} if metric_cls is LCE else {}
        instance = metric_cls(value=0.5, **extra_kwargs)
        assert instance.name == expected_name
        assert metric_cls.optimization_direction == expected_direction

    @pytest.mark.parametrize(
        "metric_cls,expected_range",
        [
            (Accuracy, (0, 100)),
            (F1, (0, 100)),
            (Precision, (0, 100)),
            (Recall, (0, 100)),
            (LCE, (0, float("inf"))),
        ],
    )
    def test_display_score_range(self, metric_cls: type, expected_range: tuple) -> None:
        assert metric_cls.display_score_range == expected_range

    @pytest.mark.parametrize(
        "metric_cls,expected_direction",
        [
            (Accuracy, "higher"),
            (F1, "higher"),
            (Precision, "higher"),
            (Recall, "higher"),
            (LCE, "lower"),
        ],
    )
    def test_display_direction(self, metric_cls: type, expected_direction: str) -> None:
        assert metric_cls.display_direction == expected_direction


# =====================================================================
# Instance Construction and Serialization
# =====================================================================


class TestMetricConstruction:
    """Verify instance creation, name auto-population, and model_dump."""

    @pytest.mark.parametrize(
        "metric_cls,expected_name,extra_kwargs",
        [
            (Accuracy, "accuracy", {}),
            (F1, "f1", {}),
            (Precision, "precision", {}),
            (Recall, "recall", {}),
            (LCE, "lce", {"num_classes": 5}),
        ],
    )
    def test_name_auto_populated(
        self, metric_cls: type, expected_name: str, extra_kwargs: dict
    ) -> None:
        instance = metric_cls(value=0.5, **extra_kwargs)
        assert instance.name == expected_name

    def test_name_preserved_when_explicitly_set(self) -> None:
        instance = Accuracy(value=0.5, name="custom_name")
        assert instance.name == "custom_name"

    def test_model_dump_contains_name_and_value(self) -> None:
        instance = Accuracy(value=0.75)
        dumped = instance.model_dump()
        assert dumped["name"] == "accuracy"
        assert dumped["value"] == 0.75
        assert set(dumped.keys()) == {"name", "value", "display_decimals"}

    def test_model_dump_lce(self) -> None:
        instance = LCE(value=1.234, num_classes=5)
        dumped = instance.model_dump()
        assert dumped["name"] == "lce"
        assert dumped["value"] == 1.234
        assert dumped["num_classes"] == 5

    def test_default_value_is_zero(self) -> None:
        instance = Accuracy()
        assert instance.value == 0.0
        assert instance.name == "accuracy"


# =====================================================================
# .display_score Formatting
# =====================================================================


class TestToScoreStr:
    """Verify display formatting for each metric type."""

    @pytest.mark.parametrize(
        "metric_cls,value,expected_str,extra_kwargs",
        [
            (Accuracy, 0.0, "0.0", {}),
            (Accuracy, 0.5, "50.0", {}),
            (Accuracy, 0.7083, "70.8", {}),
            (Accuracy, 1.0, "100.0", {}),
            (Accuracy, 0.123456, "12.3", {}),
            (F1, 0.0, "0.0", {}),
            (F1, 0.333, "33.3", {}),
            (F1, 1.0, "100.0", {}),
            (Precision, 0.856, "85.6", {}),
            (Precision, 0.0, "0.0", {}),
            (Recall, 0.125, "12.5", {}),
            (Recall, 1.0, "100.0", {}),
            (LCE, 0.0, "0.0", {"num_classes": 5}),
            (LCE, 0.342, "0.342", {"num_classes": 5}),
            (LCE, 2.7, "2.7", {"num_classes": 5}),
            (LCE, 1.23456, "1.235", {"num_classes": 5}),
        ],
    )
    def test_formatting(
        self, metric_cls: type, value: float, expected_str: str, extra_kwargs: dict
    ) -> None:
        instance = metric_cls(value=value, **extra_kwargs)
        assert instance.display_score == expected_str

    def test_base_metric_default_formatting(self) -> None:
        """Base Metric.display_score uses 3-decimal rounding when display_decimals=3."""
        instance = Metric(value=1.23456789, name="test", display_decimals=3)
        assert instance.display_score == "1.235"


# =====================================================================
# normalized_score
# =====================================================================


class TestNormalizedScore:
    """Verify direction-awareness of normalized_score."""

    def test_maximize_metric_positive(self) -> None:
        instance = Accuracy(value=0.75)
        assert instance.normalized_score == 0.75

    def test_maximize_metric_zero(self) -> None:
        instance = Accuracy(value=0.0)
        assert instance.normalized_score == 0.0

    def test_minimize_metric_negates(self) -> None:
        instance = LCE(value=0.5, num_classes=5)
        assert instance.normalized_score == -0.5

    def test_minimize_metric_zero(self) -> None:
        instance = LCE(value=0.0, num_classes=5)
        assert instance.normalized_score == 0.0

    def test_minimize_negative_value(self) -> None:
        instance = LCE(value=-0.3, num_classes=5)
        assert instance.normalized_score == pytest.approx(0.3)


# =====================================================================
# compute() — Accuracy
# =====================================================================


class TestAccuracyCompute:
    """Verify Accuracy.compute correctness and edge cases."""

    def test_perfect_accuracy(self) -> None:
        assert Accuracy.compute(y_true=[1, 2, 3], y_pred=[1, 2, 3]) == 1.0

    def test_zero_accuracy(self) -> None:
        assert Accuracy.compute(y_true=[1, 2, 3], y_pred=[4, 5, 6]) == 0.0

    def test_partial_accuracy(self) -> None:
        assert Accuracy.compute(y_true=[1, 2, 3, 4], y_pred=[1, 2, 0, 0]) == 0.5

    def test_empty_lists(self) -> None:
        assert Accuracy.compute(y_true=[], y_pred=[]) == 0.0

    def test_single_element_correct(self) -> None:
        assert Accuracy.compute(y_true=[5], y_pred=[5]) == 1.0

    def test_single_element_wrong(self) -> None:
        assert Accuracy.compute(y_true=[5], y_pred=[3]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            Accuracy.compute(y_true=[1, 2], y_pred=[1])

    def test_string_labels(self) -> None:
        result = Accuracy.compute(
            y_true=["cat", "dog", "cat"],
            y_pred=["cat", "cat", "cat"],
        )
        assert result == pytest.approx(2.0 / 3.0)

    def test_mixed_types_no_coercion(self) -> None:
        """Integer 1 != string '1' — no implicit coercion."""
        result = Accuracy.compute(y_true=[1, 2], y_pred=["1", "2"])
        assert result == 0.0


# =====================================================================
# compute() — F1
# =====================================================================


class TestF1Compute:
    """Verify F1.compute correctness and edge cases."""

    def test_perfect_f1(self) -> None:
        assert F1.compute(y_true=[1, 2, 3], y_pred=[1, 2, 3]) == 1.0

    def test_zero_f1(self) -> None:
        result = F1.compute(y_true=[1, 1, 1], y_pred=[2, 2, 2])
        assert result == 0.0

    def test_empty_lists(self) -> None:
        assert F1.compute(y_true=[], y_pred=[]) == 0.0

    def test_nan_values_filtered(self) -> None:
        result = F1.compute(
            y_true=[1, np.nan, 2, 1],
            y_pred=[1, 2, np.nan, 1],
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_all_nan_returns_zero(self) -> None:
        result = F1.compute(
            y_true=[np.nan, np.nan],
            y_pred=[np.nan, np.nan],
        )
        assert result == 0.0


# =====================================================================
# compute() — Precision
# =====================================================================


class TestPrecisionCompute:
    """Verify Precision.compute correctness and edge cases."""

    def test_perfect_precision(self) -> None:
        assert Precision.compute(y_true=[1, 2, 3], y_pred=[1, 2, 3]) == 1.0

    def test_empty_lists(self) -> None:
        assert Precision.compute(y_true=[], y_pred=[]) == 0.0

    def test_all_false_positives(self) -> None:
        result = Precision.compute(y_true=[1, 1, 1], y_pred=[2, 2, 2])
        assert result == 0.0

    def test_nan_filtered(self) -> None:
        result = Precision.compute(
            y_true=[1, np.nan, 2],
            y_pred=[1, 2, np.nan],
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# =====================================================================
# compute() — Recall
# =====================================================================


class TestRecallCompute:
    """Verify Recall.compute correctness and edge cases."""

    def test_perfect_recall(self) -> None:
        assert Recall.compute(y_true=[1, 2, 3], y_pred=[1, 2, 3]) == 1.0

    def test_empty_lists(self) -> None:
        assert Recall.compute(y_true=[], y_pred=[]) == 0.0

    def test_all_false_negatives(self) -> None:
        result = Recall.compute(y_true=[1, 1, 1], y_pred=[2, 2, 2])
        assert result == 0.0

    def test_nan_filtered(self) -> None:
        result = Recall.compute(
            y_true=[1, np.nan, 2],
            y_pred=[1, 2, np.nan],
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# =====================================================================
# compute() — LCE
# =====================================================================


class TestLCECompute:
    """Verify LCE.compute correctness and edge cases."""

    def test_perfect_prediction_at_max_scale(self) -> None:
        result = LCE.compute(y_true=[5, 5], y_pred=[5, 5], num_classes=5)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_low_prediction_gives_high_loss(self) -> None:
        result = LCE.compute(y_true=[5], y_pred=[1], num_classes=5)
        assert result > 1.0

    def test_empty_lists(self) -> None:
        assert LCE.compute(y_true=[], y_pred=[], num_classes=5) == 0.0

    def test_mid_scale_prediction(self) -> None:
        result = LCE.compute(y_true=[3], y_pred=[3], num_classes=5)
        expected = -math.log(3.0 / 5.0 + 1e-12)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_multiple_predictions_averaged(self) -> None:
        result = LCE.compute(y_true=[5, 5], y_pred=[5, 1], num_classes=5)
        loss_5 = -math.log(5.0 / 5.0 + 1e-12)
        loss_1 = -math.log(1.0 / 5.0 + 1e-12)
        expected = (loss_5 + loss_1) / 2.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_different_num_classes(self) -> None:
        """LCE with 4-class scale (0-3, e.g. BRIGHTER dataset)."""
        result = LCE.compute(y_true=[3], y_pred=[3], num_classes=4)
        expected = -math.log(3.0 / 4.0 + 1e-12)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_num_classes_affects_normalization(self) -> None:
        """Same prediction gives different loss under different scales."""
        loss_5 = LCE.compute(y_true=[3], y_pred=[3], num_classes=5)
        loss_4 = LCE.compute(y_true=[3], y_pred=[3], num_classes=4)
        assert loss_5 > loss_4

    def test_num_classes_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_classes >= 1"):
            LCE.compute(y_true=[1], y_pred=[1], num_classes=0)

    def test_num_classes_required_for_construction(self) -> None:
        """LCE cannot be constructed without num_classes."""
        with pytest.raises((ValueError, TypeError)):
            LCE(value=0.5)


# =====================================================================
# Integration: NumericFeedback delegation
# =====================================================================


class TestNumericFeedbackDelegation:
    """Verify NumericFeedback properties delegate to the Metric instance."""

    def test_accuracy_delegation(self) -> None:
        fb = NumericFeedback(
            task_name="fluency",
            metric=Accuracy(value=0.8),
            aggregated_from_samples=["s1", "s2"],
        )
        assert fb.metric_name == "accuracy"
        assert fb.value == 0.8
        assert fb.optimization_direction == "maximize"
        assert fb.normalized_score == 0.8
        assert fb.display_score == "80.0"

    def test_lce_delegation(self) -> None:
        fb = NumericFeedback(
            task_name="coherence",
            metric=LCE(value=1.5, num_classes=5),
            aggregated_from_samples=["s1"],
        )
        assert fb.metric_name == "lce"
        assert fb.value == 1.5
        assert fb.optimization_direction == "minimize"
        assert fb.normalized_score == -1.5
        assert fb.display_score == "1.5"

    def test_model_dump_nested_metric(self) -> None:
        fb = NumericFeedback(
            task_name="test",
            metric=F1(value=0.65),
            aggregated_from_samples=[],
        )
        dumped = fb.model_dump()
        assert dumped["task_name"] == "test"
        assert dumped["metric"]["name"] == "f1"
        assert dumped["metric"]["value"] == 0.65
        assert dumped["aggregated_from_samples"] == []
        assert "metric_name" not in dumped
        assert "optimization_direction" not in dumped

    def test_precision_delegation(self) -> None:
        fb = NumericFeedback(
            task_name="relevance",
            metric=Precision(value=0.456),
            aggregated_from_samples=["s1"],
        )
        assert fb.metric_name == "precision"
        assert fb.display_score == "45.6"

    def test_recall_delegation(self) -> None:
        fb = NumericFeedback(
            task_name="relevance",
            metric=Recall(value=0.789),
            aggregated_from_samples=["s1"],
        )
        assert fb.metric_name == "recall"
        assert fb.display_score == "78.9"


# =====================================================================
# Edge cases: Metric base class
# =====================================================================


class TestMetricBaseClass:
    """Edge cases for the Metric base class itself."""

    def test_base_compute_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="Metric.compute"):
            Metric.compute(y_true=[1], y_pred=[1])

    def test_base_to_score_str_uses_3_display_decimals(self) -> None:
        m = Metric(value=3.14159, name="test", display_decimals=3)
        assert m.display_score == "3.142"

    def test_negative_value_accuracy_to_score_str(self) -> None:
        """Negative accuracy values are nonsensical but should not crash."""
        m = Accuracy(value=-0.5)
        assert m.display_score == "-50.0"

    def test_very_large_value(self) -> None:
        m = Accuracy(value=999.99)
        assert m.display_score == "99999.0"

    def test_value_coercion_from_int(self) -> None:
        """Pydantic coerces int to float."""
        m = Accuracy(value=1)
        assert isinstance(m.value, float)
        assert m.value == 1.0
