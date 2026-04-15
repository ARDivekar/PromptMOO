"""
Comprehensive tests for the PromptTrajectory system.

Covers:
- NumericFeedback: normalized_score, display_score edge cases
- TrajectoryElement.ranking_key: direction, weights, lexicographic, empty, single-task
- PromptTrajectory: ordering, eviction, validation, empty trajectory, k=1
- OPROTrajectoryElement.__str__: exact output format verification
- GPOTrajectoryElement.__str__: exact output format
- get_top_k_str: multi-element output in correct order
- Stability: elements with identical scores retain insertion order
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_moo.algorithm.gpo import GPOTrajectoryElement
from prompt_moo.algorithm.opro import OPROTrajectoryElement
from prompt_moo.data_structures import NumericFeedback
from prompt_moo.metrics import Metric
from prompt_moo.prompt_trajectory import (
    PromptTrajectory,
    TrajectoryElement,
)


class DummyTrajectoryElement(TrajectoryElement):
    """Minimal concrete subclass for testing base TrajectoryElement behavior."""

    aliases = ["dummy"]

    def __str__(self) -> str:
        lines = []
        if isinstance(self.instructions, str):
            lines.append(f"Prompt: {self.instructions}")
            for task_name, feedbacks in self.numeric_scores.items():
                for fb in feedbacks:
                    lines.append(f"Score: {fb.display_score}")
        else:
            for task_name, instruction_text in self.instructions.items():
                lines.append(f"{task_name}:")
                lines.append(f"- Prompt: {instruction_text}")
                if task_name in self.numeric_scores:
                    for fb in self.numeric_scores[task_name]:
                        lines.append(f"- Score: {fb.display_score}")
        return "\n".join(lines)


def _fb(
    *,
    task: str = "t",
    metric: str = "accuracy",
    value: float = 0.5,
    **metric_kwargs: Any,
) -> NumericFeedback:
    metric_cls = Metric.get_subclass(metric)
    return NumericFeedback(
        task_name=task,
        metric=metric_cls(value=value, **metric_kwargs),
        aggregated_from_samples=[],
    )


# =====================================================================
# NumericFeedback.normalized_score
# =====================================================================


class TestNormalizedScore:
    def test_maximize_returns_value(self):
        assert _fb(value=0.75).normalized_score == 0.75

    def test_minimize_returns_negated(self):
        assert _fb(metric="lce", num_classes=5, value=2.0).normalized_score == -2.0

    def test_zero_maximize(self):
        assert _fb(value=0.0).normalized_score == 0.0

    def test_zero_minimize(self):
        assert _fb(metric="lce", num_classes=5, value=0.0).normalized_score == 0.0

    def test_negative_value_maximize(self):
        assert _fb(value=-0.5).normalized_score == -0.5

    def test_negative_value_minimize(self):
        assert _fb(metric="lce", num_classes=5, value=-0.5).normalized_score == 0.5


# =====================================================================
# NumericFeedback.display_score
# =====================================================================


class TestToScoreStr:
    def test_accuracy_50_percent(self):
        assert _fb(metric="accuracy", value=0.50).display_score == "50.0"

    def test_accuracy_rounding_up(self):
        assert _fb(metric="accuracy", value=0.7083).display_score == "70.8"

    def test_accuracy_rounding_down(self):
        assert _fb(metric="accuracy", value=0.7049).display_score == "70.5"

    def test_accuracy_100(self):
        assert _fb(metric="accuracy", value=1.0).display_score == "100.0"

    def test_accuracy_0(self):
        assert _fb(metric="accuracy", value=0.0).display_score == "0.0"

    def test_f1(self):
        assert _fb(metric="f1", value=0.333).display_score == "33.3"

    def test_precision(self):
        assert _fb(metric="precision", value=0.856).display_score == "85.6"

    def test_recall(self):
        assert _fb(metric="recall", value=0.125).display_score == "12.5"

    def test_non_percentage_metric_rounds(self):
        assert _fb(metric="lce", num_classes=5, value=2.7).display_score == "2.7"

    def test_acc_alias(self):
        assert _fb(metric="acc", value=0.60).display_score == "60.0"


# =====================================================================
# TrajectoryElement.ranking_key
# =====================================================================


class TestRankingKeyComprehensive:
    def test_empty_numeric_scores(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={},
        )
        key = elem.ranking_key(metric_priority=["accuracy"])
        assert key == (0.0,)

    def test_empty_numeric_scores_multi_metric(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={},
        )
        key = elem.ranking_key(metric_priority=["accuracy", "f1"])
        assert key == (0.0, 0.0)

    def test_single_task_single_metric(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={"t": [_fb(value=0.80)]},
        )
        assert elem.ranking_key(metric_priority=["accuracy"]) == pytest.approx((0.80,))

    def test_single_task_minimize(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={"t": [_fb(metric="lce", num_classes=5, value=1.5)]},
        )
        assert elem.ranking_key(metric_priority=["lce"]) == pytest.approx((-1.5,))

    def test_two_tasks_equal_weight(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "a": [_fb(task="a", value=1.0)],
                "b": [_fb(task="b", value=0.0)],
            },
        )
        assert elem.ranking_key(metric_priority=["accuracy"]) == pytest.approx((0.5,))

    def test_three_tasks_equal_weight(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "a": [_fb(task="a", value=0.90)],
                "b": [_fb(task="b", value=0.60)],
                "c": [_fb(task="c", value=0.30)],
            },
        )
        assert elem.ranking_key(metric_priority=["accuracy"]) == pytest.approx((0.60,))

    def test_custom_weights(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "a": [_fb(task="a", value=1.0)],
                "b": [_fb(task="b", value=0.0)],
            },
        )
        key = elem.ranking_key(
            metric_priority=["accuracy"],
            task_weights={"a": 0.9, "b": 0.1},
        )
        assert key == pytest.approx((0.9,))

    def test_weight_for_missing_task_raises(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "a": [_fb(task="a", value=1.0)],
                "b": [_fb(task="b", value=0.0)],
            },
        )
        with pytest.raises(
            ValueError, match="task 'b' is present in numeric_scores but has no weight"
        ):
            elem.ranking_key(
                metric_priority=["accuracy"],
                task_weights={"a": 1.0},
            )

    def test_lexicographic_primary_wins(self):
        high_acc_low_f1 = DummyTrajectoryElement(
            instructions="a",
            numeric_scores={"t": [_fb(value=0.90), _fb(metric="f1", value=0.10)]},
        )
        low_acc_high_f1 = DummyTrajectoryElement(
            instructions="b",
            numeric_scores={"t": [_fb(value=0.50), _fb(metric="f1", value=0.99)]},
        )
        key_a = high_acc_low_f1.ranking_key(metric_priority=["accuracy", "f1"])
        key_b = low_acc_high_f1.ranking_key(metric_priority=["accuracy", "f1"])
        assert key_a > key_b

    def test_lexicographic_tiebreak_on_secondary(self):
        elem_a = DummyTrajectoryElement(
            instructions="a",
            numeric_scores={"t": [_fb(value=0.70), _fb(metric="f1", value=0.60)]},
        )
        elem_b = DummyTrajectoryElement(
            instructions="b",
            numeric_scores={"t": [_fb(value=0.70), _fb(metric="f1", value=0.40)]},
        )
        key_a = elem_a.ranking_key(metric_priority=["accuracy", "f1"])
        key_b = elem_b.ranking_key(metric_priority=["accuracy", "f1"])
        assert key_a > key_b

    def test_mixed_directions_across_metrics(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "t": [
                    _fb(value=0.80),
                    _fb(metric="lce", num_classes=5, value=0.50),
                ]
            },
        )
        key = elem.ranking_key(metric_priority=["accuracy", "lce"])
        assert key == pytest.approx((0.80, -0.50))

    def test_mixed_directions_across_tasks(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "quality": [_fb(task="quality", value=0.80)],
                "cost": [_fb(task="cost", metric="lce", num_classes=5, value=1.0)],
            },
        )
        key = elem.ranking_key(metric_priority=["accuracy", "lce"])
        assert key == pytest.approx((0.40, -0.50))

    def test_multiple_feedbacks_per_task_averaged(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "t": [
                    _fb(value=0.80),
                    _fb(value=0.60),
                ]
            },
        )
        key = elem.ranking_key(metric_priority=["accuracy"])
        assert key == pytest.approx((0.70,))

    def test_metric_not_in_element_contributes_zero(self):
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={"t": [_fb(value=0.80)]},
        )
        key = elem.ranking_key(metric_priority=["accuracy", "f1"])
        assert key[0] == pytest.approx(0.80)
        assert key[1] == pytest.approx(0.0)

    def test_direct_comparison_raises(self):
        a = DummyTrajectoryElement(
            instructions="a",
            numeric_scores={},
        )
        b = DummyTrajectoryElement(
            instructions="b",
            numeric_scores={},
        )
        with pytest.raises(NotImplementedError):
            _ = a < b


# =====================================================================
# PromptTrajectory construction validation
# =====================================================================


class TestTrajectoryValidation:
    def test_requires_order(self):
        with pytest.raises(Exception):
            PromptTrajectory(k=5, metric_priority="accuracy")

    def test_requires_metric_priority(self):
        with pytest.raises(Exception):
            PromptTrajectory(k=5, order="worst_to_best")

    def test_string_metric_priority_converted(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        assert t.metric_priority == ["accuracy"]

    def test_list_metric_priority_preserved(self):
        t = PromptTrajectory(
            k=5, order="worst_to_best", metric_priority=["accuracy", "f1"]
        )
        assert t.metric_priority == ["accuracy", "f1"]

    def test_task_weights_sum_to_1_exact(self):
        PromptTrajectory(
            k=5,
            order="worst_to_best",
            metric_priority="accuracy",
            task_weights={"a": 0.5, "b": 0.5},
        )

    def test_task_weights_sum_less_than_1_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            PromptTrajectory(
                k=5,
                order="worst_to_best",
                metric_priority="accuracy",
                task_weights={"a": 0.3, "b": 0.3},
            )

    def test_task_weights_sum_greater_than_1_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            PromptTrajectory(
                k=5,
                order="worst_to_best",
                metric_priority="accuracy",
                task_weights={"a": 0.6, "b": 0.6},
            )

    def test_task_weights_none_is_valid(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        assert t.task_weights is None

    def test_k_must_be_positive(self):
        with pytest.raises(Exception):
            PromptTrajectory(k=0, order="worst_to_best", metric_priority="accuracy")


# =====================================================================
# PromptTrajectory push/eviction/ordering
# =====================================================================


class TestTrajectoryPushAndEviction:
    def _elem(self, value: float, instr: str = "x") -> TrajectoryElement:
        return DummyTrajectoryElement(
            instructions=instr,
            numeric_scores={"t": [_fb(value=value)]},
        )

    def test_empty_trajectory(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        assert len(t) == 0
        assert t.get_topk() == []
        assert t.get_top_k_str() == ""

    def test_single_element(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        t.push(self._elem(0.5))
        assert len(t) == 1

    def test_k_equals_1(self):
        t = PromptTrajectory(k=1, order="worst_to_best", metric_priority="accuracy")
        t.push(self._elem(0.3, "low"))
        t.push(self._elem(0.9, "high"))
        t.push(self._elem(0.6, "mid"))
        assert len(t) == 1
        topk = t.get_topk()
        assert topk[0].instructions == "high"

    def test_eviction_removes_worst(self):
        t = PromptTrajectory(k=2, order="worst_to_best", metric_priority="accuracy")
        t.push(self._elem(0.90, "best"))
        t.push(self._elem(0.30, "worst"))
        t.push(self._elem(0.60, "mid"))
        assert len(t) == 2
        instructions = {e.instructions for e in t.get_topk()}
        assert "worst" not in instructions
        assert "best" in instructions
        assert "mid" in instructions

    def test_eviction_with_minimize_metric(self):
        def _loss_elem(val: float, instr: str) -> TrajectoryElement:
            return DummyTrajectoryElement(
                instructions=instr,
                numeric_scores={"t": [_fb(metric="lce", num_classes=5, value=val)]},
            )

        t = PromptTrajectory(k=2, order="worst_to_best", metric_priority="lce")
        t.push(_loss_elem(3.0, "terrible"))
        t.push(_loss_elem(0.5, "good"))
        t.push(_loss_elem(1.0, "ok"))
        assert len(t) == 2
        instructions = {e.instructions for e in t.get_topk()}
        assert "terrible" not in instructions
        assert "good" in instructions

    def test_worst_to_best_order_verified(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for v in [0.40, 0.80, 0.20, 0.60, 1.00]:
            t.push(self._elem(v))
        topk = t.get_topk()
        values = [e.numeric_scores["t"][0].value for e in topk]
        assert values == sorted(values)

    def test_best_to_worst_order_verified(self):
        t = PromptTrajectory(k=10, order="best_to_worst", metric_priority="accuracy")
        for v in [0.40, 0.80, 0.20, 0.60, 1.00]:
            t.push(self._elem(v))
        topk = t.get_topk()
        values = [e.numeric_scores["t"][0].value for e in topk]
        assert values == sorted(values, reverse=True)

    def test_stability_equal_scores_preserve_insertion_order(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for i in range(5):
            t.push(
                DummyTrajectoryElement(
                    instructions=f"elem_{i}",
                    numeric_scores={"t": [_fb(value=0.50)]},
                )
            )
        topk = t.get_topk()
        instructions = [e.instructions for e in topk]
        assert instructions == [f"elem_{i}" for i in range(5)]

    def test_push_validates_missing_metric(self):
        t = PromptTrajectory(
            k=5, order="worst_to_best", metric_priority=["accuracy", "f1"]
        )
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={"t": [_fb(value=0.5)]},
        )
        with pytest.raises(ValueError, match="metric_priority"):
            t.push(elem)

    def test_push_allows_empty_numeric_scores(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={},
        )
        t.push(elem)
        assert len(t) == 1

    def test_push_with_task_weights(self):
        t = PromptTrajectory(
            k=5,
            order="worst_to_best",
            metric_priority="accuracy",
            task_weights={"a": 0.8, "b": 0.2},
        )
        elem = DummyTrajectoryElement(
            instructions="x",
            numeric_scores={
                "a": [_fb(task="a", value=1.0)],
                "b": [_fb(task="b", value=0.0)],
            },
        )
        t.push(elem)
        assert len(t) == 1


# =====================================================================
# OPROTrajectoryElement.__str__ exact output
# =====================================================================


class TestOPROTrajectoryStr:
    def test_single_task_single_metric(self):
        elem = OPROTrajectoryElement(
            instructions={"fluency": "Rate fluency carefully."},
            numeric_scores={"fluency": [_fb(task="fluency", value=0.75)]},
        )
        s = str(elem)
        assert s == ("fluency:\n- Instruction: Rate fluency carefully.\n- Score: 75.0")

    def test_multi_task(self):
        elem = OPROTrajectoryElement(
            instructions={"fluency": "Rate fluency.", "consistency": "Check facts."},
            numeric_scores={
                "fluency": [_fb(task="fluency", value=0.50)],
                "consistency": [_fb(task="consistency", value=0.80)],
            },
        )
        s = str(elem)
        assert "- Score: 50" in s
        assert "- Score: 80" in s
        assert "- Instruction: Rate fluency." in s
        assert "- Instruction: Check facts." in s

    def test_multiple_metrics_per_task(self):
        elem = OPROTrajectoryElement(
            instructions="Rate it.",
            numeric_scores={
                "t": [
                    _fb(metric="accuracy", value=0.70),
                    _fb(metric="f1", value=0.45),
                ]
            },
        )
        s = str(elem)
        assert "Score: 70" in s
        assert "Score: 45" in s

    def test_string_instructions(self):
        elem = OPROTrajectoryElement(
            instructions="Just a plain string.",
            numeric_scores={"t": [_fb(value=0.60)]},
        )
        s = str(elem)
        assert s.startswith("Instruction: Just a plain string.")

    def test_zero_score(self):
        elem = OPROTrajectoryElement(
            instructions={"t": "x"},
            numeric_scores={"t": [_fb(value=0.0)]},
        )
        assert "- Score: 0" in str(elem)

    def test_perfect_score(self):
        elem = OPROTrajectoryElement(
            instructions={"t": "x"},
            numeric_scores={"t": [_fb(value=1.0)]},
        )
        assert "- Score: 100" in str(elem)

    def test_empty_scores(self):
        elem = OPROTrajectoryElement(
            instructions="x",
            numeric_scores={},
        )
        s = str(elem)
        assert "Instruction: x" in s


# =====================================================================
# GPOTrajectoryElement.__str__ exact output
# =====================================================================


class TestGPOTrajectoryStr:
    def test_without_gradients(self):
        elem = GPOTrajectoryElement(
            instructions={"fluency": "Rate fluency."},
            numeric_scores={"fluency": [_fb(task="fluency", value=0.6543)]},
        )
        s = str(elem)
        assert "Scores: fluency=65.4" in s
        assert "- fluency: Rate fluency." in s
        assert s.startswith("Prompt:")

    def test_multi_task_float_format(self):
        elem = GPOTrajectoryElement(
            instructions={"a": "x", "b": "y"},
            numeric_scores={
                "a": [_fb(task="a", value=0.1234)],
                "b": [_fb(task="b", value=0.5678)],
            },
        )
        s = str(elem)
        assert "a=12.3" in s
        assert "b=56.8" in s
        assert "- a: x" in s
        assert "- b: y" in s
        assert s.startswith("Prompt:")

    def test_gpo_uses_aggregate_score_format(self):
        elem = GPOTrajectoryElement(
            instructions="x",
            numeric_scores={"t": [_fb(value=0.75)]},
        )
        s = str(elem)
        assert "Score: 75.0" in s


# =====================================================================
# get_top_k_str: multi-element ordered output
# =====================================================================


class TestGetTopKStr:
    def test_worst_to_best_string_order(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        t.push(
            DummyTrajectoryElement(
                instructions="low",
                numeric_scores={"t": [_fb(value=0.30)]},
            )
        )
        t.push(
            DummyTrajectoryElement(
                instructions="high",
                numeric_scores={"t": [_fb(value=0.90)]},
            )
        )
        text = t.get_top_k_str()
        assert text.index("low") < text.index("high")

    def test_best_to_worst_string_order(self):
        t = PromptTrajectory(k=10, order="best_to_worst", metric_priority="accuracy")
        t.push(
            DummyTrajectoryElement(
                instructions="low",
                numeric_scores={"t": [_fb(value=0.30)]},
            )
        )
        t.push(
            DummyTrajectoryElement(
                instructions="high",
                numeric_scores={"t": [_fb(value=0.90)]},
            )
        )
        text = t.get_top_k_str()
        assert text.index("high") < text.index("low")

    def test_empty_trajectory_returns_empty_string(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        assert t.get_top_k_str() == ""

    def test_single_element_str(self):
        t = PromptTrajectory(k=5, order="worst_to_best", metric_priority="accuracy")
        t.push(
            DummyTrajectoryElement(
                instructions="only one",
                numeric_scores={"t": [_fb(value=0.55)]},
            )
        )
        text = t.get_top_k_str()
        assert "only one" in text
        assert "55" in text

    def test_three_elements_worst_to_best(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for val, name in [(0.50, "mid"), (0.30, "low"), (0.80, "high")]:
            t.push(
                DummyTrajectoryElement(
                    instructions=name,
                    numeric_scores={"t": [_fb(value=val)]},
                )
            )
        text = t.get_top_k_str()
        pos_low = text.index("low")
        pos_mid = text.index("mid")
        pos_high = text.index("high")
        assert pos_low < pos_mid < pos_high

    def test_gpo_elements_in_trajectory(self):
        t = PromptTrajectory(k=5, order="best_to_worst", metric_priority="accuracy")
        t.push(
            GPOTrajectoryElement(
                instructions="good",
                numeric_scores={"t": [_fb(value=0.90)]},
            )
        )
        t.push(
            GPOTrajectoryElement(
                instructions="ok",
                numeric_scores={"t": [_fb(value=0.60)]},
            )
        )
        text = t.get_top_k_str()
        assert text.index("good") < text.index("ok")
        assert "Prompt: good" in text
        assert "Prompt: ok" in text

    def test_minimize_metric_ordering_in_string(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="lce")
        t.push(
            DummyTrajectoryElement(
                instructions="high_loss",
                numeric_scores={"t": [_fb(metric="lce", num_classes=5, value=5.0)]},
            )
        )
        t.push(
            DummyTrajectoryElement(
                instructions="low_loss",
                numeric_scores={"t": [_fb(metric="lce", num_classes=5, value=0.5)]},
            )
        )
        text = t.get_top_k_str()
        assert text.index("high_loss") < text.index("low_loss")


# =====================================================================
# PromptTrajectory.get_topk(limit=N) pagination
# =====================================================================


class TestGetTopKWithLimit:
    """Test get_topk(limit=N) where N < heap size."""

    def _elem(self, value: float, instr: str = "x") -> TrajectoryElement:
        return DummyTrajectoryElement(
            instructions=instr,
            numeric_scores={"t": [_fb(value=value)]},
        )

    def test_limit_returns_top_n_elements(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for v in [0.10, 0.30, 0.50, 0.70, 0.90]:
            t.push(self._elem(v, f"v{v}"))

        topk = t.get_topk(limit=3)
        assert len(topk) == 3
        values = [e.numeric_scores["t"][0].value for e in topk]
        assert 0.90 in values
        assert 0.70 in values
        assert 0.50 in values

    def test_limit_respects_order_worst_to_best(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for v in [0.10, 0.30, 0.50, 0.70, 0.90]:
            t.push(self._elem(v))

        topk = t.get_topk(limit=3)
        values = [e.numeric_scores["t"][0].value for e in topk]
        assert values == sorted(values)

    def test_limit_respects_order_best_to_worst(self):
        t = PromptTrajectory(k=10, order="best_to_worst", metric_priority="accuracy")
        for v in [0.10, 0.30, 0.50, 0.70, 0.90]:
            t.push(self._elem(v))

        topk = t.get_topk(limit=3)
        values = [e.numeric_scores["t"][0].value for e in topk]
        assert values == sorted(values, reverse=True)

    def test_limit_none_returns_all(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for v in [0.10, 0.30, 0.50]:
            t.push(self._elem(v))

        topk = t.get_topk(limit=None)
        assert len(topk) == 3

    def test_limit_greater_than_heap_returns_all(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for v in [0.10, 0.30]:
            t.push(self._elem(v))

        topk = t.get_topk(limit=10)
        assert len(topk) == 2

    def test_limit_1_returns_best(self):
        t = PromptTrajectory(k=10, order="worst_to_best", metric_priority="accuracy")
        for v in [0.10, 0.90, 0.50]:
            t.push(self._elem(v))

        topk = t.get_topk(limit=1)
        assert len(topk) == 1
        assert topk[0].numeric_scores["t"][0].value == 0.90
