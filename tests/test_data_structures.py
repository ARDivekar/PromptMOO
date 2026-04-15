"""
Unit tests for prompt_moo.data_structures.

These are pure unit tests — no network calls, no API keys needed.
"""

import pytest

from prompt_moo.data_structures import (
    Batch,
    CombinedFeedback,
    DatasetSample,
    NumericFeedback,
    OptimizerResult,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from prompt_moo.metrics import Accuracy
from prompt_moo.prompt_template import PromptTemplate


@pytest.mark.unit
class TestTask:
    """Tests for the Task data structure."""

    def test_create_task(self):
        task = Task(
            task_name="fluency",
            task_description="1-5 grammar score",
            task_instruction="Check grammar and readability",
            gt_col="fluency",
        )
        assert task.task_name == "fluency"
        assert task.gt_col == "fluency"

    def test_task_equality_by_name(self):
        """Tasks with the same name are equal, even with different instructions."""
        t1 = Task(
            task_name="flu", task_description="d1", task_instruction="i1", gt_col="c1"
        )
        t2 = Task(
            task_name="flu", task_description="d2", task_instruction="i2", gt_col="c2"
        )
        assert t1 == t2

    def test_task_inequality(self):
        t1 = Task(
            task_name="flu", task_description="d", task_instruction="i", gt_col="c"
        )
        t2 = Task(
            task_name="coh", task_description="d", task_instruction="i", gt_col="c"
        )
        assert t1 != t2

    def test_task_hashable(self):
        """Tasks can be used as dict keys and set members."""
        t1 = Task(
            task_name="flu", task_description="d1", task_instruction="i1", gt_col="c1"
        )
        t2 = Task(
            task_name="flu", task_description="d2", task_instruction="i2", gt_col="c2"
        )
        assert hash(t1) == hash(t2)
        assert len({t1, t2}) == 1

    def test_task_str(self):
        task = Task(
            task_name="flu",
            task_description="desc",
            task_instruction="instr",
            gt_col="c",
        )
        s = str(task)
        assert "flu" in s
        assert "desc" in s
        assert "instr" in s

    def test_task_model_dump(self):
        task = Task(
            task_name="flu",
            task_description="desc",
            task_instruction="instr",
            gt_col="c",
        )
        d = task.model_dump()
        assert d["task_name"] == "flu"
        assert d["task_description"] == "desc"
        assert d["task_instruction"] == "instr"


@pytest.mark.unit
class TestDatasetSample:
    def test_create_sample(self):
        sample = DatasetSample(
            sample_id="s1",
            inputs={"text": "hello"},
            ground_truths={"fluency": 5},
        )
        assert sample.sample_id == "s1"
        assert sample.inputs["text"] == "hello"
        assert sample.ground_truths["fluency"] == 5


@pytest.mark.unit
class TestBatch:
    def test_create_batch(self):
        samples = [
            DatasetSample(
                sample_id=f"s{i}", inputs={"x": i}, ground_truths={"y": i * 2}
            )
            for i in range(3)
        ]
        batch = Batch(step=0, samples=samples)
        assert batch.step == 0
        assert len(batch.samples) == 3

    def test_empty_batch(self):
        batch = Batch(step=5, samples=[])
        assert len(batch.samples) == 0


@pytest.mark.unit
class TestPredictionResult:
    def test_create_prediction(self):
        pred = PredictionResult(
            sample_id="s1",
            task_outputs={"fluency": 4, "coherence": 3},
            raw_response='{"fluency": 4, "coherence": 3}',
            prompt="evaluate this",
        )
        assert pred.task_outputs["fluency"] == 4
        assert pred.prompt == "evaluate this"

    def test_prediction_parser_error_defaults_to_none(self):
        pred = PredictionResult(
            sample_id="s1",
            prompt="test_prompt",
            task_outputs={"fluency": 5},
            raw_response="raw",
        )
        assert pred.prompt == "test_prompt"
        assert pred.parser_error is None


@pytest.mark.unit
class TestNumericFeedback:
    def test_create_numeric_feedback(self):
        fb = NumericFeedback(
            task_name="fluency",
            metric=Accuracy(value=0.85),
            aggregated_from_samples=["s1", "s2", "s3"],
        )
        assert fb.metric.value == 0.85
        assert fb.metric.optimization_direction == "maximize"
        assert len(fb.aggregated_from_samples) == 3


@pytest.mark.unit
class TestTextualFeedback:
    def test_create_textual_feedback(self):
        fb = TextualFeedback(
            task_name="fluency",
            feedback_text="Responses tend to be grammatically correct but overly verbose.",
            aggregated_from_samples=["s1", "s2"],
            feedback_prompt="Analyze these responses...",
        )
        assert "verbose" in fb.feedback_text
        assert fb.feedback_prompt is not None


@pytest.mark.unit
class TestCombinedFeedback:
    def test_create_combined(self):
        numeric = NumericFeedback(
            task_name="flu",
            metric=Accuracy(value=0.9),
            aggregated_from_samples=["s1"],
        )
        textual = TextualFeedback(
            task_name="flu",
            feedback_text="Good grammar.",
            aggregated_from_samples=["s1"],
            feedback_prompt="p",
        )
        combined = CombinedFeedback(
            task_name="flu",
            numeric_feedbacks=[numeric],
            textual_feedbacks=[textual],
            aggregated_from_samples=["s1"],
        )
        assert len(combined.numeric_feedbacks) == 1
        assert len(combined.textual_feedbacks) == 1


@pytest.mark.unit
class TestTextGradient:
    def test_create_gradient(self):
        grad = TextGradient(
            task_name="fluency",
            gradient_text="Focus more on sentence variety and flow.",
            based_on_feedbacks=["fb1", "fb2"],
            gradient_prompt="Based on the feedback...",
        )
        assert "variety" in grad.gradient_text


@pytest.mark.unit
class TestOptimizerResult:
    def test_create_optimizer_result(self):
        prompt: PromptTemplate = PromptTemplate(
            skeleton="Evaluate.",
            tasks=[],
            instruction={},
        )
        result = OptimizerResult(
            new_prompt=prompt,
            meta_prompt="meta prompt text",
            raw_response="raw optimizer response",
        )
        assert result.new_prompt == prompt
